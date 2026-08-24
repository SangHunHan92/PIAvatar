"""hetero_cloth.body_mesh_collider: SMPL-X body mesh as an extra collider for
the cloth subject in our separable-contact pipeline.

What it adds (versus separable-contact alone):
  Separable contact only enforces a NO-TENSION velocity constraint between the
  cloth and body MPM fields. It cannot push particles apart once they overlap,
  so body MPM particles can slowly leak through cloth over many substeps.

  This module ports MPMAvatar's mesh-collider idea (mpm_solver.py:805) and
  applies it ONLY to the cloth subject's resolved grid velocity, after the
  separable-contact resolve step. The kinematic SMPL-X body mesh is rasterized
  into per-grid (mesh_v_out, mesh_normal, weight); cloth's grid velocity at any
  node touching the body mesh is then projected so its inward-normal component
  cannot exceed the body mesh's velocity. This = signed-distance (mesh-based)
  non-penetration for cloth, on top of the existing separable-contact resolve.

Body MPM particles are NOT modified by this collider. We only stop the cloth
from being penetrated; this in turn keeps the cloth visually OUTSIDE the body
mesh so body particles cannot appear "through" cloth at render time.

Usage (from cloth_solver.MPM_Simulator_WARP_with_Cloth):
    self.body_mesh_collider = BodyMeshColliderState(
        verts=initial_verts_torch,        # (V,3) world-coord SMPL-X verts
        faces=faces_torch,                # (F,3) int triangle indices
        n_grid=self.n_grid,
        cloth_subject_id=cloth_subj_id,
        friction=0.1,
        device="cuda:0",
    )
    # per frame:
    self.body_mesh_collider.update_pose(verts_now, verts_next, frame_dt)
    # per substep (called inside Phase D after compute_vcm_and_resolve):
    self.body_mesh_collider.launch(self.mpm_state, self.mpm_model)

The update_pose call refreshes mesh.points + mesh.velocities; the per-substep
launch runs zero->scatter->normalize->collide on the (now-fresh) mesh state.
"""
from __future__ import annotations

import warp as wp
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Mesh collider state (per-grid scratch arrays + the wp.Mesh wrapping the
# kinematic body mesh)
# ---------------------------------------------------------------------------
@wp.struct
class BodyMeshColliderStruct:
    mesh_id: wp.uint64
    friction: float
    cloth_subj_id: int
    weight: wp.array(dtype=float, ndim=3)
    mesh_v_in: wp.array(dtype=wp.vec3, ndim=3)
    mesh_v_out: wp.array(dtype=wp.vec3, ndim=3)
    mesh_normal: wp.array(dtype=wp.vec3, ndim=3)


# ---------------------------------------------------------------------------
# Kernels (ported from MPMAvatar's add_mesh_collider, adapted for our
# separable-contact pipeline's grid_v_resolved[..., k] field)
# ---------------------------------------------------------------------------
@wp.kernel
def zero_body_mesh_grid(param: BodyMeshColliderStruct):
    gx, gy, gz = wp.tid()
    param.weight[gx, gy, gz] = 0.0
    param.mesh_v_in[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
    param.mesh_v_out[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
    param.mesh_normal[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def scatter_body_mesh_to_grid(
    param: BodyMeshColliderStruct,
    inv_dx: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
):
    """For each face of the body mesh, scatter face_velocity and face_normal
    to the 27 surrounding grid nodes via tricubic B-spline weights.
    """
    p = wp.tid()
    mesh = wp.mesh_get(param.mesh_id)

    vi0 = mesh.indices[3 * p + 0]
    vi1 = mesh.indices[3 * p + 1]
    vi2 = mesh.indices[3 * p + 2]

    p0 = mesh.points[vi0]
    p1 = mesh.points[vi1]
    p2 = mesh.points[vi2]
    face_point = (p0 + p1 + p2) / 3.0

    v0 = mesh.velocities[vi0]
    v1 = mesh.velocities[vi1]
    v2 = mesh.velocities[vi2]
    face_velocity = (v0 + v1 + v2) / 3.0

    face_normal = wp.mesh_eval_face_normal(param.mesh_id, p)

    grid_pos = face_point * inv_dx
    base_x = wp.int(grid_pos[0] - 0.5)
    base_y = wp.int(grid_pos[1] - 0.5)
    base_z = wp.int(grid_pos[2] - 0.5)

    if (base_x < 0 or base_x >= grid_dim_x - 3 or
        base_y < 0 or base_y >= grid_dim_y - 3 or
        base_z < 0 or base_z >= grid_dim_z - 3):
        return

    fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix = base_x + i
                iy = base_y + j
                iz = base_z + k
                weight = w[0, i] * w[1, j] * w[2, k]
                wp.atomic_add(param.mesh_v_in, ix, iy, iz, weight * face_velocity)
                wp.atomic_add(param.mesh_normal, ix, iy, iz, weight * face_normal)
                wp.atomic_add(param.weight, ix, iy, iz, weight)


@wp.kernel
def normalize_body_mesh_grid(param: BodyMeshColliderStruct):
    gx, gy, gz = wp.tid()
    if param.weight[gx, gy, gz] > 1e-15:
        inv_w = 1.0 / param.weight[gx, gy, gz]
        param.mesh_v_out[gx, gy, gz] = param.mesh_v_in[gx, gy, gz] * inv_w
        # leave mesh_normal as accumulated direction; we normalize in collide


@wp.kernel
def collide_cloth_grid_with_body_mesh(
    state_grid_v_resolved: wp.array(dtype=wp.vec3, ndim=4),
    param: BodyMeshColliderStruct,
):
    """Project cloth subject's resolved grid velocity off the body mesh.

    At every grid node where the body mesh has nonzero footprint:
      v_rel = v_cloth - v_body_mesh
      n     = normalized accumulated mesh normal (outward of body)
      v_n   = dot(v_rel, n)
      if v_n < 0  (cloth moving INTO body):
          subtract v_n * n  -> remove inward component
          apply Coulomb friction on the remaining tangential v_proj
      else: cloth separating, leave alone
    """
    gx, gy, gz = wp.tid()
    if param.weight[gx, gy, gz] <= 1e-15:
        return

    v_cloth = state_grid_v_resolved[gx, gy, gz, param.cloth_subj_id]
    v_body = param.mesh_v_out[gx, gy, gz]
    v_rel = v_cloth - v_body

    n_raw = param.mesh_normal[gx, gy, gz]
    n_len = wp.length(n_raw)
    if n_len < 1e-10:
        return
    n = n_raw * (1.0 / n_len)

    normal_component = wp.dot(v_rel, n)
    # Project out only inward component (outward is fine = separating)
    v_proj = v_rel - wp.min(normal_component, 0.0) * n

    if normal_component < 0.0 and wp.length(v_proj) > 1e-20:
        # Coulomb friction: reduce tangential speed proportional to normal pressure
        v_fric_mag = wp.max(0.0, wp.length(v_proj) + normal_component * param.friction)
        v_fric = v_fric_mag * wp.normalize(v_proj)
    else:
        v_fric = v_proj

    state_grid_v_resolved[gx, gy, gz, param.cloth_subj_id] = v_fric + v_body


# ---------------------------------------------------------------------------
# Helper kernel: copy a torch (V,3) tensor into wp.Mesh.points / .velocities
# (mirrors MPMAvatar's set_vec3_to_vec3)
# ---------------------------------------------------------------------------
@wp.kernel
def set_vec3_to_vec3(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i]


# ---------------------------------------------------------------------------
# State holder
# ---------------------------------------------------------------------------
class BodyMeshColliderState:
    """Owns the wp.Mesh wrapping the SMPL-X body mesh + scratch grid arrays.

    Args:
        verts: (V,3) torch float tensor of initial vertex positions in world
               (MPM grid) coord (post body-rotation/scale/center).
        faces: (F,3) torch int64 tensor of triangle indices into verts.
        n_grid: cubic grid dimension matching mpm_model.n_grid.
        cloth_subj_id: index of the cloth subject in the separable-contact
                       state (ie. value of mpm_state.particle_subject_id for
                       cloth particles).
        friction: Coulomb friction coefficient at the cloth-body interface.
        device: torch device string.
    """

    def __init__(self, verts, faces, n_grid, cloth_subj_id, friction=0.1, device="cuda:0"):
        verts_np = verts.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)
        assert faces_np.ndim == 2 and faces_np.shape[1] == 3, "faces must be (F,3)"

        self.n_verts = verts_np.shape[0]
        self.n_faces = faces_np.shape[0]
        self.device = device

        wp_points = wp.from_numpy(verts_np, dtype=wp.vec3, device=device, requires_grad=False)
        wp_vels = wp.zeros_like(wp_points, requires_grad=False)
        wp_indices = wp.from_numpy(faces_np.flatten(), dtype=wp.int32, device=device, requires_grad=False)
        self.mesh = wp.Mesh(points=wp_points, velocities=wp_vels, indices=wp_indices)

        self.struct = BodyMeshColliderStruct()
        self.struct.mesh_id = self.mesh.id
        self.struct.friction = float(friction)
        self.struct.cloth_subj_id = int(cloth_subj_id)
        self.struct.weight = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=float, device=device)
        self.struct.mesh_v_in = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device=device)
        self.struct.mesh_v_out = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device=device)
        self.struct.mesh_normal = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device=device)

        self.n_grid = n_grid

    def update_pose(self, verts_now, verts_next, frame_dt):
        """Refresh wp.Mesh.points to verts_now and .velocities to
        (verts_next - verts_now) / frame_dt. Both inputs are torch (V,3)
        float tensors in world (MPM grid) coord.

        Call once per frame, BEFORE the per-substep loop.
        """
        assert verts_now.shape[0] == self.n_verts
        assert verts_next.shape[0] == self.n_verts

        verts_now_wp = wp.from_torch(
            verts_now.detach().contiguous().to(self.device).float().view(-1, 3),
            dtype=wp.vec3,
        )
        vels_torch = ((verts_next - verts_now) / float(frame_dt)).detach().contiguous().to(self.device).float()
        vels_wp = wp.from_torch(vels_torch.view(-1, 3), dtype=wp.vec3)

        wp.launch(
            kernel=set_vec3_to_vec3,
            dim=self.n_verts,
            inputs=[self.mesh.points, verts_now_wp],
            device=self.device,
        )
        wp.launch(
            kernel=set_vec3_to_vec3,
            dim=self.n_verts,
            inputs=[self.mesh.velocities, vels_wp],
            device=self.device,
        )
        # rebuild BVH so wp.mesh_eval_face_normal returns up-to-date normals
        self.mesh.refit()

    def launch(self, mpm_model):
        """Run the per-substep mesh-collider pipeline:
           1. zero scratch grid arrays
           2. scatter face data (velocity, normal) from each body face to grid
           3. normalize by accumulated weight
           4. collide: project cloth-subject grid_v_resolved off the mesh

        Call once per substep, AFTER compute_vcm_and_resolve has populated
        state.grid_v_resolved.
        """
        wp.launch(
            kernel=zero_body_mesh_grid,
            dim=(self.n_grid, self.n_grid, self.n_grid),
            inputs=[self.struct],
            device=self.device,
        )
        wp.launch(
            kernel=scatter_body_mesh_to_grid,
            dim=self.n_faces,
            inputs=[
                self.struct,
                float(mpm_model.inv_dx),
                int(mpm_model.grid_dim_x),
                int(mpm_model.grid_dim_y),
                int(mpm_model.grid_dim_z),
            ],
            device=self.device,
        )
        wp.launch(
            kernel=normalize_body_mesh_grid,
            dim=(self.n_grid, self.n_grid, self.n_grid),
            inputs=[self.struct],
            device=self.device,
        )

    def collide(self, mpm_state):
        """Apply the cloth-body collision projection. Separated from launch()
        because it needs the resolved grid velocity field, which is populated
        by compute_vcm_and_resolve in Phase D — caller is responsible for
        ordering."""
        wp.launch(
            kernel=collide_cloth_grid_with_body_mesh,
            dim=(self.n_grid, self.n_grid, self.n_grid),
            inputs=[mpm_state.grid_v_resolved, self.struct],
            device=self.device,
        )
