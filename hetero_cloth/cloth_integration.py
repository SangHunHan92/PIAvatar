"""High-level helpers for integrating cloth into user's MPM_Simulator_WARP_with_Cloth.

Three responsibilities:
  - load_cloth_obj : read MPMAvatar's cloth_sim.obj (or any triangle OBJ) into torch tensors
  - extend_mpm_params_with_cloth : append cloth particles onto user's mpm_params dict
  - apply_cloth_material : after user's set_subjects_parameters runs (which leaves
    cloth slots empty since cloth is not in subject_params), fill in the cloth
    slots' E / nu / density / gravity directly via wp.launch over cloth indices.

Risks pre-empted (see documented analysis in conversation):
  - cloth_subject_id NOT in subject_params -> apply_cloth_material handles it
  - use_separable_contact=True needs per-subject scatter for vertex_force; for
    first smoke use use_separable_contact=False (caller's responsibility)
  - cloth template scale / position must match the user's sim coordinate system;
    caller passes scale/offset via load_cloth_obj
  - n_grid * grid_lim must contain cloth + body; caller ensures via offset
"""
import os
from typing import Optional

import numpy as np
import torch
import warp as wp


def pose_a1_cloth_with_body(
    canonical_npz: str,
    pose_dataset,
    device: str = "cuda:0",
    outward_offset: float = 0.0,
):
    """Forward-LBS cloth canonical to first-frame pose using BODY's SMPL-X model
    and pose params (from pose_dataset). Ensures body+cloth share identical
    A matrices and transl (R_stand-composed in pose_dataset), so they align
    perfectly in world coord.

    `outward_offset` (meters): if > 0, push each cloth vertex outward along its
    surface normal by this amount AFTER LBS posing. Useful as a buffer to
    prevent body MPM particles from penetrating cloth during fast motion (e.g.
    jumps). 0.01 = 1cm offset works well for most cases.

    Returns (cloth_v (22424,3) float32, cloth_f (...,3) int64) on `device`.
    """
    can = np.load(canonical_npz)
    cloth_v_can = torch.from_numpy(can["cloth_v_canonical"]).float().to(device)  # (22424,3)
    cloth_lbs_w = torch.from_numpy(can["cloth_lbs_w"]).float().to(device)        # (22424,55)
    new_cloth_faces = can["cloth_faces"]                                          # (F, 3)

    # Forward SMPL-X using body's first-frame pose params (already R_stand-composed)
    smpl_model = pose_dataset.smpl_model
    first_idx  = pose_dataset.pose_list[0]
    out = smpl_model.forward(
        betas=pose_dataset.smpl_shape[None],
        global_orient=pose_dataset.body_poses[first_idx, :3][None],
        transl=pose_dataset.transl[first_idx][None],
        body_pose=pose_dataset.body_poses[first_idx, 3:66][None],
        left_hand_pose=pose_dataset.left_hand_pose[first_idx][None],
        right_hand_pose=pose_dataset.right_hand_pose[first_idx][None],
    )
    A = out.A[0]                                                                  # (55,4,4)
    # T per cloth-vert = sum_k w_k * A_k
    T_per_vert = torch.einsum("vj,jab->vab", cloth_lbs_w, A.view(55, 4, 4))       # (V,4,4)
    homo = torch.cat([cloth_v_can, torch.ones(cloth_v_can.shape[0], 1, device=device)], 1)
    cloth_v_world = (T_per_vert @ homo.unsqueeze(-1)).squeeze(-1)[:, :3]           # (V,3)

    # Optional: push cloth verts outward along surface normals by `outward_offset`
    # to create a buffer between cloth and body (prevents penetration during jumps).
    if outward_offset > 0.0:
        faces_t = torch.from_numpy(new_cloth_faces.astype(np.int64)).to(device)
        v0 = cloth_v_world[faces_t[:, 0]]
        v1 = cloth_v_world[faces_t[:, 1]]
        v2 = cloth_v_world[faces_t[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)                       # (F, 3)
        face_normals = face_normals / (face_normals.norm(dim=1, keepdim=True) + 1e-9)
        vert_normals = torch.zeros_like(cloth_v_world)
        for c in range(3):
            vert_normals.index_add_(0, faces_t[:, c], face_normals)
        vert_normals = vert_normals / (vert_normals.norm(dim=1, keepdim=True) + 1e-9)
        cloth_v_world = cloth_v_world + vert_normals * float(outward_offset)
        print(f"[cloth] outward_offset={outward_offset*100:.1f}cm applied along surface normals")

    cloth_v = cloth_v_world.detach().cpu().numpy().astype(np.float32)
    v = torch.from_numpy(cloth_v).to(device).contiguous()
    f = torch.from_numpy(new_cloth_faces.astype(np.int64)).to(device).contiguous()
    return v, f


def get_body_mesh_at_frame(
    pose_dataset,
    frame_idx_in_pose_list: int,
    body_rot_mats,
    body_ori_mean,
    body_scale: float,
    body_center,
    device: str = "cuda:0",
):
    """Forward SMPL-X at one frame and apply the body subject's world transform
    (rotation -> (x - ori_mean) * scale + center). Returns the body mesh
    vertices in MPM-grid world coord, matching where the body MPM particles
    actually live.

    `frame_idx_in_pose_list` is an index into pose_dataset.pose_list. We clamp
    to the last valid index so callers can pass `frame+1` past the end of the
    sequence and still get a sensible verts_next for velocity estimation.

    Returns (verts_world (V,3) float32, faces (F,3) int64).
    """
    from utils.transformation_utils import apply_rotations as _apply_rot

    smpl_model = pose_dataset.smpl_model

    # clamp to valid range
    n_poses = len(pose_dataset.pose_list)
    fi = max(0, min(frame_idx_in_pose_list, n_poses - 1))
    pose_idx = pose_dataset.pose_list[fi]

    out = smpl_model.forward(
        betas=pose_dataset.smpl_shape[None],
        global_orient=pose_dataset.body_poses[pose_idx, :3][None],
        transl=pose_dataset.transl[pose_idx][None],
        body_pose=pose_dataset.body_poses[pose_idx, 3:66][None],
        left_hand_pose=pose_dataset.left_hand_pose[pose_idx][None],
        right_hand_pose=pose_dataset.right_hand_pose[pose_idx][None],
    )
    verts_local = out.vertices[0]                                              # (V,3) SMPL-X coord
    faces_np = smpl_model.faces.astype(np.int64)                               # (F,3) int

    # Apply same world transform chain as body MPM particles (mirrors
    # simulation_with_cloth.py:285-291 for cloth, and load_smplx_a1 for body).
    verts_world = _apply_rot(verts_local, body_rot_mats)
    body_ori_mean_t = body_ori_mean.to(verts_world.device, dtype=verts_world.dtype)
    body_center_t = body_center.to(verts_world.device, dtype=verts_world.dtype) \
        if torch.is_tensor(body_center) else \
        torch.tensor(body_center, device=verts_world.device, dtype=verts_world.dtype)
    verts_world = (verts_world - body_ori_mean_t) * float(body_scale) + body_center_t

    faces_t = torch.from_numpy(faces_np).to(device).contiguous()
    return verts_world.contiguous(), faces_t


def setup_cloth_lbs_pin(
    solver,
    cloth_state,
    canonical_npz: str,
    pin_top_pct: float = 0.2,
    device: str = "cuda:0",
):
    """Mark top `pin_top_pct` cloth verts (by canonical Y, head=+Y) as PINNED.
    These verts get their particle_v overwritten each substep with an LBS-driven
    velocity computed once per frame from now/next body pose via cloth_lbs_w.

    Sets cloth_state.pin_mask. The actual velocity update is done by
    `update_cloth_pin_target_v` once per frame and the
    `apply_cloth_pin_velocity_kernel` (called from cloth_solver per substep).

    Returns (n_pinned, pinned_local_indices_np, cloth_v_can_np, cloth_lbs_w_np).
    """
    can = np.load(canonical_npz)
    cloth_v_can = can["cloth_v_canonical"].astype(np.float32)                    # (n_v, 3) Y-up
    cloth_lbs_w = can["cloth_lbs_w"].astype(np.float32)                          # (n_v, 55)
    n_v = cloth_v_can.shape[0]
    n_pin = int(round(n_v * pin_top_pct))
    if n_pin <= 0:
        return 0, np.array([], dtype=np.int32), cloth_v_can, cloth_lbs_w

    pinned_local = np.argsort(cloth_v_can[:, 1])[-n_pin:].astype(np.int32)
    mask_np = np.zeros(n_v, dtype=np.int32)
    mask_np[pinned_local] = 1

    cloth_state.pin_mask = wp.array(mask_np, dtype=int, device=device)
    cloth_state.pin_target_v = wp.zeros(shape=n_v, dtype=wp.vec3, device=device)
    print(f"[cloth] LBS-pinned top {n_pin}/{n_v} verts (top {pin_top_pct*100:.0f}%)")
    return n_pin, pinned_local, cloth_v_can, cloth_lbs_w


def update_cloth_pin_target_v(
    cloth_state,
    pose_dataset,
    human_step: int,
    cloth_v_can_np: np.ndarray,
    cloth_lbs_w_np: np.ndarray,
    pinned_local: np.ndarray,
    body_rot_mats,
    body_scale: float,
    smplx_dt: float,
    device: str = "cuda:0",
):
    """Compute target velocity (in MPM space) for pinned cloth verts at the
    current frame and write to cloth_state.pin_target_v.

    Velocity is forward-finite-difference of LBS-posed positions:
        v_world = (T(A_next) - T(A_now)) @ canonical / smplx_dt
        v_mpm   = (v_world @ rot_mats) * body_scale

    Run once per frame, BEFORE substepping (called from cloth_solver at step==0).
    """
    n_v = cloth_v_can_np.shape[0]
    if pinned_local.size == 0:
        return

    smpl_model = pose_dataset.smpl_model
    first_idx  = pose_dataset.pose_list[0]
    if human_step + 1 >= len(pose_dataset.pose_list):
        # past sequence end: no velocity update (keep last)
        return
    now_frame  = pose_dataset.pose_list[human_step]
    next_frame = pose_dataset.pose_list[human_step + 1]

    def fwd(frame_idx):
        return smpl_model.forward(
            betas=pose_dataset.smpl_shape[None],
            global_orient=pose_dataset.body_poses[frame_idx, :3][None],
            transl=pose_dataset.transl[frame_idx][None],
            body_pose=pose_dataset.body_poses[frame_idx, 3:66][None],
            left_hand_pose=pose_dataset.left_hand_pose[first_idx][None],
            right_hand_pose=pose_dataset.right_hand_pose[first_idx][None],
        )

    with torch.no_grad():
        out_now  = fwd(now_frame)
        out_next = fwd(next_frame)
        A_now  = out_now.A[0].view(55, 4, 4)
        A_next = out_next.A[0].view(55, 4, 4)

        cloth_v_can_t = torch.from_numpy(cloth_v_can_np[pinned_local]).to(device).float()
        cloth_lbs_w_t = torch.from_numpy(cloth_lbs_w_np[pinned_local]).to(device).float()
        homo = torch.cat([cloth_v_can_t, torch.ones(cloth_v_can_t.shape[0], 1, device=device)], 1)

        T_now  = torch.einsum("vj,jab->vab", cloth_lbs_w_t, A_now)               # (P,4,4)
        T_next = torch.einsum("vj,jab->vab", cloth_lbs_w_t, A_next)
        v_world_now  = (T_now  @ homo.unsqueeze(-1)).squeeze(-1)[:, :3]          # (P,3)
        v_world_next = (T_next @ homo.unsqueeze(-1)).squeeze(-1)[:, :3]
        delta = v_world_next - v_world_now                                       # (P,3)

        # Map world delta into MPM-space velocity: rotate by body rot_mats (using
        # apply_rotation's pos@R.T convention) then scale.
        rot = body_rot_mats[0] if isinstance(body_rot_mats, list) else body_rot_mats
        delta_mpm = delta @ rot.T                                                # (P,3)
        v_target = (delta_mpm / float(smplx_dt)) * float(body_scale)             # MPM-space velocity

        # write to pin_target_v at pinned local indices (other indices stay zero)
        pin_v_full = torch.zeros((n_v, 3), device=device, dtype=torch.float32)
        pin_v_full[torch.from_numpy(pinned_local).long().to(device)] = v_target

        # to warp
        cloth_state.pin_target_v = wp.from_torch(pin_v_full.contiguous(), dtype=wp.vec3)


def load_a1_cloth_subset(
    canonical_npz: str = "third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz",
    tracked_npz: str = "third_party/MPMAvatar/output/tracking/a1_s1_460_200/params_460.npz",
    start_frame: int = 460,
    pose_dir: str = "third_party/MPMAvatar/data/a1_s1/smplx_fitted",
    device: str = "cuda:0",
):
    """Load cloth subset (22424 verts) in UNSCALED SMPL-X coord at start_frame.

    Aligns with `load_smplx_a1`'s body output (also unscaled SMPL-X coord).

    Two paths:
      (a) start_frame == 460 AND tracked_npz exists  →  use the image-fit
          tracked params_460['vertices'] (best hand shapes), divided by
          pose_scale to remove MPMAvatar's scale.
      (b) any other start_frame  →  forward LBS from cloth_v_canonical
          (T-pose) using cloth_lbs_w + start_frame's pose params. This works
          for any frame in pose_dir (e.g., 460..1728), not just 460-659.

    Returns (verts (22424,3) float32, faces (...,3) int64) on `device`.
    """
    can = np.load(canonical_npz)
    new_cloth_faces = can["cloth_faces"]
    sp = np.load("third_party/MPMAvatar/data/a1_s1/split_idx.npz")
    cloth_v_idx = sp["reordered_cloth_v_idx"]

    # Stand-up rotation matrix to convert MPMAvatar's Y-up pose to our Z-up
    # framework world. Applied as a final step after re-posing.
    R_stand_np = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]], dtype=np.float32)

    use_tracked = (start_frame == 460) and os.path.exists(tracked_npz)
    if use_tracked:
        pose_scale = float(can["pose460_scale"])
        tracked = np.load(tracked_npz)["vertices"]               # (48280,3) MPMAvatar-scaled
        v_unscaled = tracked / pose_scale
        cloth_v = v_unscaled[cloth_v_idx].astype(np.float32)
        cloth_v = cloth_v @ R_stand_np.T                          # apply Y->Z stand-up
    else:
        # Forward LBS from cloth canonical
        cloth_v_can  = can["cloth_v_canonical"]                  # (22424, 3)
        cloth_lbs_w  = can["cloth_lbs_w"]                        # (22424, 55)

        import sys as _sys
        _MPMA = "third_party/MPMAvatar"
        if _MPMA not in _sys.path:
            _sys.path.insert(0, _MPMA)
        try:
            from utils.smplx_deformer import SmplxDeformer
        finally:
            if _sys.path[0] == _MPMA:
                _sys.path.pop(0)

        pose_pth = os.path.join(pose_dir, f"{start_frame:06d}.pth")
        if not os.path.exists(pose_pth):
            raise FileNotFoundError(f"Pose file missing: {pose_pth}")
        pose_d = torch.load(pose_pth, map_location="cpu")
        pose_d = {k: (v.to(device) if hasattr(v, "to") else torch.tensor(v).to(device))
                  for k, v in pose_d.items()}

        dfm = SmplxDeformer(
            model_path="third_party/MPMAvatar/data/body_models",
            gender="neutral", num_betas=300, use_pca=False,
        )
        smplx_out = dfm.smplx_forward(pose_d)                    # smplx_out.vertices is *= scale
        # transform_to_pose returns scaled world → divide by scale to get unscaled
        v_can_t = torch.from_numpy(cloth_v_can).float().to(device)
        lbs_t   = torch.from_numpy(cloth_lbs_w).float().to(device)
        deformed, _ = dfm.transform_to_pose(
            v_can_t.unsqueeze(0), lbs_t.unsqueeze(0),
            smplx_out, pose_d["trans"], pose_d["scale"],
        )                                                         # MPMAvatar-scaled
        scale_v = float(pose_d["scale"])
        cloth_v = (deformed.reshape(-1, 3) / scale_v).detach().cpu().numpy().astype(np.float32)
        cloth_v = cloth_v @ R_stand_np.T                          # apply Y->Z stand-up

    v = torch.from_numpy(cloth_v).to(device).contiguous()
    f = torch.from_numpy(new_cloth_faces.astype(np.int64)).to(device).contiguous()
    return v, f


def load_cloth_obj(
    path: str,
    scale: float = 1.0,
    offset: tuple = (0.0, 0.0, 0.0),
    rotation_deg: float = 0.0,
    rotation_axis: int = 0,
    device: str = "cuda:0",
):
    """Read a triangle OBJ file (verts only, no UV/normals required).

    Returns (verts (N_v,3) float32, faces (N_f,3) int64) on `device`.

    Each vertex is transformed by:  v' = R(angle, axis) @ (scale * v) + offset
    where rotation is applied first (around origin), then scale, then offset.
    rotation_axis: 0=X, 1=Y, 2=Z. rotation_deg in degrees.
    Useful for aligning cloth Y-up template with sim Z-up coord (rotate -90° around X).
    """
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                faces.append(idx)
    v = torch.tensor(verts, dtype=torch.float32) * scale
    if abs(rotation_deg) > 1e-6:
        ang = float(rotation_deg) * 3.14159265358979 / 180.0
        c, s = float(np.cos(ang)), float(np.sin(ang))
        if rotation_axis == 0:    # X axis
            R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        elif rotation_axis == 1:  # Y axis
            R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        else:                     # Z axis
            R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
        v = v @ R.T
    v = v + torch.tensor(offset, dtype=torch.float32)
    f = torch.tensor(faces, dtype=torch.int64)
    return v.to(device).contiguous(), f.to(device).contiguous()


def extend_mpm_params_with_cloth(
    mpm_params: dict,
    cloth_x: torch.Tensor,
    cloth_v: torch.Tensor,
    cloth_vol: torch.Tensor,
    cloth_subject_id: int,
):
    """Append cloth particle arrays at the end of user's mpm_params dict.

    mpm_params is mutated in-place. Required keys present:
        pos (N,3), vol (N,), index (N,), cov (N,6)
    Other keys (shs, opacity, ...) are not required for cloth - caller can omit
    them when not rendering cloth via gaussian splatting.

    The cloth's particle_id (subject id) is set to `cloth_subject_id`. Caller
    must increment n_subjects accordingly when calling load_initial_data_from_torch.
    """
    n_cloth = cloth_x.shape[0]
    device = mpm_params["pos"].device

    cloth_index = torch.full((n_cloth,), cloth_subject_id, dtype=torch.int32, device=device)
    cloth_cov = torch.zeros((n_cloth, 6), dtype=torch.float32, device=device)

    mpm_params["pos"] = torch.cat([mpm_params["pos"], cloth_x.to(device)], dim=0).contiguous()
    mpm_params["vol"] = torch.cat([mpm_params["vol"], cloth_vol.to(device)], dim=0).contiguous()
    mpm_params["index"] = torch.cat([mpm_params["index"], cloth_index], dim=0).contiguous()
    mpm_params["cov"] = torch.cat([mpm_params["cov"], cloth_cov], dim=0).contiguous()
    return mpm_params


@wp.kernel
def _set_cloth_E_nu_density_kernel(
    state_density: wp.array(dtype=float),
    state_mass: wp.array(dtype=float),
    state_vol: wp.array(dtype=float),
    state_gravity: wp.array(dtype=wp.vec3),
    model_E: wp.array(dtype=float),
    model_nu: wp.array(dtype=float),
    n_existing: int,
    n_elements: int,
    n_vertices: int,
    E_value: float,
    nu_value: float,
    density_value: float,
    gravity_x: float,
    gravity_y: float,
    gravity_z: float,
    elements_have_mass: int,
):
    """Set per-particle E, nu, density, mass, gravity for cloth slots.

    Launch dim = n_elements + n_vertices; thread idx in [0, n_e + n_v).
    First n_elements threads -> cloth elements; rest -> cloth vertices.

    Element mass: 0 (stress carriers, no mass-momentum role) unless
    elements_have_mass != 0 in which case mass = density * vol.
    Vertex mass = density * vol.
    """
    tid = wp.tid()
    is_element = tid < n_elements
    if is_element:
        p = n_existing + tid
    else:
        p = n_existing + n_elements + (tid - n_elements)

    model_E[p] = E_value
    model_nu[p] = nu_value
    state_density[p] = density_value
    state_gravity[p] = wp.vec3(gravity_x, gravity_y, gravity_z)

    if is_element and elements_have_mass == 0:
        state_mass[p] = 0.0
    else:
        state_mass[p] = density_value * state_vol[p]


@wp.kernel
def _zero_gravity_for_indices_kernel(
    state_gravity: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
):
    """Zero state.particle_gravity for given particle indices.
    Used to suspend top cloth vertices from gravity (cheap pin-to-body)."""
    i = wp.tid()
    p = indices[i]
    state_gravity[p] = wp.vec3(0.0, 0.0, 0.0)


def apply_cloth_pin_top_no_gravity(
    solver,
    canonical_npz: str,
    cloth_prep,
    pin_top_pct: float = 0.2,
    device: str = "cuda:0",
):
    """Zero gravity for the top `pin_top_pct` fraction of cloth vertices,
    selected by canonical Y-axis position (head=+Y in SMPL-X canonical).

    These verts effectively pin to the body — without gravity they don't fall,
    while body collision still pushes them along with the body. Bottom verts
    keep their gravity and drape freely. Run AFTER apply_cloth_material.
    """
    can = np.load(canonical_npz)
    cloth_v_can = can["cloth_v_canonical"]                         # (n_v, 3) Y-up
    n_v = cloth_v_can.shape[0]
    n_pin = int(round(n_v * pin_top_pct))
    if n_pin <= 0:
        return 0
    # top by canonical y (head +Y in SMPL-X canonical)
    top_local = np.argsort(cloth_v_can[:, 1])[-n_pin:].astype(np.int32)
    pin_particles = (
        cloth_prep.n_existing + cloth_prep.n_elements + top_local
    ).astype(np.int32)
    pin_indices_wp = wp.array(pin_particles, dtype=int, device=device)
    wp.launch(
        kernel=_zero_gravity_for_indices_kernel,
        dim=n_pin,
        inputs=[solver.mpm_state.particle_gravity, pin_indices_wp],
        device=device,
    )
    print(f"[cloth] pinned top {n_pin}/{n_v} verts (top {pin_top_pct*100:.0f}%) — gravity=0")
    return n_pin


def apply_cloth_material(
    solver,
    n_existing: int,
    n_elements: int,
    n_vertices: int,
    E: float,
    nu: float,
    density: float,
    gravity: tuple = (0.0, 0.0, 0.0),
    elements_have_mass: bool = False,
    device: str = "cuda:0",
):
    """After user's set_subjects_parameters / set_simulater_parameters runs,
    fill in cloth slots' E/nu/density/gravity/mass directly. Element mass = 0
    by default (per cloth pipeline design); set elements_have_mass=True to
    treat them as mass carriers (rare).

    Call BEFORE solver.finalize_mu_lam() so mu/lam derive from cloth E/nu.
    """
    wp.launch(
        kernel=_set_cloth_E_nu_density_kernel,
        dim=n_elements + n_vertices,
        inputs=[
            solver.mpm_state.particle_density,
            solver.mpm_state.particle_mass,
            solver.mpm_state.particle_vol,
            solver.mpm_state.particle_gravity,
            solver.mpm_model.E,
            solver.mpm_model.nu,
            n_existing, n_elements, n_vertices,
            float(E), float(nu), float(density),
            float(gravity[0]), float(gravity[1]), float(gravity[2]),
            1 if elements_have_mass else 0,
        ],
        device=device,
    )
