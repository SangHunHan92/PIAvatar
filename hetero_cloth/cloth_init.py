"""Cloth initialisation: convert (verts, faces) garment template into the
particle arrays the user's MPM_Simulator_WARP will register, plus the
ClothStateStruct fields the cloth kernels need.

Mirrors MPMAvatar's compute_dir_vol / compute_rest_dir_inv / setup_simulation
init pipeline ([run_demo.py:240-274, 381-426]) but as a side-effect-free
helper that returns torch tensors. The caller (cloth_solver.py) decides
how/where to allocate.

Usage:
    prep = prepare_cloth_template(
        verts_world,           # (N_v, 3) torch float32 in world coords
        faces,                 # (N_f, 3) torch int64
        n_existing,            # number of pre-cloth particles in the global array
        thickness=1e-5,        # element thickness for volume = 0.25 * t * |d1 x d2|
        density=1.0,           # cloth density (kg/m^3 in sim units; cf run_demo material_params)
        gamma=1e3,             # shear stiffness (per element, broadcast)
        kappa=1e5,             # thickness compression stiffness
        friction_angle_deg=20.0,
    )
    # prep is a ClothPrep dataclass; see below for fields.

The caller then:
    1. concatenates prep.element_x and prep.vertex_x AFTER the user's existing
       particle positions to produce the full particle_x of length N_total.
    2. calls cloth_state.init(n_existing, n_e, n_v) and copies prep.* into
       cloth_state arrays via populate_cloth_state(cloth_state, prep, device).
    3. sets the per-particle flags (traditional/element/vertex) on cloth_state.
"""
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp


@dataclass
class ClothPrep:
    # particle positions, velocities, volumes, masses (CONCATENATED: elements then vertices)
    element_x: torch.Tensor         # (N_e, 3) float32, world coords
    element_v: torch.Tensor         # (N_e, 3) float32, zeros
    element_vol: torch.Tensor       # (N_e,)    float32
    element_mass: torch.Tensor      # (N_e,)    float32, zeros (stress-carriers, no mass)
    vertex_x: torch.Tensor          # (N_v, 3)
    vertex_v: torch.Tensor          # (N_v, 3) zeros
    vertex_vol: torch.Tensor        # (N_v,)
    vertex_mass: torch.Tensor       # (N_v,) = density * vertex_vol

    # per-element fields for ClothStateStruct
    particle_d: torch.Tensor        # (N_e, 3, 3) initial deformation matrix [d1 | d2 | d3]
    particle_D_inv: torch.Tensor    # (N_e, 3, 3) full inverse of init_dir
    particle_R_inv: torch.Tensor    # (N_e, 3)   upper-tri rest-direction inverse [iR11, iR12, iR22]

    # faces stored as GLOBAL particle indices (so kernels can directly index state.particle_x[v])
    faces_global: torch.Tensor      # (N_e, 3) float32 (will be wp.vec3)

    # per-element material params
    gamma_per_elem: torch.Tensor    # (N_e,)
    kappa_per_elem: torch.Tensor    # (N_e,)
    friction_coeff: float           # scalar = tan(friction_angle_deg * pi / 180)

    # per-element/per-vertex flags for ClothStateStruct (length N_total = n_existing + N_e + N_v)
    n_existing: int
    n_elements: int
    n_vertices: int


def _compute_dir_vol(verts: torch.Tensor, faces: torch.Tensor, thickness: float):
    """Mirrors MPMAvatar Trainer.compute_dir_vol exactly.
    Returns (init_dir, rest_dir, element_vol, vertex_vol)."""
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    d1 = verts[f1] - verts[f0]                                  # (N_e, 3)
    d2 = verts[f2] - verts[f0]
    d3 = torch.linalg.cross(d1, d2)                             # area * 2 * normal
    d3 = d3 / d3.norm(dim=1, keepdim=True).clamp_min(1e-20)
    init_dir = torch.stack([d1, d2, d3], dim=-1)                # (N_e, 3, 3)

    R11 = d1.norm(dim=1)
    R12 = (d1 * d2).sum(dim=1) / R11.clamp_min(1e-20)
    R22 = (d2 - (R12 / R11.clamp_min(1e-20))[:, None] * d1).norm(dim=1)
    rest_dir = torch.stack([R11, R12, R22], dim=-1)             # (N_e, 3)

    area = 0.5 * torch.linalg.cross(d1, d2).norm(dim=1)
    element_vol = 0.25 * thickness * area                       # (N_e,)
    vertex_vol = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    vertex_vol.index_add_(0, faces.reshape(-1),
                          element_vol[:, None].expand(-1, 3).reshape(-1))
    return init_dir, rest_dir, element_vol, vertex_vol


def _compute_rest_dir_inv(rest_dir: torch.Tensor) -> torch.Tensor:
    """Mirrors MPMAvatar.compute_rest_dir_inv. (N_e, 3) -> (N_e, 3) packed iR11/iR12/iR22."""
    R11, R12, R22 = rest_dir[:, 0], rest_dir[:, 1], rest_dir[:, 2]
    iR11 = 1.0 / R11.clamp_min(1e-20)
    iR22 = 1.0 / R22.clamp_min(1e-20)
    iR12 = -R12 * iR11 * iR22
    return torch.stack([iR11, iR12, iR22], dim=-1)


def prepare_cloth_template(
    verts: torch.Tensor,
    faces: torch.Tensor,
    n_existing: int,
    thickness: float = 1e-5,
    density: float = 1.0,
    gamma: float = 1e3,
    kappa: float = 1e5,
    friction_angle_deg: float = 20.0,
) -> ClothPrep:
    """Take a cloth garment (verts, faces) and produce all arrays needed to
    register cloth into the user's MPM_Simulator_WARP and to populate
    ClothStateStruct.

    Argument verts / faces are expected in WORLD coordinates (any units
    consistent with the rest of your scene). Caller may apply wld2sim later.

    The element particles are placed at face centroids (verts[face].mean(1)).
    Vertices stay at their original positions.

    Layout produced for the global particle array (caller assembles):
        [..., n_existing)              user's existing particles (CALLER provides)
        [n_existing, n_existing + N_e) cloth elements (this function)
        [n_existing + N_e, ...)        cloth vertices  (this function)
    """
    assert verts.dtype == torch.float32, "verts must be float32"
    assert faces.dtype in (torch.int32, torch.int64), "faces must be int"
    device = verts.device
    n_v = verts.shape[0]
    n_e = faces.shape[0]

    faces_long = faces.to(torch.int64)

    # element centroids
    element_x = verts[faces_long].mean(dim=1).contiguous()
    element_v = torch.zeros_like(element_x)

    # vertex positions / velocities
    vertex_x = verts.contiguous()
    vertex_v = torch.zeros_like(vertex_x)

    # initial direction matrices and volumes
    init_dir, rest_dir, element_vol, vertex_vol = _compute_dir_vol(verts, faces_long, thickness)

    # full 3x3 inverse of init_dir (used as particle_D_inv for cloth_state)
    D_inv = torch.linalg.inv(init_dir).contiguous()
    # upper-tri rest-direction inverse packed as vec3
    R_inv = _compute_rest_dir_inv(rest_dir).contiguous()

    # masses: BOTH element and vertex carry density*vol (matches MPMAvatar
    # run_demo.py:366-370 update_mass=True). Element mass is the critical
    # anchor: when two non-adjacent folds share a grid cell, both contribute
    # mass so the grid velocity is a well-defined weighted average (else the
    # cell receives stress-divergence forces with no mass to damp them →
    # phantom cohesion / sticking that can't release).
    element_mass = (density * element_vol).to(torch.float32)
    vertex_mass  = (density * vertex_vol).to(torch.float32)

    # per-element material params (uniform for now; broadcast)
    gamma_per_elem = torch.full((n_e,), gamma, dtype=torch.float32, device=device)
    kappa_per_elem = torch.full((n_e,), kappa, dtype=torch.float32, device=device)

    # face indices stored as GLOBAL particle indices into state.particle_x.
    # Vertices live at [n_existing + n_e, n_existing + n_e + n_v).
    base = n_existing + n_e
    faces_global = (faces_long + base).to(torch.float32).contiguous()  # cloth_state.faces is wp.vec3

    friction_coeff = float(np.tan(np.deg2rad(friction_angle_deg)))

    return ClothPrep(
        element_x=element_x, element_v=element_v,
        element_vol=element_vol, element_mass=element_mass,
        vertex_x=vertex_x, vertex_v=vertex_v,
        vertex_vol=vertex_vol, vertex_mass=vertex_mass,
        particle_d=init_dir.contiguous(),
        particle_D_inv=D_inv,
        particle_R_inv=R_inv,
        faces_global=faces_global,
        gamma_per_elem=gamma_per_elem, kappa_per_elem=kappa_per_elem,
        friction_coeff=friction_coeff,
        n_existing=n_existing, n_elements=n_e, n_vertices=n_v,
    )


def populate_cloth_state(cloth_state, prep: ClothPrep, device: str = "cuda:0"):
    """Copy ClothPrep tensors into an already-initialised ClothStateStruct.
    `cloth_state` must have been allocated via cloth_state.init(n_existing, n_e, n_v, device).

    Also sets the per-particle flags (traditional / elements / vertices). Caller
    is responsible for ensuring user's existing particles correspond to indices
    [0, n_existing) - those are marked particle_traditional=1 here.
    """
    n_e = prep.n_elements
    n_v = prep.n_vertices
    n_total = prep.n_existing + n_e + n_v

    # per-element fields
    cloth_state.particle_d = wp.from_torch(prep.particle_d.contiguous(), dtype=wp.mat33, requires_grad=False)
    cloth_state.particle_D_inv = wp.from_torch(prep.particle_D_inv.contiguous(), dtype=wp.mat33, requires_grad=False)
    cloth_state.particle_R_inv = wp.from_torch(prep.particle_R_inv.contiguous(), dtype=wp.vec3, requires_grad=False)
    cloth_state.faces = wp.from_torch(prep.faces_global.contiguous(), dtype=wp.vec3, requires_grad=False)
    cloth_state.gamma = wp.from_torch(prep.gamma_per_elem.contiguous(), dtype=wp.float32, requires_grad=False)
    cloth_state.kappa = wp.from_torch(prep.kappa_per_elem.contiguous(), dtype=wp.float32, requires_grad=False)

    # per-vertex field starts zero (zero_vertex_force_kernel will reset each substep anyway)
    cloth_state.vertex_force = wp.zeros(shape=n_v, dtype=wp.vec3, device=device, requires_grad=False)

    # scalar
    cloth_state.friction_coeff = prep.friction_coeff
    cloth_state.n_existing = prep.n_existing
    cloth_state.n_elements = n_e
    cloth_state.n_vertices = n_v

    # per-particle flags (full N_total length, matching MPMStateStruct.particle_x)
    flags_traditional = np.zeros(n_total, dtype=np.int32)
    flags_elements    = np.zeros(n_total, dtype=np.int32)
    flags_vertices    = np.zeros(n_total, dtype=np.int32)
    flags_traditional[:prep.n_existing] = 1
    flags_elements[prep.n_existing : prep.n_existing + n_e] = 1
    flags_vertices[prep.n_existing + n_e : prep.n_existing + n_e + n_v] = 1
    cloth_state.particle_traditional = wp.from_numpy(flags_traditional, dtype=int, device=device, requires_grad=False)
    cloth_state.particle_elements    = wp.from_numpy(flags_elements,    dtype=int, device=device, requires_grad=False)
    cloth_state.particle_vertices    = wp.from_numpy(flags_vertices,    dtype=int, device=device, requires_grad=False)


def assemble_cloth_particle_arrays(prep: ClothPrep):
    """Convenience: stack element + vertex arrays in the [elements | vertices]
    order the global particle array expects.
    Returns (cloth_x, cloth_v, cloth_vol, cloth_mass) - all length n_e + n_v.

    Caller concatenates these AFTER user's existing particle arrays to form the
    full state.particle_x / particle_v / particle_vol / particle_mass tensors.
    """
    cloth_x    = torch.cat([prep.element_x,    prep.vertex_x],    dim=0).contiguous()
    cloth_v    = torch.cat([prep.element_v,    prep.vertex_v],    dim=0).contiguous()
    cloth_vol  = torch.cat([prep.element_vol,  prep.vertex_vol],  dim=0).contiguous()
    cloth_mass = torch.cat([prep.element_mass, prep.vertex_mass], dim=0).contiguous()
    return cloth_x, cloth_v, cloth_vol, cloth_mass
