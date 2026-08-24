"""Warp port of particle_filling/filling_new.py.

Replaces the Taichi inside-mesh test (which serialized due to unbounded
`while` loops in collision_search/collision_times) with Warp kernels using
bounded `for` + `break`, restoring GPU parallelism. Algorithm — three-stage
density-then-ray-crossing-parity — is preserved 1:1 from filling_new.py.

API mirror of filling_new.py:
- fill_particles_warp(pos, opacity, cov, grid_n, ...)              # core
- fill_particles_subjects_warp(subjects, subject_params, sim_params)  # entry point used by simulation

Extra over filling_new.py:
- attribute propagation (cov / opacity / shs / index) for new interior
  particles is done in a single pytorch3d.knn_points call instead of the
  O(N_interior x N_surface) Taichi inner loop in get_attr_from_closest.
- get_particle_volume_from_subjects_warp mirrors the Taichi version.
"""

import os
import sys
import torch
import numpy as np
import warp as wp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "gaussian-splatting"))

from mpm_solver_warp.engine_utils import particle_position_tensor_to_ply
from tqdm import tqdm

# --- Warp kernels --------------------------------------------------------

@wp.func
def _node_density(
    nx: int, ny: int, nz: int,
    pos: wp.vec3,
    opacity: float,
    cov_inv: wp.mat33,
    grid_dx: float,
) -> float:
    """Sum Gaussian weights at the 8 corners of cell (nx,ny,nz). Mirrors
    Taichi compute_density(): exp(-0.5 * (p-corner)·cov_inv·(p-corner)) summed
    over 2x2x2 corners, scaled by opacity / 8."""
    s = float(0.0)
    for ci in range(2):
        for cj in range(2):
            for ck in range(2):
                cx = float(nx + ci) * grid_dx
                cy = float(ny + cj) * grid_dx
                cz = float(nz + ck) * grid_dx
                d = pos - wp.vec3(cx, cy, cz)
                q = wp.dot(d, cov_inv * d)
                s = s + wp.exp(-0.5 * q)
    return opacity * s / 8.0


@wp.kernel
def densify_grids_warp(
    pos: wp.array(dtype=wp.vec3),
    opacity: wp.array(dtype=float),
    cov_upper: wp.array(dtype=float, ndim=2),     # [N, 6]
    grid: wp.array(dtype=int, ndim=3),
    grid_density: wp.array(dtype=float, ndim=3),
    grid_n: int,
    grid_dx: float,
):
    pi = wp.tid()
    p = pos[pi]
    i = int(wp.floor(p[0] / grid_dx))
    j = int(wp.floor(p[1] / grid_dx))
    k = int(wp.floor(p[2] / grid_dx))

    if i >= 0 and i < grid_n and j >= 0 and j < grid_n and k >= 0 and k < grid_n:
        wp.atomic_add(grid, i, j, k, 1)

    cxx = cov_upper[pi, 0]
    cxy = cov_upper[pi, 1]
    cxz = cov_upper[pi, 2]
    cyy = cov_upper[pi, 3]
    cyz = cov_upper[pi, 4]
    czz = cov_upper[pi, 5]
    cov = wp.mat33(cxx, cxy, cxz,
                   cxy, cyy, cyz,
                   cxz, cyz, czz)
    cov_inv = wp.inverse(cov)

    # conservative bounding-box radius via trace (>= largest eigenvalue).
    # Taichi version uses sym_eig + max(sigma); trace is an upper bound.
    trace = cxx + cyy + czz
    r_world = wp.sqrt(wp.max(trace, 1.0e-12))
    r = int(wp.ceil(r_world / grid_dx))
    # cap r so we don't blow up for degenerate covariances
    if r > 8:
        r = 8

    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                ii = i + dx
                jj = j + dy
                kk = k + dz
                if ii >= 0 and ii < grid_n and jj >= 0 and jj < grid_n and kk >= 0 and kk < grid_n:
                    d = _node_density(ii, jj, kk, p, opacity[pi], cov_inv, grid_dx)
                    wp.atomic_add(grid_density, ii, jj, kk, d)


@wp.kernel
def fill_dense_grids_warp(
    grid: wp.array(dtype=int, ndim=3),
    grid_density: wp.array(dtype=float, ndim=3),
    new_particles_cell: wp.array(dtype=wp.vec3i),  # output cell index per slot
    new_particles_count: wp.array(dtype=int),       # [1] atomic counter
    grid_n: int,
    density_thres: float,
    max_particles_per_cell: int,
):
    """For each cell with density>thres and current count<max, reserves
    (max - count) slots in the global counter and stores the cell index for
    each slot. Random offsets are sampled later (host-side or in a follow-up
    kernel) to avoid Warp randomness API differences across versions."""
    i, j, k = wp.tid()
    if grid_density[i, j, k] > density_thres:
        cur = grid[i, j, k]
        if cur < max_particles_per_cell:
            diff = max_particles_per_cell - cur
            grid[i, j, k] = max_particles_per_cell
            base = wp.atomic_add(new_particles_count, 0, diff)
            for s in range(diff):
                new_particles_cell[base + s] = wp.vec3i(i, j, k)


@wp.kernel
def internal_filling_warp(
    grid: wp.array(dtype=int, ndim=3),
    grid_density: wp.array(dtype=float, ndim=3),
    new_particles_cell: wp.array(dtype=wp.vec3i),
    new_particles_count: wp.array(dtype=int),
    grid_n: int,
    threshold: float,
    exclude_dir: int,
    ray_cast_dir: int,
    max_particles_per_cell: int,
):
    """Per empty cell: in 6 axis-aligned directions, search for at least one
    high-density cell on each ray (except exclude_dir). If all 5 hit, count
    high-density crossings along ray_cast_dir; odd → cell is interior →
    reserve (max_particles_per_cell - 0) slots like fill_dense_grids."""
    i, j, k = wp.tid()
    if grid[i, j, k] != 0:
        return  # already populated cell

    # 1) collision_search in 5 directions (skip exclude_dir)
    # `int(...)` declarations force Warp to treat these as dynamic (mutable in loops)
    all_hit = int(1)
    for dir_type in range(6):
        if dir_type == exclude_dir:
            continue
        dx = int(0)
        dy = int(0)
        dz = int(0)
        if dir_type == 0: dx = 1
        elif dir_type == 1: dx = -1
        elif dir_type == 2: dy = 1
        elif dir_type == 3: dy = -1
        elif dir_type == 4: dz = 1
        elif dir_type == 5: dz = -1
        ii = i + dx
        jj = j + dy
        kk = k + dz
        hit = int(0)
        oob = int(0)
        # bounded loop: at most grid_n steps; we don't break, just stop updating
        for _ in range(grid_n):
            if oob == 0 and hit == 0:
                if ii < 0 or ii >= grid_n or jj < 0 or jj >= grid_n or kk < 0 or kk >= grid_n:
                    oob = 1
                else:
                    if grid_density[ii, jj, kk] > threshold:
                        hit = 1
                    else:
                        ii = ii + dx
                        jj = jj + dy
                        kk = kk + dz
        if hit == 0:
            all_hit = 0

    if all_hit == 0:
        return

    # 2) collision_times along ray_cast_dir (parity test)
    rdx = int(0)
    rdy = int(0)
    rdz = int(0)
    if ray_cast_dir == 0: rdx = 1
    elif ray_cast_dir == 1: rdx = -1
    elif ray_cast_dir == 2: rdy = 1
    elif ray_cast_dir == 3: rdy = -1
    elif ray_cast_dir == 4: rdz = 1
    elif ray_cast_dir == 5: rdz = -1

    times = int(0)
    # starting cell occupancy state
    state = int(0)
    if grid[i, j, k] > 0:
        state = 1
    ii2 = i + rdx
    jj2 = j + rdy
    kk2 = k + rdz
    oob2 = int(0)
    for _ in range(grid_n):
        if oob2 == 0:
            if ii2 < 0 or ii2 >= grid_n or jj2 < 0 or jj2 >= grid_n or kk2 < 0 or kk2 >= grid_n:
                oob2 = 1
            else:
                new_state = int(0)
                if grid_density[ii2, jj2, kk2] > threshold:
                    new_state = 1
                if new_state != state and state == 0:
                    times = times + 1
                state = new_state
                ii2 = ii2 + rdx
                jj2 = jj2 + rdy
                kk2 = kk2 + rdz

    if (times % 2) == 1:
        # interior — reserve slots
        diff = max_particles_per_cell  # current grid is 0
        grid[i, j, k] = max_particles_per_cell
        base = wp.atomic_add(new_particles_count, 0, diff)
        for s in range(diff):
            new_particles_cell[base + s] = wp.vec3i(i, j, k)


@wp.kernel
def cell_index_to_position(
    new_particles_cell: wp.array(dtype=wp.vec3i),
    rand_offsets: wp.array(dtype=wp.vec3),  # [N, 3] in [0,1)^3 from torch.rand
    new_particles_pos: wp.array(dtype=wp.vec3),
    n_total: int,
    grid_dx: float,
):
    s = wp.tid()
    if s >= n_total:
        return
    c = new_particles_cell[s]
    o = rand_offsets[s]
    new_particles_pos[s] = wp.vec3(
        (float(c[0]) + o[0]) * grid_dx,
        (float(c[1]) + o[1]) * grid_dx,
        (float(c[2]) + o[2]) * grid_dx,
    )


@wp.kernel
def assign_particle_to_grid_warp(
    pos: wp.array(dtype=wp.vec3),
    grid: wp.array(dtype=int, ndim=3),
    grid_n: int,
    grid_dx: float,
):
    pi = wp.tid()
    p = pos[pi]
    i = int(wp.floor(p[0] / grid_dx))
    j = int(wp.floor(p[1] / grid_dx))
    k = int(wp.floor(p[2] / grid_dx))
    if i >= 0 and i < grid_n and j >= 0 and j < grid_n and k >= 0 and k < grid_n:
        wp.atomic_add(grid, i, j, k, 1)


@wp.kernel
def compute_particle_volume_warp(
    pos: wp.array(dtype=wp.vec3),
    grid: wp.array(dtype=int, ndim=3),
    particle_vol: wp.array(dtype=float),
    grid_n: int,
    grid_dx: float,
):
    pi = wp.tid()
    p = pos[pi]
    i = int(wp.floor(p[0] / grid_dx))
    j = int(wp.floor(p[1] / grid_dx))
    k = int(wp.floor(p[2] / grid_dx))
    if i >= 0 and i < grid_n and j >= 0 and j < grid_n and k >= 0 and k < grid_n:
        n = grid[i, j, k]
        if n < 1:
            n = 1
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / float(n)
    else:
        particle_vol[pi] = grid_dx * grid_dx * grid_dx


# --- High-level python wrappers -----------------------------------------

def _torch_to_wp_vec3(t: torch.Tensor) -> wp.array:
    """Zero-copy view from a [N,3] float CUDA tensor to a Warp vec3 array."""
    assert t.is_cuda and t.dtype == torch.float32 and t.shape[1] == 3
    # detach + clone is required when t has requires_grad or grad-related state,
    # otherwise warp.from_torch tries to convert .grad and trips on host pointers.
    return wp.from_torch(t.detach().contiguous(), dtype=wp.vec3)


def fill_particles_warp(
    pos: torch.Tensor,
    opacity: torch.Tensor,
    cov: torch.Tensor,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres: float = 2.0,
    search_thres: float = 1.0,
    max_particles_per_cell: int = 1,
    search_exclude_dir: int = 5,
    ray_cast_dir: int = 4,
    boundary=None,
    smooth: bool = False,
    device: str = "cuda",
):
    """Drop-in Warp replacement for filling_new.fill_particles."""
    pos_clone = pos.clone()
    new_origin = None
    # Auto-compute boundary from positions if requested
    if boundary == "auto" or (isinstance(boundary, list) and len(boundary) == 1 and boundary[0] == "auto"):
        pmin = pos.detach().min(dim=0).values
        pmax = pos.detach().max(dim=0).values
        pad = 0.1 * (pmax - pmin).max().item()
        boundary = [
            pmin[0].item() - pad, pmax[0].item() + pad,
            pmin[1].item() - pad, pmax[1].item() + pad,
            pmin[2].item() - pad, pmax[2].item() + pad,
        ]
    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool, device=device)
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])
        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]
        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]],
                                  dtype=torch.float32, device=device)
        pos = pos - new_origin

    pos_f = pos.detach().contiguous().to(device=device, dtype=torch.float32)
    opacity_f = opacity.detach().reshape(-1).contiguous().to(device=device, dtype=torch.float32)
    cov_f = cov.detach().reshape(-1, 6).contiguous().to(device=device, dtype=torch.float32)

    # Warp arrays
    wp_pos = _torch_to_wp_vec3(pos_f)
    wp_opacity = wp.from_torch(opacity_f, dtype=wp.float32)
    wp_cov = wp.from_torch(cov_f, dtype=wp.float32)

    grid = wp.zeros(shape=(grid_n, grid_n, grid_n), dtype=int, device=device)
    grid_density = wp.zeros(shape=(grid_n, grid_n, grid_n), dtype=float, device=device)

    # Stage 1: density field
    wp.launch(
        kernel=densify_grids_warp,
        dim=pos_f.shape[0],
        inputs=[wp_pos, wp_opacity, wp_cov, grid, grid_density, grid_n, grid_dx],
        device=device,
    )

    # Stage 2: dense fill
    cell_buf = wp.zeros(shape=(max_samples,), dtype=wp.vec3i, device=device)
    count_buf = wp.zeros(shape=(1,), dtype=int, device=device)
    wp.launch(
        kernel=fill_dense_grids_warp,
        dim=(grid_n, grid_n, grid_n),
        inputs=[grid, grid_density, cell_buf, count_buf,
                grid_n, density_thres, max_particles_per_cell],
        device=device,
    )
    n_after_dense = int(count_buf.numpy()[0])

    # (optional) smooth: reproduce by transferring to numpy then back, mcubes is not GPU
    if smooth:
        try:
            import mcubes
            df = grid_density.numpy()
            smoothed = mcubes.smooth(df, method="constrained", max_iters=500).astype(np.float32)
            grid_density.assign(smoothed)
        except Exception as e:
            print(f"[filling_warp] smooth skipped: {e}")

    # Stage 3: internal filling
    wp.launch(
        kernel=internal_filling_warp,
        dim=(grid_n, grid_n, grid_n),
        inputs=[grid, grid_density, cell_buf, count_buf,
                grid_n, search_thres, search_exclude_dir, ray_cast_dir,
                max_particles_per_cell],
        device=device,
    )
    n_total = int(count_buf.numpy()[0])
    n_total = min(n_total, max_samples)
    print(f"[filling_warp] dense_fill={n_after_dense}, internal_fill_total={n_total}")

    # Convert reserved cell indices to actual positions with per-slot random offsets
    new_particles_pos = wp.zeros(shape=(max_samples,), dtype=wp.vec3, device=device)
    if n_total > 0:
        rand_offsets_torch = torch.rand((n_total, 3), device=device, dtype=torch.float32)
        # pad to max_samples (kernel guards via n_total)
        pad = torch.zeros((max_samples - n_total, 3), device=device, dtype=torch.float32) if max_samples > n_total else None
        rand_full = torch.cat([rand_offsets_torch, pad], dim=0) if pad is not None else rand_offsets_torch
        wp_rand = _torch_to_wp_vec3(rand_full)
        wp.launch(
            kernel=cell_index_to_position,
            dim=max_samples,
            inputs=[cell_buf, wp_rand, new_particles_pos, n_total, grid_dx],
            device=device,
        )

    new_particles_torch = wp.to_torch(new_particles_pos)[:n_total].to(device=device).contiguous().clone()

    if boundary is not None and new_origin is not None:
        new_particles_torch = new_particles_torch + new_origin

    out = torch.cat([pos_clone, new_particles_torch], dim=0)
    return out


def _generalized_winding_number(points: torch.Tensor, V: torch.Tensor, F: torch.Tensor,
                                chunk: int = 1024) -> torch.Tensor:
    """Generalized winding number (Jacobson, Kavan, Sorkine-Hornung 2013) for
    deciding inside/outside w.r.t. a closed triangle mesh, on GPU. ~|W|≈1 inside,
    ~0 outside. Chunked over points to bound memory.

    points: [N, 3]   query points
    V:      [Nv, 3]  mesh vertices
    F:      [Nf, 3]  mesh face indices
    returns boolean mask [N] (True = inside)
    """
    N = points.shape[0]
    a_all = V[F[:, 0]]  # [Nf, 3]
    b_all = V[F[:, 1]]
    c_all = V[F[:, 2]]
    out = torch.empty(N, dtype=torch.bool, device=points.device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        p = points[s:e].unsqueeze(1)              # [B, 1, 3]
        a = (a_all.unsqueeze(0) - p)              # [B, Nf, 3]
        b = (b_all.unsqueeze(0) - p)
        c = (c_all.unsqueeze(0) - p)
        la = torch.linalg.norm(a, dim=-1)
        lb = torch.linalg.norm(b, dim=-1)
        lc = torch.linalg.norm(c, dim=-1)
        # solid angle via Van Oosterom-Strackee:
        # tan(omega/2) = numerator / denom
        # numerator = a · (b × c)
        # denom = la*lb*lc + (a·b)*lc + (a·c)*lb + (b·c)*la
        num = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
        omega = 2.0 * torch.atan2(num, den)        # signed solid angle per face
        W = omega.sum(dim=-1) / (4.0 * 3.141592653589793)
        out[s:e] = W.abs() > 0.5
    return out


def _fill_human_canonical(human_seq, subject_param, sim_params, k_lbs: int = 4,
                          eps: float = 1e-8, device: str = "cuda"):
    """Sample interior particles for an SMPL-X human subject in CANONICAL space, then
    forward-LBS to the FIRST-FRAME-POSED frame so the result is mathematically
    consistent with the surface particles' kinematic pipeline.

    Returns:
      interior_first_frame: [N_int, 3]   first-frame-posed positions, to append to subject['pos']
      interior_canonical:   [N_int, 3]   canonical positions (= pt_mats_first^{-1} · interior_first_frame)
      interior_lbs:         [N_int, 55]  LBS weights per interior particle

    Why canonical-space fill: in posed frame, k-NN can pick anatomically wrong
    surface neighbours when limbs fold near the body (e.g. an interior chest
    particle finds an arm vertex as nearest because the arm was folded close).
    In canonical (T-pose) the limbs are spread out and Euclidean k-NN respects
    anatomical adjacency. Forward-LBS using A_first then places the interior at
    the correct first-frame-posed location, identical-by-construction to how the
    surface mesh was placed by the same LBS at first frame — so the
    every-frame `positions_now_total_pos[interior]` matches `particle_x_ori[interior]`
    perfectly, eliminating phantom shape-matching velocity.
    """
    from pytorch3d.ops import knn_points

    fp = subject_param["particle_filling"]
    bone_cano = human_seq["bone_cano"].to(device).float()
    cano_pts = human_seq["cano_pts"].to(device).float()
    bone_n = bone_cano.shape[0]
    surface_n = cano_pts.shape[0]

    # 1. Build canonical body cloud and synthetic per-vertex Gaussian for fill input
    cano_body = torch.cat([bone_cano, cano_pts], dim=0).contiguous()
    cano_opacity = torch.ones(cano_body.shape[0], device=device, dtype=torch.float32)
    # SMPL-X-edge-scale isotropic Gaussian: sigma ~ 0.15 * mean_edge ≈ 0.0023m
    sigma2 = float(fp.get("sigma_canonical", 0.0023)) ** 2
    cano_cov = torch.zeros(cano_body.shape[0], 6, device=device, dtype=torch.float32)
    cano_cov[:, 0] = sigma2
    cano_cov[:, 3] = sigma2
    cano_cov[:, 5] = sigma2

    # 2. Run the same Warp filling pipeline in canonical space
    new_pos = fill_particles_warp(
        pos=cano_body,
        opacity=cano_opacity,
        cov=cano_cov,
        grid_n=fp["n_grid"],
        max_samples=fp["max_particles_num"],
        grid_dx=sim_params["grid_lim"] / fp["n_grid"],
        density_thres=fp["density_threshold"],
        search_thres=fp["search_threshold"],
        max_particles_per_cell=fp["max_partciels_per_cell"],
        search_exclude_dir=fp["search_exclude_direction"],
        ray_cast_dir=fp["ray_cast_direction"],
        boundary=fp.get("boundary", "auto"),
        smooth=fp.get("smooth", True),
        device=device,
    ).to(device)
    candidates = new_pos[cano_body.shape[0]:]
    n_candidates = candidates.shape[0]
    print(f"[filling_warp][canonical] human candidates={n_candidates}")
    if n_candidates == 0:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        empty_lbs = torch.empty((0, 55), device=device, dtype=torch.float32)
        return empty, empty, empty_lbs

    # 3a. Winding-number filter against the canonical SMPL-X surface mesh
    if "faces_only_smplx" in subject_param:
        faces = subject_param["faces_only_smplx"].to(device).long()
        inside_mask = _generalized_winding_number(candidates, cano_pts, faces)
        n_kept = int(inside_mask.sum().item())
        print(f"[filling_warp][canonical] kept {n_kept}/{n_candidates} after canonical-mesh winding-number filter")
        interior_canonical = candidates[inside_mask]
    else:
        interior_canonical = candidates

    # 3b. (optional) exclude points inside the bone meshes (osso bones inside the body)
    if fp.get("exclude_bone_interior", False) and "faces" in subject_param and "bone_faces_idx" in human_seq:
        bone_faces_idx = int(human_seq["bone_faces_idx"])
        bone_faces = subject_param["faces"][:bone_faces_idx].to(device).long()  # indices into bone_cano
        if bone_faces.shape[0] > 0 and interior_canonical.shape[0] > 0:
            inside_bone = _generalized_winding_number(interior_canonical, bone_cano, bone_faces)
            n_before = interior_canonical.shape[0]
            interior_canonical = interior_canonical[~inside_bone]
            n_after = interior_canonical.shape[0]
            print(f"[filling_warp][canonical] excluded {n_before - n_after} bone-interior points; remaining={n_after}")

    if interior_canonical.shape[0] == 0:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        empty_lbs = torch.empty((0, 55), device=device, dtype=torch.float32)
        return empty, empty, empty_lbs

    # 4. k-NN in canonical space (no fold issue here — T-pose) → reliable LBS
    # config-driven k: 1 = rigid binding to nearest surface vertex (each interior
    # particle inherits one surface vert's exact LBS — strongest spatial consistency
    # with the surface kinematic field, so the F=Fe·Fk decomposition cancels cleanly).
    # Higher k (default 4) gives smoother spatial blending of LBS but can be discontinuous
    # at body-part Voronoi boundaries → leaks kinematic motion into elastic F.
    k_eff = int(fp.get("interior_lbs_k", k_lbs))
    smplx_model = human_seq["smplx_model"]
    lbs_full = smplx_model.lbs_weights.to(device).float()
    knn = knn_points(
        interior_canonical.unsqueeze(0).contiguous(),
        cano_pts.unsqueeze(0).contiguous(),
        K=min(k_eff, surface_n),
    )
    knn_idx = knn.idx[0]
    if k_eff == 1:
        # rigid-binding: copy the single nearest surface vertex's LBS exactly
        interior_lbs = lbs_full[knn_idx[:, 0]].contiguous()
    else:
        knn_d2 = knn.dists[0].clamp_min(0.0)
        weights = 1.0 / (knn_d2.sqrt() + eps)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        interior_lbs = (weights.unsqueeze(-1) * lbs_full[knn_idx]).sum(dim=1).contiguous()
    print(f"[filling_warp][canonical] interior LBS k_eff={k_eff}")

    # 5. First-frame joint matrices, MIRRORING the A_now_55 hack convention used in
    # compute_smplx_velocity_tgt (mpm_human_utils_separable_contact.py:2355-2368).
    # Otherwise: my fill's `live_smpl.A[0, 22:55]` (true SMPL-X jaw/eye/finger A)
    # disagrees with velocity computation's A_now_55[22:55] (head-copy + only_finger
    # hack) → interior particles with LBS weight on those joints get a false target
    # every step → body breaks. Both forward-LBS pipelines must use the SAME 55-joint
    # convention.
    pose_dataset = human_seq["pose_dataset"]
    betas = human_seq["betas"]
    first_idx = pose_dataset.pose_list[0]
    live_smpl = smplx_model.forward(
        betas=betas,
        global_orient=pose_dataset.body_poses[first_idx, :3][None],
        transl=pose_dataset.transl[first_idx][None],
        body_pose=pose_dataset.body_poses[first_idx, 3:66][None],
        left_hand_pose=pose_dataset.left_hand_pose[first_idx][None].to(device),
        right_hand_pose=pose_dataset.right_hand_pose[first_idx][None].to(device),
    )
    only_finger = smplx_model.forward(
        betas=betas,
        global_orient=torch.zeros([1, 3], device=device),
        transl=torch.zeros([1, 3], device=device),
        body_pose=torch.zeros([1, 63], device=device),
        left_hand_pose=pose_dataset.left_hand_pose[first_idx][None].to(device),
        right_hand_pose=pose_dataset.right_hand_pose[first_idx][None].to(device),
    )
    # Reproduce A_now_55 hack from velocity compute, but at first frame
    A_first_55 = torch.zeros((55, 4, 4), device=device, dtype=live_smpl.A.dtype)
    A_first_55[:22] = live_smpl.A[0, :22]
    A_first_55[22:25] = live_smpl.A[0, 15]
    A_first_55[25:40] = live_smpl.A[0, 20] @ only_finger.A[0, 25:40]
    A_first_55[40:55] = live_smpl.A[0, 21] @ only_finger.A[0, 40:55]

    # 6. Forward LBS using the same 55-joint convention
    pt_mats_first = torch.einsum('nj,jxy->nxy', interior_lbs, A_first_55)        # [N_int, 4, 4]
    interior_first_frame = (
        torch.einsum('nxy,ny->nx', pt_mats_first[..., :3, :3], interior_canonical)
        + pt_mats_first[..., :3, 3]
    )                                              # [N_int, 3]

    return interior_first_frame, interior_canonical, interior_lbs


def _propagate_attrs_via_knn(
    surface_pos: torch.Tensor,    # [Ns, 3]
    interior_pos: torch.Tensor,   # [Ni, 3]
    surface_attrs: dict,           # {key: tensor[Ns, ...]} per attribute
    k: int = 1,
    eps: float = 1e-8,
):
    """Propagate per-vertex attributes from surface particles to interior
    particles by k-NN (Euclidean) on surface_pos. Replaces filling_new
    Taichi get_attr_from_closest with a pytorch3d.knn_points call."""
    from pytorch3d.ops import knn_points
    Ni = interior_pos.shape[0]
    if Ni == 0:
        return {key: torch.empty((0,) + v.shape[1:], dtype=v.dtype, device=v.device)
                for key, v in surface_attrs.items()}
    knn = knn_points(
        interior_pos.unsqueeze(0).contiguous(),
        surface_pos.unsqueeze(0).contiguous(),
        K=k,
    )
    knn_idx = knn.idx[0]      # [Ni, k]
    if k == 1:
        gather_idx = knn_idx[:, 0]  # [Ni]
        out = {key: v[gather_idx] for key, v in surface_attrs.items()}
        return out
    knn_d2 = knn.dists[0].clamp_min(0.0)         # [Ni, k]
    w = 1.0 / (knn_d2.sqrt() + eps)
    w = w / w.sum(dim=-1, keepdim=True)          # [Ni, k]
    out = {}
    for key, v in surface_attrs.items():
        gathered = v[knn_idx]                    # [Ni, k, ...]
        # broadcast w to gathered shape
        w_bcast = w.view(Ni, k, *([1] * (gathered.dim() - 2)))
        out[key] = (w_bcast * gathered.float()).sum(dim=1).to(v.dtype)
    return out


def fill_particles_subjects_warp(subjects, subject_params, sim_params, device='cuda',
                                 human_sequences=None):
    """Drop-in replacement for filling_new.fill_particles_subjects.
    - Runs Warp filling for any subject with `particle_filling` block.
    - For human subjects with a triangle mesh (SMPL-X faces in subject_param),
      filters the candidate interior particles through trimesh.contains() so that
      only points truly inside the surface mesh survive. This compensates for the
      density-halo approach over-detecting near the surface (esp. with smooth=true).
    - Always extends subject['cov'/'opacity'/'shs'/'index'] to match the
      new pos length, using nearest-surface-vertex copy (k=1 by default).
      This makes the no-`visualize` path work correctly downstream
      (merge_subjects requires aligned-length arrays).
    """
    init_gs_nums = []
    for i, (subject, param) in tqdm(list(enumerate(zip(subjects, subject_params)))):
        gs_num = subject['pos'].shape[0]
        init_gs_nums.append(gs_num)

        if "particle_filling" in param.keys():
            fp = param["particle_filling"]
            # Human SMPL-X path: canonical-space fill + forward-LBS so the resulting
            # interior particles are mathematically consistent with the LBS pipeline.
            if (param.get("human", False)
                    and human_sequences is not None
                    and human_sequences[i] is not None
                    and "smplx_model" in human_sequences[i]
                    and "faces_only_smplx" in param):
                interior_first_frame, interior_canonical, interior_lbs = _fill_human_canonical(
                    human_sequences[i], param, sim_params, device=device,
                )
                # stash on smplx_model so compute_smplx_velocity_* picks them up
                smplx_model = human_sequences[i]["smplx_model"]
                smplx_model._interior_canonical = interior_canonical
                smplx_model._interior_lbs = interior_lbs
                # build the new subject pos by appending interior_first_frame
                new_pos = torch.cat([subject['pos'].to(device), interior_first_frame], dim=0)
                n_interior = interior_first_frame.shape[0]
                print(f"[filling_warp] subject {i}: canonical-space fill, surface={gs_num}, interior added={n_interior}")
                # Propagate cov/opacity/shs/index/screen_points (k=1 nearest in canonical)
                if n_interior > 0 and gs_num > 0:
                    surface_pos_first = subject['pos'].to(device)
                    interior_pos_first = interior_first_frame.to(device)
                    attrs = {
                        "cov":     subject["cov"].to(device),
                        "opacity": subject["opacity"].to(device),
                        "shs":     subject["shs"].to(device),
                        "index":   subject["index"].to(device),
                    }
                    propagated = _propagate_attrs_via_knn(
                        surface_pos=surface_pos_first,
                        interior_pos=interior_pos_first,
                        surface_attrs=attrs,
                        k=1,
                    )
                    subject["cov"]     = torch.cat([attrs["cov"],     propagated["cov"]],     dim=0)
                    subject["opacity"] = torch.cat([attrs["opacity"], propagated["opacity"]], dim=0)
                    subject["shs"]     = torch.cat([attrs["shs"],     propagated["shs"]],     dim=0)
                    subject["index"]   = torch.cat([attrs["index"],   propagated["index"]],   dim=0)
                    sp_zeros = torch.zeros(
                        (n_interior,) + tuple(subject["screen_points"].shape[1:]),
                        dtype=subject["screen_points"].dtype,
                        device=subject["screen_points"].device,
                    )
                    subject["screen_points"] = torch.cat([subject["screen_points"], sp_zeros], dim=0)
                subject['pos'] = new_pos

                if sim_params.get("debug", False):
                    particle_position_tensor_to_ply(
                        subject['pos'],
                        f"./log/{sim_params['name']}/filled_particles_{i:02d}.ply",
                    )
                continue  # skip the legacy posed-space fill below

            new_pos = fill_particles_warp(
                pos=subject['pos'],
                opacity=subject['opacity'],
                cov=subject["cov"],
                grid_n=fp["n_grid"],
                max_samples=fp["max_particles_num"],
                grid_dx=sim_params["grid_lim"] / fp["n_grid"],
                density_thres=fp["density_threshold"],
                search_thres=fp["search_threshold"],
                max_particles_per_cell=fp["max_partciels_per_cell"],
                search_exclude_dir=fp["search_exclude_direction"],
                ray_cast_dir=fp["ray_cast_direction"],
                boundary=fp.get("boundary", None),
                smooth=fp.get("smooth", False),
                device=device,
            ).to(device=device)

            n_candidates = new_pos.shape[0] - gs_num
            print(f"[filling_warp] subject {i}: surface={gs_num}, candidates={n_candidates}")

            # Mesh-based filter for human subjects: keep only candidates strictly
            # inside the SMPL-X surface mesh, via generalized winding number on GPU
            # (avoids the rtree dependency that trimesh.contains() needs).
            if (param.get("human", False)
                    and "faces_only_smplx" in param
                    and human_sequences is not None
                    and human_sequences[i] is not None
                    and "pos_pts" in human_sequences[i]
                    and n_candidates > 0):
                hs = human_sequences[i]
                bone_n = hs["bone_cano"].shape[0]
                surface_n = hs["cano_pts"].shape[0]
                surface_pts_first = hs["pos_pts"][bone_n : bone_n + surface_n].to(device).float()
                faces = param["faces_only_smplx"].to(device).long()
                cand_pts = new_pos[gs_num:].to(device).float()
                inside_mask = _generalized_winding_number(cand_pts, surface_pts_first, faces)
                n_kept = int(inside_mask.sum().item())
                print(f"[filling_warp] subject {i}: kept {n_kept}/{n_candidates} after winding-number inside-mesh filter")
                if n_kept < n_candidates:
                    interior_kept = new_pos[gs_num:][inside_mask]
                    new_pos = torch.cat([new_pos[:gs_num], interior_kept], dim=0)

            n_interior = new_pos.shape[0] - gs_num
            print(f"[filling_warp] subject {i}: final interior added={n_interior}")
            subject['pos'] = new_pos

            if sim_params.get("debug", False):
                particle_position_tensor_to_ply(
                    subject['pos'],
                    f"./log/{sim_params['name']}/filled_particles_{i:02d}.ply",
                )

            if n_interior > 0:
                surface_pos = new_pos[:gs_num].to(device)
                interior_pos = new_pos[gs_num:].to(device)
                attrs = {
                    "cov":     subject["cov"].to(device),
                    "opacity": subject["opacity"].to(device),
                    "shs":     subject["shs"].to(device),
                    "index":   subject["index"].to(device),
                }
                # screen_points are zeros-with-grad; just allocate matching shape
                propagated = _propagate_attrs_via_knn(
                    surface_pos=surface_pos,
                    interior_pos=interior_pos,
                    surface_attrs=attrs,
                    k=fp.get("attr_knn_k", 1),
                )
                subject["cov"]     = torch.cat([attrs["cov"],     propagated["cov"]],     dim=0)
                subject["opacity"] = torch.cat([attrs["opacity"], propagated["opacity"]], dim=0)
                subject["shs"]     = torch.cat([attrs["shs"],     propagated["shs"]],     dim=0)
                subject["index"]   = torch.cat([attrs["index"],   propagated["index"]],   dim=0)

                # extend screen_points so merge_subjects slicing works
                sp_zeros = torch.zeros(
                    (n_interior,) + tuple(subject["screen_points"].shape[1:]),
                    dtype=subject["screen_points"].dtype,
                    device=subject["screen_points"].device,
                )
                subject["screen_points"] = torch.cat([subject["screen_points"], sp_zeros], dim=0)
        else:
            # mirror filling_new behaviour: just pad cov to [N, 6] if needed
            mpm_init_cov = torch.zeros((subject['pos'].shape[0], 6), device=device)
            mpm_init_cov[:gs_num] = subject['cov']
            subject['cov'] = mpm_init_cov

    return subjects, init_gs_nums


def get_particle_volume_from_subjects_warp(subjects, subject_params, sim_params,
                                            init_gs_nums=None, device='cuda'):
    """Per-particle volume in MERGE_SUBJECTS order, i.e. [surface_0, surface_1, ...,
    interior_0, interior_1, ...]. This matches mpm_params['pos'] from merge_subjects.

    Without init_gs_nums (legacy single-subject use): per-subject concat, which only
    coincides with merge_subjects order when there's a single subject (since the second
    pass of merge_subjects has nothing to interleave). For multi-subject + filling,
    NOT passing init_gs_nums silently misaligns vol↔pos and corrupts particle masses.
    """
    per_subject_vols = []
    for subject, params in zip(subjects, subject_params):
        vol = _get_particle_volume_warp(
            subject["pos"].to(device),
            sim_params["n_grid"],
            sim_params["grid_lim"] / sim_params["n_grid"],
            uniform=(params["material"] == "sand"),
            device=device,
        )
        per_subject_vols.append(vol)

    if init_gs_nums is None:
        # legacy: simple per-subject concatenation; only correct for single subject
        # (single-subject's [bone, surface, interior] order coincides with merge_subjects)
        return torch.cat(per_subject_vols, dim=0)

    # Reorder to match merge_subjects: surfaces first (per subject), then interiors (per subject)
    surface_vols = []
    interior_vols = []
    for vol, n_surf in zip(per_subject_vols, init_gs_nums):
        surface_vols.append(vol[:n_surf])
        if vol.shape[0] > n_surf:
            interior_vols.append(vol[n_surf:])
    return torch.cat(surface_vols + interior_vols, dim=0)


def _get_particle_volume_warp(pos: torch.Tensor, grid_n: int, grid_dx: float,
                              uniform: bool = False, device: str = "cuda"):
    pos_f = pos.detach().contiguous().to(dtype=torch.float32, device=device)
    wp_pos = _torch_to_wp_vec3(pos_f)
    grid = wp.zeros(shape=(grid_n, grid_n, grid_n), dtype=int, device=device)
    vol = wp.zeros(shape=(pos_f.shape[0],), dtype=float, device=device)
    wp.launch(
        kernel=assign_particle_to_grid_warp,
        dim=pos_f.shape[0],
        inputs=[wp_pos, grid, grid_n, grid_dx],
        device=device,
    )
    wp.launch(
        kernel=compute_particle_volume_warp,
        dim=pos_f.shape[0],
        inputs=[wp_pos, grid, vol, grid_n, grid_dx],
        device=device,
    )
    vol_t = wp.to_torch(vol).clone()
    if uniform:
        vol_t = vol_t.mean().repeat(pos_f.shape[0])
    return vol_t
