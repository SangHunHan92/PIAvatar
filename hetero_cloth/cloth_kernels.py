"""Cloth-specific Warp kernels ported from MPMAvatar.

Kernel suite for the cloth model only - operates on cloth elements and
vertices that live at indices [n_existing, n_existing + n_elements + n_vertices)
inside the user's MPMStateStruct.particle_x / particle_v / etc. arrays.

Design:
- All cloth-specific per-element fields (particle_d, R_inv, D_inv, gamma, kappa,
  faces) and per-vertex field (vertex_force) live in ClothStateStruct. They are
  indexed locally (0 .. n_elements or 0 .. n_vertices); kernels use the offsets
  cloth_state.n_existing / .n_elements / .n_vertices to translate.
- Element thickness stress is written into state.particle_stress[element_idx]
  exactly as MPMAvatar does - this lets the user's existing P2G stress-divergence
  path handle the thickness coupling without modification.
- In-plane stress is FEM-assembled into cloth_state.vertex_force[v_local] and
  must be P2G'd separately (see p2g_cloth_vertex_force kernel).

Integration sequence per substep (driven from cloth_solver.py):
    1. zero_vertex_force_kernel             # clear vertex_force buffer
    2. user's compute_stress_from_F_trial   # for traditional particles only
                                            # (cloth elements/vertices have particle_stress=0 at this point)
    3. compute_cloth_stress_kernel          # for cloth elements:
                                            #   - applies plastic return mapping on particle_d
                                            #   - writes thickness stress to state.particle_stress[elem]
                                            #   - atomic-adds in-plane FEM forces to cloth_state.vertex_force[v]
                                            # Cloth vertices keep particle_stress=0 (never written).
    4. user's p2g (over ALL particles)      # mass + APIC velocity for everyone;
                                            # stress-divergence using state.particle_stress -
                                            # works correctly because:
                                            #   traditional: user's stress (set in step 2)
                                            #   element:     thickness stress (set in step 3)
                                            #   vertex:      0 (zero stress -> zero force)
    5. p2g_vertex_force_kernel              # adds the in-plane FEM contribution onto grid_v_in
                                            # via force = weight * vertex_force[v_local], for cloth vertices
    6. user's grid update (gravity, damping, collider, separable_contact)
    7. user's g2p (over all particles, including cloth vertices - they get v/x/C from grid)
    8. g2p_cloth_elements_kernel            # cloth element kinematics from vertex average + d update

Notes on initialisation invariants (set up by cloth_init.py):
    - cloth element particle_stress is zero-initialised; only compute_cloth_stress_kernel writes it.
    - cloth vertex particle_stress remains zero throughout the simulation.
    - cloth vertex particle_mass / particle_vol / particle_x / particle_v are set normally so
      user's P2G transfers vertex mass+momentum to the grid.
    - cloth element particle_mass should be 0 (or very small) - elements are pure stress
      carriers, no mass-momentum role; their kinematics come from vertex average post-G2P.
"""
import warp as wp

from mpm_solver_warp.warp_utils_separable_contact import MPMStateStruct, MPMModelStruct
from .cloth_state import ClothStateStruct


# ---------------------------------------------------------------------------
# Helper @wp.func - direct copy from third_party/MPMAvatar/warp_mpm/mpm_utils.py
# ---------------------------------------------------------------------------

@wp.func
def inverse_lower_triangle(M: wp.mat33):
    M11 = M[0, 0]
    M21 = M[1, 0]
    M22 = M[1, 1]
    M31 = M[2, 0]
    M32 = M[2, 1]
    M33 = M[2, 2]
    invdet = 1.0 / (M11 * M22 * M33)
    return invdet * wp.mat33(
        M22 * M33, 0.0, 0.0,
        -M21 * M33, M11 * M33, 0.0,
        M21 * M32 - M31 * M22, -M11 * M32, M11 * M22,
    )


# ---------------------------------------------------------------------------
# Anisotropic membrane stress - adapted to take ClothStateStruct.
#
# In MPMAvatar this writes vertex_force directly via state.vertex_force.
# Here we split: returns thickness stress (for state.particle_stress[elem]),
# and atomic-adds in-plane FEM forces into cloth_state.vertex_force[v_local].
#
# v_local = (face_idx_v - n_existing - n_elements)  : local index into vertex_force
# Faces are stored as wp.vec3 of (v_global, v_global, v_global) in cloth_state.faces.
# ---------------------------------------------------------------------------

@wp.func
def kirchoff_stress_Anisotropy_cloth(
    R_inv: wp.vec3,
    d: wp.mat33,
    face: wp.vec3,
    vol: float,
    cloth_state: ClothStateStruct,
    mu: float,
    lam: float,
    gamma: float,
    kappa: float,
):
    iD11 = R_inv[0]
    iD12 = R_inv[1]
    iD22 = R_inv[2]

    # QR-decompose deformation matrix d; sign-correct so R[0,0], R[1,1] >= 0
    Q_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.qr3(d, Q_0, R_0)
    if R_0[0, 0] < 0.0:
        Q_1 = wp.mat33(-Q_0[0, 0], Q_0[0, 1], -Q_0[0, 2],
                       -Q_0[1, 0], Q_0[1, 1], -Q_0[1, 2],
                       -Q_0[2, 0], Q_0[2, 1], -Q_0[2, 2])
        R_1 = wp.mat33(-R_0[0, 0], -R_0[0, 1], -R_0[0, 2],
                       0.0, R_0[1, 1], R_0[1, 2],
                       0.0, 0.0, -R_0[2, 2])
    else:
        Q_1 = Q_0
        R_1 = R_0
    if R_1[1, 1] < 0.0:
        Q = wp.mat33(Q_1[0, 0], -Q_1[0, 1], -Q_1[0, 2],
                     Q_1[1, 0], -Q_1[1, 1], -Q_1[1, 2],
                     Q_1[2, 0], -Q_1[2, 1], -Q_1[2, 2])
        R = wp.mat33(R_1[0, 0], R_1[0, 1], R_1[0, 2],
                     0.0, -R_1[1, 1], -R_1[1, 2],
                     0.0, 0.0, -R_1[2, 2])
    else:
        Q = Q_1
        R = R_1

    F11 = R[0, 0] * iD11
    F12 = R[0, 0] * iD12 + R[0, 1] * iD22
    F22 = R[1, 1] * iD22
    F2 = wp.mat22(F11, F12, 0.0, F22)

    RiDT = wp.mat33(F11, 0.0, 0.0,
                    F12, F22, 0.0,
                    R[0, 2], R[1, 2], R[2, 2])
    iFTJ = wp.mat22(F22, 0.0, -F12, F11)

    # 2D corotational membrane stress
    U3 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V3 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig3 = wp.vec3(0.0)
    F3 = wp.mat33(F11, F12, 0.0, 0.0, F22, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(F3, U3, sig3, V3)
    U = wp.mat22(U3[0, 0], U3[0, 1], U3[1, 0], U3[1, 1])
    V = wp.mat22(V3[0, 0], V3[0, 1], V3[1, 0], V3[1, 1])

    Rot = U * wp.transpose(V)
    J = F11 * F22

    K2 = 2.0 * mu * (F2 - Rot) + lam * (J - 1.0) * iFTJ

    dr11 = K2[0, 0]
    dr12 = K2[0, 1]
    dr22 = K2[1, 1]
    dr13 = gamma * R[0, 2]
    dr23 = gamma * R[1, 2]
    if R[2, 2] > 1.0:
        dr33 = 0.0
    else:
        dr33 = -kappa * (1.0 - R[2, 2]) * (1.0 - R[2, 2])

    dr = wp.mat33(dr11, dr12, dr13, 0.0, dr22, dr23, 0.0, 0.0, dr33)
    K3 = dr * RiDT
    K3_sym = wp.mat33(K3[0, 0], K3[0, 1], K3[0, 2],
                      K3[0, 1], K3[1, 1], K3[1, 2],
                      K3[0, 2], K3[1, 2], K3[2, 2])
    RiDT_inv = inverse_lower_triangle(RiDT)
    P = Q * K3_sym * RiDT_inv

    P1 = wp.vec3(P[0, 0], P[1, 0], P[2, 0])
    P2 = wp.vec3(P[0, 1], P[1, 1], P[2, 1])
    P3 = wp.vec3(P[0, 2], P[1, 2], P[2, 2])
    d3 = wp.vec3(d[0, 2], d[1, 2], d[2, 2])

    # in-plane FEM nodal forces -> vertex_force buffer
    f2 = -vol * (iD11 * P1 + iD12 * P2)
    f3 = -vol * iD22 * P2
    f1 = -(f2 + f3)

    # face stores GLOBAL particle indices for v1, v2, v3.
    # local index into vertex_force = global - n_existing - n_elements
    base = cloth_state.n_existing + cloth_state.n_elements
    v1 = int(face[0]) - base
    v2 = int(face[1]) - base
    v3 = int(face[2]) - base
    wp.atomic_add(cloth_state.vertex_force, v1, f1)
    wp.atomic_add(cloth_state.vertex_force, v2, f2)
    wp.atomic_add(cloth_state.vertex_force, v3, f3)

    # thickness stress (returned to be stored in state.particle_stress[elem_idx])
    return vol * wp.outer(P3, d3)


# ---------------------------------------------------------------------------
# Anisotropy plastic return mapping - same body as MPMAvatar; reads
# cloth_state.kappa/.gamma/.friction_coeff instead of MPMModelStruct.
# elem_local: index in [0, n_elements)
# ---------------------------------------------------------------------------

@wp.func
def anisotropy_return_mapping_cloth(
    d: wp.mat33,
    cloth_state: ClothStateStruct,
    elem_local: int,
):
    Q_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.qr3(d, Q_0, R_0)
    if R_0[0, 0] < 0.0:
        Q_1 = wp.mat33(-Q_0[0, 0], Q_0[0, 1], -Q_0[0, 2],
                       -Q_0[1, 0], Q_0[1, 1], -Q_0[1, 2],
                       -Q_0[2, 0], Q_0[2, 1], -Q_0[2, 2])
        R_1 = wp.mat33(-R_0[0, 0], -R_0[0, 1], -R_0[0, 2],
                       0.0, R_0[1, 1], R_0[1, 2],
                       0.0, 0.0, -R_0[2, 2])
    else:
        Q_1 = Q_0
        R_1 = R_0
    if R_1[1, 1] < 0.0:
        Q = wp.mat33(Q_1[0, 0], -Q_1[0, 1], -Q_1[0, 2],
                     Q_1[1, 0], -Q_1[1, 1], -Q_1[1, 2],
                     Q_1[2, 0], -Q_1[2, 1], -Q_1[2, 2])
        R_2 = wp.mat33(R_1[0, 0], R_1[0, 1], R_1[0, 2],
                       0.0, -R_1[1, 1], -R_1[1, 2],
                       0.0, 0.0, -R_1[2, 2])
    else:
        Q = Q_1
        R_2 = R_1
    if R_2[2, 2] > 1.0:
        # plastic clamp: thickness can't stretch beyond rest
        R = wp.mat33(R_2[0, 0], R_2[0, 1], R_2[0, 2],
                     R_2[1, 0], R_2[1, 1], R_2[1, 2],
                     0.0, 0.0, 1.0)
    else:
        fn = cloth_state.kappa[elem_local] * (1.0 - R_2[2, 2]) * (1.0 - R_2[2, 2])
        ff = cloth_state.gamma[elem_local] * wp.sqrt(R_2[0, 2] * R_2[0, 2] + R_2[1, 2] * R_2[1, 2])
        if ff > cloth_state.friction_coeff * fn:
            scale = cloth_state.friction_coeff * fn / ff
            R = wp.mat33(R_2[0, 0], R_2[0, 1], R_2[0, 2] * scale,
                         R_2[1, 0], R_2[1, 1], R_2[1, 2] * scale,
                         R_2[2, 0], R_2[2, 1], R_2[2, 2])
        else:
            R = R_2

    d3 = Q * wp.vec3(R[0, 2], R[1, 2], R[2, 2])
    new_d = wp.mat33(d[0, 0], d[0, 1], d3[0],
                     d[1, 0], d[1, 1], d3[1],
                     d[2, 0], d[2, 1], d3[2])
    return new_d


# ---------------------------------------------------------------------------
# Kernels (launched per substep)
# ---------------------------------------------------------------------------

@wp.kernel
def apply_cloth_pin_velocity_kernel(
    state: MPMStateStruct,
    cloth_state: ClothStateStruct,
):
    """For each pinned cloth vertex (pin_mask==1), overwrite particle_v with
    cloth_state.pin_target_v (LBS-driven velocity in MPM space, computed once
    per frame in Python from pose_dataset's now/next pose).

    Mirrors body's kinematic_velocity injection so pinned cloth verts follow
    body kinematically while free verts continue MPM dynamics. Also writes to
    particle_vk to keep separable-contact subject velocity aligned.

    Launch dim = n_vertices.
    """
    v_local = wp.tid()
    if cloth_state.pin_mask[v_local] == 0:
        return
    p = cloth_state.n_existing + cloth_state.n_elements + v_local
    if state.particle_selection[p] != 0:
        return
    target_v = cloth_state.pin_target_v[v_local]
    state.particle_v[p]   = target_v
    state.particle_vk[p]  = target_v
    state.particle_vko[p] = target_v


@wp.kernel
def zero_vertex_force_kernel(cloth_state: ClothStateStruct):
    """Zero cloth_state.vertex_force at the start of each substep
    (before compute_cloth_stress accumulates new FEM nodal forces).
    Launch dim = n_vertices.
    """
    v = wp.tid()
    cloth_state.vertex_force[v] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_cloth_vertex_stress_kernel(
    state: MPMStateStruct,
    cloth_state: ClothStateStruct,
):
    """Zero state.particle_stress for cloth vertex particles only.

    Defensive: user's compute_stress_from_F_trial runs over all particles with
    particle_selection==0, including cloth vertices. With particle_F=I the
    user's elastic stress is naturally 0, but if F drifts (g2p updates F every
    step) the user's kernel could write nonzero stress that user's P2G would
    treat as a real force. Zero it explicitly so cloth vertices contribute
    only via vertex_force (in p2g_vertex_force_kernel) and never via stress.

    Launch dim = n_vertices.
    """
    v_local = wp.tid()
    p = cloth_state.n_existing + cloth_state.n_elements + v_local
    state.particle_stress[p] = wp.mat33(0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0)


@wp.kernel
def compute_cloth_stress_kernel(
    state: MPMStateStruct,
    cloth_state: ClothStateStruct,
    model: MPMModelStruct,
    dt: float,
):
    """For each cloth element: apply plastic return mapping on particle_d,
    compute anisotropic stress (in-plane -> vertex_force, thickness -> particle_stress).

    Launch dim = n_elements; thread idx = element_local in [0, n_elements).
    Global particle index of this element: p = elem_local + cloth_state.n_existing.
    """
    elem_local = wp.tid()
    p = elem_local + cloth_state.n_existing

    if state.particle_selection[p] != 0:
        return

    # plastic return mapping on d
    cloth_state.particle_d[elem_local] = anisotropy_return_mapping_cloth(
        cloth_state.particle_d[elem_local], cloth_state, elem_local
    )

    # anisotropic stress: in-plane forces -> vertex_force,  thickness stress -> particle_stress[p]
    thickness_stress = kirchoff_stress_Anisotropy_cloth(
        cloth_state.particle_R_inv[elem_local],
        cloth_state.particle_d[elem_local],
        cloth_state.faces[elem_local],
        state.particle_vol[p],
        cloth_state,
        model.mu[p],
        model.lam[p],
        cloth_state.gamma[elem_local],
        cloth_state.kappa[elem_local],
    )
    state.particle_stress[p] = thickness_stress


@wp.kernel
def p2g_vertex_force_kernel(
    state: MPMStateStruct,
    cloth_state: ClothStateStruct,
    model: MPMModelStruct,
    dt: float,
):
    """Inject the FEM vertex_force into grid_v_in for each cloth vertex.
    force = weight * vertex_force[v_local].

    User's existing P2G already handles mass + APIC velocity transfer for cloth
    vertices (with particle_stress=0 -> zero stress-divergence force), so this
    kernel only ADDS the missing in-plane FEM contribution.

    Launch dim = n_vertices; thread idx = v_local in [0, n_vertices).
    Run AFTER user's P2G, BEFORE grid update.
    """
    v_local = wp.tid()
    p = cloth_state.n_existing + cloth_state.n_elements + v_local

    if state.particle_selection[p] != 0:
        return

    vertex_force = cloth_state.vertex_force[v_local]

    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )

    sid = state.particle_id[p]
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]
                v_in_add = dt * weight * vertex_force
                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                # separable-contact: also scatter to per-subject momentum grid so
                # the FEM force lands in cloth's own subject layer rather than
                # being averaged with body's mass-weighted velocity.
                if model.use_separable_contact == 1:
                    wp.atomic_add(state.grid_p_in_s, ix, iy, iz, sid, v_in_add)


@wp.kernel
def g2p_cloth_elements_kernel(
    state: MPMStateStruct,
    cloth_state: ClothStateStruct,
    model: MPMModelStruct,
    dt: float,
):
    """Update cloth element kinematics and deformation matrix d.
    - particle_v / particle_x of element = mean of its three vertices' v / x
      (vertices' v/x are already updated by user's g2p kernel running over all particles).
    - d update: first two columns from current vertex deltas, third column by
      grid velocity gradient applied to previous d3.

    Launch dim = n_elements; thread idx = elem_local.
    Run AFTER user's g2p kernel that updates vertex v/x.
    """
    elem_local = wp.tid()
    p = elem_local + cloth_state.n_existing

    if state.particle_selection[p] != 0:
        return

    # gather grid velocity gradient (new_F) - only the thickness column needs it
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

    sid = state.particle_id[p]
    new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                weight = w[0, i] * w[1, j] * w[2, k]
                # separable-contact: read cloth's own subject layer (post-contact-resolve)
                # so element kinematics aren't dragged by body's grid velocity.
                if model.use_separable_contact == 1:
                    grid_v = state.grid_v_resolved[ix, iy, iz, sid]
                else:
                    grid_v = state.grid_v_out[ix, iy, iz]
                new_C = new_C + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
                dw_i = dw[0, i] * w[1, j] * w[2, k] * model.inv_dx
                dw_j = w[0, i] * dw[1, j] * w[2, k] * model.inv_dx
                dw_k = w[0, i] * w[1, j] * dw[2, k] * model.inv_dx
                dweight = wp.vec3(dw_i, dw_j, dw_k)
                new_F = new_F + wp.outer(grid_v, dweight)

    # element kinematics from vertex average
    face = cloth_state.faces[elem_local]
    v1 = int(face[0])
    v2 = int(face[1])
    v3 = int(face[2])

    state.particle_v[p] = (state.particle_v[v1] + state.particle_v[v2] + state.particle_v[v3]) / 3.0
    state.particle_x[p] = (state.particle_x[v1] + state.particle_x[v2] + state.particle_x[v3]) / 3.0
    state.particle_C[p] = new_C

    # update particle_d: first two columns from vertex edges (in-plane),
    # third column by grid velocity gradient applied to previous d3 (thickness).
    d1 = state.particle_x[v2] - state.particle_x[v1]
    d2 = state.particle_x[v3] - state.particle_x[v1]
    d_old = cloth_state.particle_d[elem_local]
    d3_old = wp.vec3(d_old[0, 2], d_old[1, 2], d_old[2, 2])
    I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    d3_new = (I3 + new_F * dt) * d3_old

    new_d = wp.mat33(d1[0], d2[0], d3_new[0],
                     d1[1], d2[1], d3_new[1],
                     d1[2], d2[2], d3_new[2])
    cloth_state.particle_d[elem_local] = new_d
