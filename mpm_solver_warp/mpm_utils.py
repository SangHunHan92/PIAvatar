import warp as wp
from warp_utils import *
import numpy as np
import math

@wp.func
def _flip_last_row(M: wp.mat33) -> wp.mat33:
    # helper: 마지막 행 부호 반전 (det 보정용)
    return wp.mat33(
        M[0,0], M[0,1], M[0,2],
        M[1,0], M[1,1], M[1,2],
       -M[2,0],-M[2,1],-M[2,2]
    )

@wp.func
def _make_R_with_det_fix(U: wp.mat33, V: wp.mat33) -> wp.mat33:
    # R = U V^T, det(R) < 0면 V^T의 마지막 행을 뒤집어 재계산
    Vt = wp.transpose(V)
    R = U * Vt
    detR = wp.determinant(R)
    if detR < 0.0:
        Vt = _flip_last_row(Vt)
        R = U * Vt
    return R

@wp.func
def kirchhoff_stress_corotated_hybrid(  # returns Cauchy stress σ
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # 1) R (det 보정 포함)
    R = _make_R_with_det_fix(U, V)

    # 2) 국소 프레임
    Uloc = wp.transpose(R) * F

    # 3) Ec = sym(Uloc) - I
    I = wp.mat33(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
    Ec = 0.5 * (Uloc + wp.transpose(Uloc)) - I
    # (선택) 소신호 클램프 – 큰 λ, μ에서 잡음 억제용
    # eps = 1e-4
    # Ec = clamp_sym(Ec, eps)

    # 4) 전단: corotated
    tau_shear = (2.0 * mu) * Ec

    # 5) 체적: Neo-Hookean 스타일 (kappa = K = lam + 2μ/3)
    kappa = lam + 2.0 * mu / 3.0
    tau_vol = (kappa * (J - 1.0) * J) * I

    # 6) Kirchhoff → Cauchy
    tau = tau_shear + tau_vol
    sigma = (1.0 / J) * tau
    return sigma

@wp.func
def kirchhoff_stress_corotated(  # returns Cauchy stress σ
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # 1) R (det 보정)
    R = _make_R_with_det_fix(U, V)

    # 2) 국소 프레임
    Uloc = wp.transpose(R) * F

    # 3) Ec = sym(Uloc) - I
    I = wp.mat33(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
    Ec = 0.5 * (Uloc + wp.transpose(Uloc)) - I
    trEc = Ec[0,0] + Ec[1,1] + Ec[2,2]

    # 4) Kirchhoff (co-rotated)
    tau = (2.0 * mu) * Ec + (lam * trEc) * I

    # 5) Cauchy로 변환
    sigma = (1.0 / J) * tau
    return sigma

# compute stress from F
@wp.func
def kirchoff_stress_FCR( # Finite-strain Co-Rotated
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float, p: int
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)    
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)
    # if p == 0:
    #     print("R")
    #     print(R)
    #     print("J")      
    #     print(J)
    #     print("stress")
    #     print(2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0))


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK( # Saint Venant–Kirchhoff, Hencky/Log strain
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic

@wp.kernel
def set_material_paramater(state: MPMStateStruct, index: int, material: int):
    p = wp.tid()
    if state.particle_id[p] == index:
        state.particle_material[p] = material

@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z, k = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0    
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    
    state.grid_vk[grid_x, grid_y, grid_z, k] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_mk[grid_x, grid_y, grid_z, k] = 0.0    
    # state.grid_id[grid_x, grid_y, grid_z, k] = 0x7fffffff # SENTINEL = 0x7fffffff, -1
    # state.grid_count[grid_x, grid_y, grid_z] = 0
    
    # state.grid_v_in_prescribed[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_out_prescribed[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_mean_pos[grid_x, grid_y, grid_z, sub_n] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_particle_num[grid_x, grid_y, grid_z, sub_n] = 0
    # state.grid_v_check[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_check2[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_grid_base(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0    
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    
    # state.grid_vk[grid_x, grid_y, grid_z, k] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_mk[grid_x, grid_y, grid_z, k] = 0.0    
    # state.grid_id[grid_x, grid_y, grid_z, k] = 0x7fffffff # SENTINEL = 0x7fffffff, -1
    # state.grid_count[grid_x, grid_y, grid_z] = 0
    
    # state.grid_v_in_prescribed[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_out_prescribed[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_mean_pos[grid_x, grid_y, grid_z, sub_n] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_particle_num[grid_x, grid_y, grid_z, sub_n] = 0
    # state.grid_v_check[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_check2[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]

# 5, p2g
@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return # pass
    
    stress = state.particle_stress[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    # if p == 100000:
    #     wp.print(grid_pos)
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    # grid-bounds guard: particles that left the simulation domain are skipped instead of
    # producing an out-of-bounds grid access (CUDA error 700)
    if (base_pos_x < 0 or base_pos_y < 0 or base_pos_z < 0 or
        base_pos_x > model.n_grid - 3 or base_pos_y > model.n_grid - 3 or base_pos_z > model.n_grid - 3):
        return
    fx = grid_pos - wp.vec3(                        # 입자의 실체 위치와 그리드 셀의 차이
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx                          # 보간 가중치 계산
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    # wp.printf("tid=%d  id=%d\n", p, state.particle_id[p])
    
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (
                    wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                ) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]                  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state.particle_C[p]                               
                # if model.rpic = 0, standard apic
                C = (1.0 - model.rpic_damping) * C + \
                    model.rpic_damping / 2.0 * ( C - wp.transpose(C) ) # RPIC damping
                if model.rpic_damping < -0.001:
                    # standard pic
                    C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                elastic_force = -state.particle_vol[p] * stress * dweight
                v_in_add = (
                    weight * state.particle_mass[p] * (state.particle_v[p] + C * dpos)
                    # weight * state.particle_mass[p] * (state.particle_v[p])
                    + dt * elastic_force
                    + dt * weight * state.particle_mass[p] * state.particle_gravity[p]
                )
                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                wp.atomic_add(state.grid_m,    ix, iy, iz, weight * state.particle_mass[p])
                # continue
                
                # 주의!!
                # vk, mk = 슬롯 수
                # 나중에 analytic 
                if state.particle_id[p] >= model.n_humans:
                # if state.particle_id[p] != -1:
                    continue
                
                vk_in_add = weight * state.particle_mass[p] * state.particle_vk[p]
                # vk_in_add = weight * state.particle_mass[p] * (state.particle_vk[p] + state.particle_vSM[p])
                wp.atomic_add(state.grid_vk, ix, iy, iz, state.particle_id[p], vk_in_add)
                wp.atomic_add(state.grid_mk, ix, iy, iz, state.particle_id[p], weight * state.particle_mass[p])
                
                '''
                if 0:
                    # 이게 3중 for문 안에 있어야 할까?
                    # K = state.grid_mk.shape[-1] # K : 같은 grid에 부여하는 Avatar의 속도장(vk)의 최대 개수
                    # grid_id[ix, iy, iz] 안에 내 id가 있다면, 해당 index를 slot으로 반환
                    slot = -1
                    K = model.n_slot
                    for k in range(K):
                        if state.grid_id[ix, iy, iz, k] == state.particle_id[p]:
                            slot = k
                            break
                    
                    # 만약 없다면, grid_id[ix, iy, iz]를 0번부터 탐색해서 빈곳에 내 id를 넣
                    SENTINEL = 0x7fffffff
                    if slot == -1:
                        for s in range(K):
                            if state.grid_id[ix,iy,iz,s] == SENTINEL:
                                prev = wp.atomic_min(state.grid_id, # prev = SENTINEL, 내 id, 다른 id
                                                    ix,iy,iz,s,
                                                    state.particle_id[p])
                                if prev == SENTINEL or prev == state.particle_id[p]: # 내가 차지했거나 이미 내 ID가 들어감
                                    slot = s
                                    # 첫 번째 성공 쓰레드만 카운터 증가
                                    if prev == SENTINEL:
                                        wp.atomic_add(state.grid_count, ix,iy,iz, 1)
                                    break
                                
                    vk_in_add = weight * state.particle_mass[p] * state.particle_vk[p]
                    wp.atomic_add(state.grid_vk, ix, iy, iz, slot, vk_in_add)
                    wp.atomic_add(state.grid_mk, ix, iy, iz, slot, weight * state.particle_mass[p])
                
                    # 아바타의 Fe가 가하는 응력, 굳이 필요는 없을 것 같지만...
                    # stress = state.particle_stress[p]
                    # elastic_force = -state.particle_vol[p] * stress * dweight
                    # wp.atomic_add(state.grid_f, ix, iy, iz, slot, dt * elastic_force) # mv   
                '''


# 5, p2g
@wp.kernel
def p2g_apic_with_stress_base(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return # pass
    
    stress = state.particle_stress[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    # if p == 100000:
    #     wp.print(grid_pos)
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    # grid-bounds guard: particles that left the simulation domain are skipped instead of
    # producing an out-of-bounds grid access (CUDA error 700)
    if (base_pos_x < 0 or base_pos_y < 0 or base_pos_z < 0 or
        base_pos_x > model.n_grid - 3 or base_pos_y > model.n_grid - 3 or base_pos_z > model.n_grid - 3):
        return
    fx = grid_pos - wp.vec3(                        # 입자의 실체 위치와 그리드 셀의 차이
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx                          # 보간 가중치 계산
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    # wp.printf("tid=%d  id=%d\n", p, state.particle_id[p])
    
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (
                    wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                ) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]                  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state.particle_C[p]                               
                # if model.rpic = 0, standard apic
                C = (1.0 - model.rpic_damping) * C + \
                    model.rpic_damping / 2.0 * ( C - wp.transpose(C) ) # RPIC damping
                if model.rpic_damping < -0.001:
                    # standard pic
                    C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                elastic_force = -state.particle_vol[p] * stress * dweight
                v_in_add = (
                    weight * state.particle_mass[p] * (state.particle_v[p] + C * dpos)
                    # weight * state.particle_mass[p] * (state.particle_v[p])
                    + dt * elastic_force
                    + dt * weight * state.particle_mass[p] * state.particle_gravity[p]
                )
                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                wp.atomic_add(state.grid_m,    ix, iy, iz, weight * state.particle_mass[p])
                # continue

   
# 6, add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z, k = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * ( # v = mv / m
            1.0 / state.grid_m[grid_x, grid_y, grid_z])
        v_out = v_out + dt * model.gravitational_accelaration # add gravity, v += dt * g
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out
    
    # grid_vk, grid_mk
    if state.grid_mk[grid_x, grid_y, grid_z, k] > 1e-15:
        vk_out = state.grid_vk[grid_x, grid_y, grid_z, k] * ( # v = mv / m
            # 1.0 / state.grid_m[grid_x, grid_y, grid_z])
            1.0 / state.grid_mk[grid_x, grid_y, grid_z, k])        
        state.grid_vk[grid_x, grid_y, grid_z, k] = vk_out
        
    # f_out = state.grid_f[grid_x, grid_y, grid_z] * ( # v = mv / m
    #     1.0 / state.grid_m[grid_x, grid_y, grid_z])
    # f_out = f_out + dt * model.gravitational_accelaration # add gravity, v += dt * g
    # state.grid_f[grid_x, grid_y, grid_z] = f_out

  
# 6, add gravity
@wp.kernel
def grid_normalization_and_gravity_base(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z, k = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * ( # v = mv / m
            1.0 / state.grid_m[grid_x, grid_y, grid_z])
        v_out = v_out + dt * model.gravitational_accelaration # add gravity, v += dt * g
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out

# 9, g2p
@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    # if state.particle_selection[p] == 0:
    if state.particle_selection[p] != 0:
        return # return
    
    grid_pos = state.particle_x[p] * model.inv_dx # 그리드 좌표 변환
    base_pos_x = wp.int(grid_pos[0] - 0.5)        # 중간점 찾기
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    # grid-bounds guard: particles that left the simulation domain are skipped instead of
    # producing an out-of-bounds grid access (CUDA error 700)
    if (base_pos_x < 0 or base_pos_y < 0 or base_pos_z < 0 or
        base_pos_x > model.n_grid - 3 or base_pos_y > model.n_grid - 3 or base_pos_z > model.n_grid - 3):
        return
    fx = grid_pos - wp.vec3(                      # 입자의 실체 위치와 그리드 셀의 차이
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx                        # 보간 가중치 계산
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(                                 # 3x3 행렬로, 각 방향(x,y,z)에 대해 보간 가중치 계산
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)) # tricubic interpolation에 사용되는 가중치의 미분값
    new_v  = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    new_vk = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    new_ve = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    new_C  = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 속도 그래디언트
    new_F  = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    new_Fk = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    new_Fe = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    mvk_sum = wp.vec3(0.0)   # Σ (m · v_k)
    mk_sum  = float(0.0)
    
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                
                grid_v = state.grid_v_out[ix, iy, iz]
                new_v  = new_v  + grid_v * weight
                new_C  = new_C  + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
                new_F  = new_F  + wp.outer(grid_v, dweight)
                # continue
                
                if state.particle_id[p] >= model.n_humans: # not human
                # if state.particle_id[p] != -1:
                    continue
                
                k = state.particle_id[p]
                grid_vk = state.grid_vk[ix, iy, iz, k]
                new_vk = new_vk + grid_vk * weight
                new_Fk = new_Fk + wp.outer(grid_vk, dweight)
                
                # grid_ve = grid_v - grid_vk
                # new_ve = new_ve + grid_ve * weight
                # new_Fe = new_Fe + wp.outer(grid_ve, dweight)
                
                if 0:
                    # avatar
                    # K = state.grid_f.shape[-1]
                    K = model.n_slot
                    mvk_sum = wp.vec3(0.0)   # Σ (m · v_k)
                    mk_sum  = float(0.0)
                    for s in range(K):
                        if state.grid_id[ix,iy,iz,s] == state.particle_id[p]:
                            mvk_sum += state.grid_vk[ix, iy, iz, s]
                            mk_sum  += state.grid_mk[ix, iy, iz, s]                        
                    if mk_sum > 0.0:
                        grid_vk = mvk_sum / mk_sum
                    else:
                        grid_vk = wp.vec3(0.0)
                    
                    new_Fk = new_Fk + wp.outer(grid_vk, dweight)
    
    state.particle_v[p]  = new_v
    state.particle_vk[p] = new_vk # state.particle_vko[p] 와의 오차 존재
    state.particle_x[p]  = state.particle_x[p] + dt * new_v
    state.particle_C[p]  = new_C
    
    # vk refine
    # 이게 어째서 문제지? C * dpos가 문제?
    # state.particle_v[p]  = new_v + state.particle_vko[p] - new_vk
    state.particle_vk[p] = state.particle_vko[p]
    
    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # F_tmp = (I33 + new_F * dt) * state.particle_F[p] # ori
    # state.particle_F_trial[p] = F_tmp # ori
    Fk_tmp = (I33 + new_Fk * dt) * state.particle_Fk[p] # Fk     
    Fe_tmp = (I33 + new_Fe * dt) * state.particle_Fe[p] # Fe
    if model.method == 0:
        F_tmp = (I33 + new_F  * dt) * wp.inverse(I33 + new_Fk * dt) * state.particle_F[p] # Fe = F * Fk^(-1)
        # if state.particle_id[p] >= model.n_humans:
        #     F_tmp = (I33 + new_F  * dt) * state.particle_F[p]
        # else:
        #     F_tmp = (I33 + new_F  * dt) * wp.inverse(I33 + new_Fk * dt) * state.particle_F[p] # Fe = F * Fk^(-1)
    elif model.method == 1:
        F_tmp = (I33 + new_F  * dt) * state.particle_F[p] # Fe = F * Fk^(-1)        
    # F_tmp = (I33 + new_F  * dt) * state.particle_F[p] # Fe = F * Fk^(-1)        
    state.particle_F_trial[p] = F_tmp # Fk는 return mapping이 필요없다, Fe_trial = F_trial 사용

    # for test
    state.particle_F_add[p] = (I33 + new_F  * dt) * wp.inverse(I33 + new_Fk * dt) # 이거 누적이 되나 ? 야매로 시간차로 감쇠해서 보여줄까 ?
    state.particle_Fe[p] = Fe_tmp # test
    state.particle_Fk[p] = Fk_tmp
    state.particle_F_before[p] = state.particle_F_trial[p]
    if model.update_cov_with_F:
        update_cov(state, p, new_F, dt)

# 9, g2p
@wp.kernel
def g2p_base(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    # if state.particle_selection[p] == 0:
    if state.particle_selection[p] != 0:
        return # return
    
    grid_pos = state.particle_x[p] * model.inv_dx # 그리드 좌표 변환
    base_pos_x = wp.int(grid_pos[0] - 0.5)        # 중간점 찾기
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    # grid-bounds guard: particles that left the simulation domain are skipped instead of
    # producing an out-of-bounds grid access (CUDA error 700)
    if (base_pos_x < 0 or base_pos_y < 0 or base_pos_z < 0 or
        base_pos_x > model.n_grid - 3 or base_pos_y > model.n_grid - 3 or base_pos_z > model.n_grid - 3):
        return
    fx = grid_pos - wp.vec3(                      # 입자의 실체 위치와 그리드 셀의 차이
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx                        # 보간 가중치 계산
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(                                 # 3x3 행렬로, 각 방향(x,y,z)에 대해 보간 가중치 계산
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)) # tricubic interpolation에 사용되는 가중치의 미분값
    new_v  = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    # new_vk = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    # new_ve = wp.vec3(0.0, 0.0, 0.0)                                                  # particle의 새로운 속도
    new_C  = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 속도 그래디언트
    new_F  = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    # new_Fk = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    # new_Fe = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                   # particle의 새로운 변형 그래디언트
    # mvk_sum = wp.vec3(0.0)   # Σ (m · v_k)
    # mk_sum  = float(0.0)
    
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                
                grid_v = state.grid_v_out[ix, iy, iz]
                new_v  = new_v  + grid_v * weight
                new_C  = new_C  + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
                new_F  = new_F  + wp.outer(grid_v, dweight)
                # continue
                
                if state.particle_id[p] >= model.n_humans:
                # if state.particle_id[p] != -1:
                    continue
    
    state.particle_v[p]  = new_v
    state.particle_x[p]  = state.particle_x[p] + dt * new_v
    state.particle_C[p]  = new_C
    
    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F_tmp = (I33 + new_F * dt) * state.particle_F[p]
    state.particle_F_trial[p] = F_tmp

    if model.update_cov_with_F:
        update_cov(state, p, new_F, dt)

# 10, Shape-Matching-damping
# 클러스터 내의 n개 파티클 속도를 덮어쓰기
# XPBD의 α = 0와 같은 현상을 ApplyDamping의 감쇠율 β=1 (완전 덮어쓰기)로 구현 (강체 속도장)
# XPBD에서 구현되는 shape-matching을 MPM에 구현하기 위해 약간의 수식 변경(위치 프로젝션)
# 1. rigid best-fit (shape matching)으로 모든 뼈 파티클의 질량중심, 공분산, 회전행렬을 구한다.
# 2. 위치 프로젝션 (강체화)
# 3. 강체 속도장으로 덮어쓰기 (감쇠율 β=1, 완전 덮어쓰기)
# 4. 강체 회전, 뼈 클러스터는 실제 변형이 없고 순수 회전만 가진다
# F_k = R (순수 뼈대에 따른 회전), F_e = I (탄성 변형 0), σ = 0 (응력 0), C = skew(ω) (APIC ∇v 슬롯으로 각운동량 보존)
# 어짜피 뼈대는 강체이므로 응력을 발생시키지 않는다 -> F_k = R, F_e = I

# shape matching을 위해서는 미리 뼈 particle들을 저장해 줘야한다.
# kabsch

# bone_mx:   wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
# bone_mv:   wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
# bone_m:    wp.array(dtype=float, ndim=2)     # [num of human, 20]
# bone_L:    wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
# bone_I:    wp.array(dtype=wp.mat33, ndim=2)  # [num of human, 20]
# bone_w:    wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
# bone_A:  wp.array(dtype=wp.mat33, ndim=2)  # [num of human, 20]
# bone_x0: wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 74496]
# bone_x0cm: wp.array(dtype=wp.vec3, ndim=2) # [num of human, 20]
# bone_idx:  wp.array(dtype=int)
# bone_pnum: wp.array(dtype=int)

@wp.kernel
def zero_bone(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    p, b = wp.tid()
    state.bone_mx [p, b] = wp.vec3(0.0, 0.0, 0.0)
    state.bone_mv [p, b] = wp.vec3(0.0, 0.0, 0.0)
    state.bone_m  [p, b] = 0.0
    # state.bone_L  [p, b] = wp.vec3(0.0, 0.0, 0.0)
    # state.bone_I  [p, b] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.bone_A  [p, b] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # state.bone_R  [p, b] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # state.bone_w  [p, b] = wp.vec3(0.0, 0.0, 0.0)

# reduce
@wp.kernel
def shape_matching1_reduce(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    p = wp.tid()
    bid = state.bone_idx[p] # bone id
    if bid < 0: # bid == -1
        return
    pid = state.particle_id[p] # avatar id
    
    # m = state.particle_mass[p] # not mass
    # wp.atomic_add(state.bone_mx, pid, bid, m * state.particle_x[p])
    # wp.atomic_add(state.bone_mv, pid, bid, m * state.particle_v[p])
    # wp.atomic_add(state.bone_m , pid, bid, m)
    wp.atomic_add(state.bone_mx, pid, bid, state.particle_x[p])
    # wp.atomic_add(state.bone_mv, pid, bid, state.particle_v[p])  # for damping
    wp.atomic_add(state.bone_m , pid, bid, 1.0)

@wp.kernel
def shape_matching2_solve(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    p = wp.tid()
    bid = state.bone_idx[p] # bone id
    if bid < 0: # bid == -1
        return    
    pid = state.particle_id[p] # avatar id    
    M  = state.bone_m[pid, bid] # bone mass
    invM = 1.0 / M
    
    # if p == 0:
    #     wp.printf("pid=%d, bid=%d \n", pid, bid)
    #     wp.printf("x_cm=%f, %f, %f\n", x_cm[0], x_cm[1], x_cm[2])
    
    x_cm = state.bone_mx[pid, bid] * invM  # now bone center
    # v_cm = state.bone_mv[pid, bid] * invM  # used in damping3, not used here    
    x  = state.particle_x[p]
    # v  = state.particle_v[p]  # for damping
    # m  = state.particle_mass[p] # not use mass
    r  = x - x_cm # p_i
    
    # q  = state.bone_q[pid, p - state.avatar_offset[pid]]     # cano bone center
    q  = state.bone_x0[pid, p - state.avatar_offset[pid]] - state.bone_x0cm[pid, bid] # cano bone center
    
    if p - state.avatar_offset[pid] < 0 or p - state.avatar_offset[pid] >= state.bone_x0.shape[1]:
        wp.printf("Out of bone index 1\n")
    
    # I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # A = m * wp.outer(r, q)    
    # L = m * wp.cross(r, v)
    # I = m * (wp.dot(r, r) * I33 - wp.outer(r, r))
    A = wp.outer(r, q)    
    # L = wp.cross(r, v)  # for damping
    # I = (wp.dot(r, r) * I33 - wp.outer(r, r))  # for damping
    
    wp.atomic_add(state.bone_A, pid, bid, A)
    # wp.atomic_add(state.bone_L, pid, bid, L)  # for damping
    # wp.atomic_add(state.bone_I, pid, bid, I)  # for damping

# shape_matching_apply_b2p
@wp.kernel
def shape_matching3_Rw(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    pid, bid = wp.tid()
    
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    S = wp.vec3(0.0)
    A = state.bone_A[pid, bid]
    # L = state.bone_L[pid, bid]  # for damping
    # I = state.bone_I[pid, bid]  # for damping
    wp.svd3(A, U, S, V)
    
    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]
    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    # w = wp.inverse(I) * L  # for damping
    
    state.bone_R[pid, bid] = R
    # state.bone_w[pid, bid] = w # for damping
    
    #######
    # M  = state.bone_m[pid, bid] # bone mass
    # invM = 1.0 / M
    # x_cm = state.bone_mx[pid, bid] * invM  # now bone center
    
@wp.kernel
def shape_matching4_bone_particle(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    p = wp.tid()
    bid = state.bone_idx[p] # bone id
    if bid < 0: # bid == -1 : avatar
        return    
    pid = state.particle_id[p] # avatar id
    
    R = state.bone_R[pid, bid] # rotation matrix
    # w = state.bone_w[pid, bid] # angular velocity # for damping
    invM = 1.0 / state.bone_m[pid, bid]
    x_cm = state.bone_mx[pid, bid] * invM
    # v_cm = state.bone_mv[pid, bid] * invM # for damping
    
    # if p == 0:
    #     wp.printf("pid=%d, bid=%d \n", pid, bid)
    #     wp.printf("x_cm=%.6f, %.6f, %.6f\n", x_cm[0], x_cm[1], x_cm[2])
    #     wp.printf("v_cm=%.6f, %.6f, %.6f\n", v_cm[0], v_cm[1], v_cm[2])

    p_local = p - state.avatar_offset[pid]
    if p_local < 0 or p_local >= state.bone_x0.shape[1]:
        wp.printf("Out of bone index 2\n")
    # p_i  = state.particle_x[p] - x_cm # for damping
    q_i  = state.bone_x0[pid, p - state.avatar_offset[pid]] - state.bone_x0cm[pid, bid]
    x_new = R * q_i + x_cm
    
    # # damping    
    # v_damp = v_cm + wp.cross(w, p_i) # for damping
    # Fk_new = R # wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) # for damping
    # Fe_new = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) # for damping
    # C_new = wp.skew(w) # for damping
    
    # Meshless Deformations Based on Shape Matching
    # 3.4 Integration의 (9) 수식
    # v, F, C에 Damping 적용
    # alpha1 = 0.00023
    # alpha2 = 0.00023
    # alpha1 = 0.00092
    # alpha2 = 0.00092
    # alpha1 = 0.02
    # alpha2 = 0.02
    alpha1 = 0.01 # 0.03 error
    # alpha2 = 0.005
    x_now = state.particle_x[p]
    # v_now = state.particle_v[p]
    # vk_now = state.particle_vk[p]
    # Fk_now = state.particle_Fk[p]
    # Fe_now = state.particle_F_trial[p] # Fe
    # C = state.particle_C[p]
    v_sm = (x_new - x_now) / dt # shape matching position projection
    
    # vk에 줘야하나?
    # state.particle_v[p] += alpha1 * (v_sm - v_now)   # shape matching 유도
    # state.particle_v[p] += alpha2 * (v_damp - v_now) # rigid damping 유도
    # state.particle_vk[p] += alpha1 * (v_sm - vk_now)   # shape matching 유도
    # state.particle_vk[p] += alpha2 * (v_damp - vk_now) # rigid damping 유도
    
    #######################################################################
    # shape matching + rigid damping 유도
    state.particle_vk[p]  += alpha1 * v_sm - state.particle_vSM[p]
    # state.particle_vk[p]  += alpha1 * v_sm + alpha1 * v_damp - state.particle_vSM[p]
    state.particle_vko[p]  += alpha1 * v_sm - state.particle_vSM[p]
        # state.particle_vko[p]  += alpha1 * v_sm + alpha1 * v_damp - state.particle_vSM[p]
    state.particle_vSM[p] = alpha1 * v_sm
    
    # state.particle_vko[p]에 값 넣을때 state.particle_vSM[p] 고려해야한다
    ##########################################################################
    # sym  = 0.5*(C + C.T)
    # skew = 0.5*(C - C.T)
    # gamma = 0.3   # 0.2~0.5 시도
    # sym = (1.0 - gamma) * sym
    # C = skew + sym

    # # (B) Ω_kin 정합(추천)
    # # R에서 각속도 근사: Ω_kin ≈ log(R)/dt (소각도면 ω×r 방식도 OK)
    # C = (1.0 - gamma) * C + gamma * Ω_kin
    
    # state.particle_C[p] = ( C_now - wp.transpose(C_now) ) / 2.0    
    # state.particle_C[p] += alpha1 * (C_new - C_now) # rigid damping 유도
    ##########################################################################
    # state.particle_vSM[p] = alpha1 * v_sm + alpha2 * v_damp
    # state.particle_vk[p]  += alpha1 * v_sm # 어짜피 g2p에서 vko로 초기화 됨, 근데 이러면 p2g에서 vk에 더하는거랑 무슨 차이일까
    
    # state.particle_Fk[p] += alpha1 * (Fk_new - Fk_now)   # shape matching 유도
    # state.particle_F_trial[p] += alpha1 * (Fe_new - Fe_now) # rigid damping 유도, 이게 필요할까?
    # state.particle_C[p] += alpha1 * (C_new - C_now) # rigid damping 유도
    
    # state.particle_x[p] = x_new
    # state.particle_v[p] = v_new
    # state.particle_F_trial[p] = Fe_new
    # state.particle_Fk[p] = Fk_new
    # state.particle_C[p] = C_new
    
    ########### only for test #########################
    # state.particle_SM_test[p] = x_new

   
@wp.kernel
def shape_matching5_avatar_particle(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    p = wp.tid()
    bid = state.bone_idx[p] # bone id
    if bid >= 0: # bid == -1
        return
    
    alpha = 0.01
    # state.particle_x[p] += alpha * state.particle_vSM[p]
    # state.particle_vk[p]  += alpha * v_sm - state.particle_vSM[p] # particle_v인가? 확인 필요
    # state.particle_vSM[p] = alpha * v_sm # 둘중 하나만


@wp.kernel
def shape_matching_damping(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    pid, bid = wp.tid()
    
    # 1, compute center of mass and velocity of the bone particles
    num = wp.float32(0)
    x_c = wp.vec3(0.0, 0.0, 0.0)
    v_c = wp.vec3(0.0, 0.0, 0.0)
    ps = state.avatar_offset[pid]  # particle start index for this avatar
    srt, end = model.bone_index[bid], model.bone_index[bid + 1]
    
    for i in range(ps + srt, ps + end):
        x_c = x_c + state.particle_x[i]
        v_c = v_c + state.particle_v[i]
        num = num + 1.0
    x_c = x_c / num
    v_c = v_c / num
    
    # 2, compute Apq, L, I
    Apq = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    L = wp.vec3(0.0, 0.0, 0.0)
    I = wp.mat33(0.0)
    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for i in range(ps + srt, ps + end):
        p = state.particle_x[i] - x_c
        q = state.bone_q[pid, i-srt]
        Apq = Apq + wp.outer(p, q)
        L   = L + wp.cross(p, state.particle_v[i])
        I   = I + (wp.dot(p, p) * I33 - wp.outer(p, p))

    # state.bone_A[pid, bid] = Apq
    # state.bone_L[pid, bid] = L
    # state.bone_I[pid, bid] = I
    
    # 3, compute R, w
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    S = wp.vec3(0.0, 0.0, 0.0)
    wp.svd3(Apq, U, S, V)
    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]
    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    w = wp.inverse(I) * L
    if pid == 0 and bid == 0:
        wp.printf("R=(%.6f, %.6f, %.6f)\n", R[0, 0], R[0, 1], R[0, 2])
        wp.printf("R=(%.6f, %.6f, %.6f)\n", R[1, 0], R[1, 1], R[1, 2])
        wp.printf("R=(%.6f, %.6f, %.6f)\n", R[2, 0], R[2, 1], R[2, 2])

    state.bone_R[pid, bid] = R
    state.bone_w[pid, bid] = w

    # wp.printf("pid=%d bid=%d R=(%.6f, %.6f, %.6f)\n", pid, bid, num[0], num[1], num[2])

# 4, compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if model.material == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        elif model.material == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        if model.material == 0 or model.material == 5:
            # stress = kirchhoff_stress_corotated_hybrid(
            #     state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            # )            
            # stress = kirchhoff_stress_corotated(
            #     state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            # )
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p], p
            )    
        if model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress

@wp.kernel
def compute_quat_scale_from_F(state: MPMStateStruct, model: MPMModelStruct):
    # R가 틀릴 수 있음, 자세한 차이는 gpt에 물어보기
    p = wp.tid()

    # 1) F 읽기
    F = state.particle_F_trial[p]

    # 2) init_cov (packed-6) -> mat33
    base = p * 6
    init_cov = mat33_from_packed6(state.particle_init_cov, base)

    # 3) 새 공분산 Σ = F Σ0 F^T
    cov = F * init_cov * wp.transpose(F)

    # (선택) cov 저장을 원하면 packed-6로 써두기
    packed6_from_mat33(cov, state.particle_cov, base)

    # 4) SVD로 고유분해 (대칭 ⇒ U≈V)
    U = wp.mat33f(0.0)
    V = wp.mat33f(0.0)
    s = wp.vec3f(0.0)
    wp.svd3(cov, U, s, V)          # cov = U diag(s) V^T, s >= 0 (이상적)

    # 5) 회전 R = U (det 보정으로 proper rotation)
    #    대칭 SPD에서는 U와 V가 거의 동일하지만, 수치적으로 det(U) < 0일 수 있으므로 보정
    if wp.determinant(U) < 0.0:
        U[0,2] = -U[0,2]
        U[1,2] = -U[1,2]
        U[2,2] = -U[2,2]

    R = U

    # 6) 스케일 = sqrt(eigenvalues) = sqrt(s)
    eps = 1e-20
    sx = wp.sqrt(wp.max(s[0], eps))
    sy = wp.sqrt(wp.max(s[1], eps))
    sz = wp.sqrt(wp.max(s[2], eps))

    # (선택) 내림차순 정렬로 축 일관화 하고 싶다면 정렬/재정렬 로직 추가 가능

    # 7) 쿼터니언
    q = quat_from_mat33(R)

    # 8) 저장
    state.particle_quat[p]  = q                    # (w,x,y,z)
    state.particle_scale[p] = wp.vec3f(sx, sy, sz) # principal std (σx,σy,σz)
    state.particle_R[p]  = R

@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)

@wp.func
def overwrite_R_to_F(F_trial:wp.mat33, R_new: wp.mat33):
    # p = wp.tid()
    # F = state.particle_F_trial[p]
    
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]
        
    # stretch S = V diag(sig) V^T
    Sig = wp.mat33(sig[0],0.0,0.0, 0.0,sig[1],0.0, 0.0,0.0,sig[2])
    S = V * Sig * wp.transpose(V)

    # optional: if you also want to ensure R_new is proper rotation (det=+1),
    # you should det-fix R_new too (not shown).
    return R_new * S

# 7
@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point # hun : 여기에 multiple points가능한가?

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0

@wp.func
def mat33_from_packed6(p6: wp.array(dtype=wp.float32), base: int) -> wp.mat33f:
    # p6[base + 0..5] = [xx, xy, xz, yy, yz, zz] (대칭)
    M = wp.mat33f(0.0)
    xx = p6[base + 0]
    xy = p6[base + 1]
    xz = p6[base + 2]
    yy = p6[base + 3]
    yz = p6[base + 4]
    zz = p6[base + 5]
    M[0,0] = xx; M[0,1] = xy; M[0,2] = xz
    M[1,0] = xy; M[1,1] = yy; M[1,2] = yz
    M[2,0] = xz; M[2,1] = yz; M[2,2] = zz
    return M

@wp.func
def packed6_from_mat33(M: wp.mat33f, p6: wp.array(dtype=wp.float32), base: int):
    p6[base + 0] = M[0,0]
    p6[base + 1] = M[0,1]
    p6[base + 2] = M[0,2]
    p6[base + 3] = M[1,1]
    p6[base + 4] = M[1,2]
    p6[base + 5] = M[2,2]

@wp.func
def quat_from_mat33(R: wp.mat33f) -> wp.vec4f:
    # 반환 (w,x,y,z), trace 분기 + 정규화 + 부호 고정(w>=0)
    t = R[0,0] + R[1,1] + R[2,2]
    qw = wp.float32(0.0); qx = wp.float32(0.0); qy = wp.float32(0.0); qz = wp.float32(0.0)
    if t > 0.0:
        s = wp.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = wp.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = wp.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = wp.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    mag2 = qw*qw + qx*qx + qy*qy + qz*qz
    inv = 1.0 / wp.sqrt(wp.max(mag2, 1e-20))
    qw *= inv; qx *= inv; qy *= inv; qz *= inv
    if qw < 0.0:
        qw = -qw; qx = -qx; qy = -qy; qz = -qz
    return wp.vec4f(qw, qx, qy, qz)


# 67, grid penalty (Multi-Material Contact)
# @wp.kernel
def grid_penalty(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()  
    # [n_grid, n_grid, n_grid, 0, 3]  : not human
    # [n_grid, n_grid, n_grid, 1:, 3] : human
    # grid_v_mean_pos: wp.array(dtype=wp.vec3, ndim=4), [200, 200, 200, 3, 3]
    # grid_v_particle_num: wp.array(dtype=int, ndim=4), [200, 200, 200, 3]
    
    if (grid_x == 0 or grid_x == model.n_grid-1 or 
        grid_y == 0 or grid_y == model.n_grid-1 or 
        grid_z == 0 or grid_z == model.n_grid-1):
        return
    
    # self.mpm_model.n_subjects = 3
    # self.mpm_model.n_humans = 2
    
    # grid_pos * model.dx = particle_x
    penalty_d = model.penalty_d # 5000.0
    penalty_v = model.penalty_v # 200.0
    penalty_th = model.penalty_th # 1.5
    threshold = penalty_th * model.dx
    # threshold = 2.0 * model.dx
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
    # for i in range(-2, 3):
    #     for j in range(-2, 3):
    #         for k in range(-2, 3):
                nx = grid_x + i
                ny = grid_y + j
                nz = grid_z + k
                
                if (nx < 0 or nx >= model.n_grid or
                    ny < 0 or ny >= model.n_grid or
                    nz < 0 or nz >= model.n_grid):
                    continue
                
                for sub1 in range(model.n_humans+1): # 0, 1, 2
                    # 이 노드(grid_x,y,z, sub1)의 파티클 개수
                    num1 = state.grid_v_particle_num[grid_x, grid_y, grid_z, sub1]
                    if num1 <= 0:
                        continue
                    
                    for sub2 in range(model.n_humans+1):
                        
                        if sub1 == sub2: # 자신 물체와는 반발력 발생 안한다
                            continue                       
                        
                        num2 = state.grid_v_particle_num[nx, ny, nz, sub2]
                        if num2 <= 0:
                            continue   
                        
                        sum_pos_1 = state.grid_v_mean_pos[grid_x, grid_y, grid_z, sub1]
                        sum_pos_2 = state.grid_v_mean_pos[nx, ny, nz, sub2]
                        
                        mean1 = sum_pos_1 * ( 1.0 / wp.float(num1) )
                        mean2 = sum_pos_2 * ( 1.0 / wp.float(num2) )

                        r = mean2 - mean1
                        dist = wp.length(r)
                                                
                        # threshold = 0.1 * model.dx    
                        overlap = threshold - dist
                        if overlap <= 0.0 :
                            overlap = 0.0
                            # continue # continue 말고 0으로?
                        
                        # i, j, k
                        n = r / (dist + 1.0e-6)  # 법선
                        
                        # penalty_d = 5000.0
                        # f_overlap = wp.float(num1) * wp.float(num2) * penalty_k * overlap * n
                        # f_overlap = k_penalty * (mass_1 * mass_2) / (mass_1 + mass_2) * overlap * n
                        f_overlap = penalty_d * overlap * n
                        
                        # velocity damping
                        # v1 = state.grid_v_out[grid_x, grid_y, grid_z]
                        # v2 = state.grid_v_out[nx, ny, nz]
                        v1 = state.grid_v_out_prescribed[grid_x, grid_y, grid_z]
                        v2 = state.grid_v_out_prescribed[nx, ny, nz]
                        m1 = state.grid_m[nx, ny, nz]
                        m2 = state.grid_m[nx, ny, nz]
                        
                        relative_v = v2 - v1 # 이게 나름 잘되는 예시
                        # relative_v = v1 - v2
                        # relative_v[0] = wp.min(relative_v[0], 0.0)
                        # relative_v[1] = wp.min(relative_v[1], 0.0)
                        # relative_v[2] = wp.min(relative_v[2], 0.0)
                        # if relative_v < 0:                        
                        direction = wp.vec3(float(-i), float(-j), float(-k))                        
                        projection_magnitude = wp.dot(relative_v, direction)
                        if projection_magnitude < 0:
                            relative_v = wp.vec3(0.0, 0.0, 0.0)
                        else:
                            relative_v = projection_magnitude * direction
                            
                        f_rel_vol = penalty_v * relative_v 
                        # cosine simility
                        
                        ############################################################
                        # 1. 침투량 기반 penalty                     
                        f = f_overlap + f_rel_vol
                        # f = f_overlap
                        
                        # 원래 코드
                        dv1 = (f / (m1 + 1.0e-9)) * dt
                        dv2 = -(f / (m2 + 1.0e-9)) * dt
                        
                        # 방향 수정, 끌려온다
                        # dv1 = -(f / (m1 + 1.0e-9)) * dt
                        # dv2 = (f / (m2 + 1.0e-9)) * dt
                        # wp.print("dv1:"); wp.print(dv1)
                        # wp.print("m1:"); wp.print(m1)
                        
                        ############################################################
                        # # 2. 임펄스 기반 반발 계수 e
                        # v_rel_norm = wp.dot(relative_v, n)
                        # if v_rel_norm >= 0.0:
                        #     continue
                        
                        # mass1 = wp.float(num1)
                        # mass2 = wp.float(num2)
                        # m_eff = mass1 * mass2 / (mass1 + mass2 + 1.0e-6)

                        # # 임펄스 크기: J = -(1+ e) * v_rel_norm * m_eff
                        # # (음수 => approach)
                        # e = 0.8 # model.rest_coeff, 0~1
                        # J = -(1.0 + e) * v_rel_norm * m_eff

                        # # 임펄스 벡터
                        # # normal 방향
                        # impulse = J * n

                        # # => v1_new = v1 + impulse / m1
                        # # => v2_new = v2 - impulse / m2
                        # # 여기선 한 번에 atomic_add 형태로 반영
                        # df1 = (impulse / (mass1 + 1.0e-6)) * dt   # dt 곱해서 'velocity change'
                        # df2 = (-impulse / (mass2 + 1.0e-6)) * dt
                        ############################################################
                                                    
                        # atomic_add
                        wp.atomic_add(state.grid_v_out, grid_x, grid_y, grid_z, dv1)
                        wp.atomic_add(state.grid_v_out,     nx,     ny,     nz, dv2)
                        
                        wp.atomic_add(state.grid_v_check, grid_x, grid_y, grid_z, dv1)
                        wp.atomic_add(state.grid_v_check,     nx,     ny,     nz, dv2)
                                                
                        # if sub1 == 0 and sub2 == 1:
                        #     wp.print("f_overlap:"); wp.print(f_overlap)
                        #     wp.print("f_damp:"); wp.print(f_damp)
