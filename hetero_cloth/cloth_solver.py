"""hetero_cloth.cloth_solver: Form A subclass that extends user's
MPM_Simulator_WARP with MPMAvatar-style cloth, without modifying user's code.

Design (decided in conversation):
  - Subclass MPM_Simulator_WARP
  - Override p2g2p; if cloth has not been registered, defer to super().p2g2p
    so the user's existing simulations behave bit-identically.
  - When cloth is registered, p2g2p body is decomposed into 6 phase helpers
    (A: pre-stress, B: stress, C: P2G, D: grid update, E: G2P, F: post-G2P)
    that each mirror the corresponding chunk of user's p2g2p verbatim.
  - 3 cloth kernel groups are inserted at the boundaries:
      after Phase B  -> compute_cloth_stress_kernel  (also clears vertex_force)
      after Phase C  -> p2g_vertex_force_kernel
      after Phase E  -> g2p_cloth_elements_kernel

The phase methods reproduce user's launch sequence but never call private
attributes that user's code itself doesn't expose. Anything user's code
references on self (instance attributes, methods like self.kabsch, etc.)
is reached through the inherited self.

WARNING: this wrapper mirrors user's p2g2p line-by-line.  If
mpm_solver_warp_separable_contact.py:p2g2p is edited (kernel order,
new launches, etc.), the corresponding _phase_* method here must be
updated to match.  This is the unavoidable cost of "no-modification"
integration.
"""
import time

import warp as wp
import warp.torch
import torch
import numpy as np

from mpm_solver_warp.mpm_solver_warp_separable_contact import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils_separable_contact import (
    zero_grid,
    zero_grid_separable,
    compute_stress_from_F_trial,
    p2g_apic_with_stress,
    grid_normalization_and_gravity,
    grid_normalization_and_gravity_separable,
    add_damping_via_grid,
    compute_vcm_and_resolve,
    apply_bounding_box_separable,
    g2p,
    zero_bone,
    shape_matching1_reduce,
    shape_matching2_solve,
    shape_matching3_Rw,
    shape_matching4_bone_particle,
)
from mpm_solver_warp.mpm_human_utils_separable_contact import (
    compute_animatable_gaussians_velocity_rel,
    compute_animatable_gaussians_velocity_tgt,
    compute_animatable_gaussians_velocity_gt,
    compute_smplx_velocity_rel,
    compute_smplx_velocity_tgt,
    compute_smplx_velocity_gt,
)

from .cloth_state import ClothStateStruct
from .cloth_init import ClothPrep, populate_cloth_state
from .cloth_kernels import (
    zero_vertex_force_kernel,
    zero_cloth_vertex_stress_kernel,
    compute_cloth_stress_kernel,
    p2g_vertex_force_kernel,
    g2p_cloth_elements_kernel,
    apply_cloth_pin_velocity_kernel,
)


class MPM_Simulator_WARP_with_Cloth(MPM_Simulator_WARP):
    """Drop-in replacement for MPM_Simulator_WARP that adds cloth integration.

    Usage:
        solver = MPM_Simulator_WARP_with_Cloth(n_particles=N_total, ...)  # N_total includes cloth
        # ... user's normal load_initial_data_from_torch / set_parameters etc.
        # cloth particles are appended after user's existing particles.
        prep = prepare_cloth_template(verts, faces, n_existing=N_user, ...)
        solver.register_cloth(prep)
        # ... then call solver.p2g2p(...) as usual.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cloth_state: ClothStateStruct | None = None
        self.has_cloth: bool = False
        # LBS-pin context for top cloth verts (set externally by caller after register_cloth)
        # dict with keys: pose_dataset, cloth_v_can_np, cloth_lbs_w_np, pinned_local_np,
        #                 body_rot_mats, body_scale
        self.cloth_pin_context: dict | None = None
        # Body-mesh collider for cloth: opt-in extra constraint that projects
        # cloth's resolved grid velocity off the kinematic SMPL-X body mesh,
        # preventing visual body-through-cloth penetration that separable
        # contact alone cannot fix. Set via register_body_mesh_collider().
        self.body_mesh_collider = None  # BodyMeshColliderState | None

    # ------------------------------------------------------------------
    # cloth registration
    # ------------------------------------------------------------------
    def register_cloth(self, prep: ClothPrep, device: str = "cuda:0"):
        """Allocate ClothStateStruct from a ClothPrep and mark cloth as active.

        The user is responsible for having already initialised mpm_state with
        n_existing + n_elements + n_vertices total particles, and for having
        loaded the cloth particle positions/velocities/volumes/masses into the
        last n_elements + n_vertices slots (see cloth_init.assemble_cloth_particle_arrays).
        """
        cs = ClothStateStruct()
        cs.init(prep.n_existing, prep.n_elements, prep.n_vertices, device=device)
        populate_cloth_state(cs, prep, device=device)
        self.cloth_state = cs
        self.has_cloth = True

    def register_body_mesh_collider(self, verts, faces, cloth_subj_id,
                                    friction=0.1, device: str = "cuda:0"):
        """Attach a kinematic SMPL-X body mesh as an extra collider for the
        cloth subject. After this is set, every substep's Phase D will project
        the cloth subject's resolved grid velocity off this mesh, preventing
        body-through-cloth leak that the separable-contact alone cannot avoid.

        Caller is responsible for refreshing the mesh's pose every frame via
        self.body_mesh_collider.update_pose(verts_now, verts_next, frame_dt).
        """
        from .body_mesh_collider import BodyMeshColliderState
        self.body_mesh_collider = BodyMeshColliderState(
            verts=verts,
            faces=faces,
            n_grid=self.mpm_model.n_grid,
            cloth_subj_id=cloth_subj_id,
            friction=friction,
            device=device,
        )

    # ------------------------------------------------------------------
    # p2g2p override
    # ------------------------------------------------------------------
    def p2g2p(self, frame, step, dt, mpm_params, device="cuda:0", smplx_dt=4e-2,
              is_3d_measure=False, f=None, sim_params=None):
        # Guard: when cloth is not active, behave bit-identically to user's solver.
        if not self.has_cloth:
            return super().p2g2p(
                frame, step, dt, mpm_params, device=device, smplx_dt=smplx_dt,
                is_3d_measure=is_3d_measure, f=f, sim_params=sim_params,
            )

        # ============================================================
        # Cloth-active path: mirror of user's p2g2p body, with 3 cloth
        # kernel groups interleaved.
        # ============================================================
        time_total = time.time()
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        grid_size_nsub = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            self.mpm_model.n_humans,
        )
        grid_size_subj = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            self.mpm_model.n_subjects,
        )
        grid_size_3d = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )

        # ---- LBS-pin: top cloth verts kinematically follow body via per-substep velocity override ----
        if self.has_cloth and self.cloth_state is not None and self.cloth_pin_context is not None:
            if step == 0:
                from .cloth_integration import update_cloth_pin_target_v
                ctx = self.cloth_pin_context
                update_cloth_pin_target_v(
                    self.cloth_state,
                    pose_dataset=ctx["pose_dataset"],
                    human_step=int(round(self.time / smplx_dt)),
                    cloth_v_can_np=ctx["cloth_v_can_np"],
                    cloth_lbs_w_np=ctx["cloth_lbs_w_np"],
                    pinned_local=ctx["pinned_local_np"],
                    body_rot_mats=ctx["body_rot_mats"],
                    body_scale=ctx["body_scale"],
                    smplx_dt=smplx_dt,
                    device=device,
                )
            wp.launch(
                kernel=apply_cloth_pin_velocity_kernel,
                dim=self.cloth_state.n_vertices,
                inputs=[self.mpm_state, self.cloth_state],
                device=device,
            )

        # ---- Phase A : pre-stress (zero grid + human velocity + impulse + dirichlet) ----
        self._phase_A_pre_stress(
            frame, step, dt, device, smplx_dt, is_3d_measure, f, sim_params,
            grid_size_nsub, grid_size_subj,
        )

        # ---- Phase B : compute stress (traditional particles) ----
        self._phase_B_stress(dt, device)

        # === CLOTH INSERT #1 :
        #   1) zero vertex_force buffer (per substep)
        #   2) defensively zero cloth vertex stress (user's compute_stress_from_F_trial
        #      may write nonzero stress into cloth vertex slots once F drifts)
        #   3) compute anisotropic cloth element stress + FEM nodal forces
        wp.launch(
            kernel=zero_vertex_force_kernel,
            dim=self.cloth_state.n_vertices,
            inputs=[self.cloth_state],
            device=device,
        )
        wp.launch(
            kernel=zero_cloth_vertex_stress_kernel,
            dim=self.cloth_state.n_vertices,
            inputs=[self.mpm_state, self.cloth_state],
            device=device,
        )
        wp.launch(
            kernel=compute_cloth_stress_kernel,
            dim=self.cloth_state.n_elements,
            inputs=[self.mpm_state, self.cloth_state, self.mpm_model, dt],
            device=device,
        )

        # ---- Phase C : P2G (mass + APIC + stress-divergence for everyone) ----
        self._phase_C_p2g(dt, device)

        # === CLOTH INSERT #2 : add FEM vertex_force into grid ===
        wp.launch(
            kernel=p2g_vertex_force_kernel,
            dim=self.cloth_state.n_vertices,
            inputs=[self.mpm_state, self.cloth_state, self.mpm_model, dt],
            device=device,
        )

        # ---- Phase D : grid update + collider + separable resolve ----
        self._phase_D_grid_update(
            dt, device, grid_size, grid_size_nsub, grid_size_subj, grid_size_3d,
        )

        # ---- Phase E : G2P ----
        self._phase_E_g2p(dt, device)

        # === CLOTH INSERT #3 : cloth element kinematics from vertex average + d update ===
        wp.launch(
            kernel=g2p_cloth_elements_kernel,
            dim=self.cloth_state.n_elements,
            inputs=[self.mpm_state, self.cloth_state, self.mpm_model, dt],
            device=device,
        )

        # ---- Phase F : post-G2P (bone shape matching, time advance) ----
        self._phase_F_post_g2p(frame, step, dt, device)

        time_total = time.time() - time_total
        # NOTE: user's p2g2p has a print_time block at the end that aggregates
        # per-phase timings. We omit it here to keep the wrapper focused; the
        # cost is just losing the per-substep timing print.

    # ==================================================================
    # Phase A : pre-stress
    # Lines 1054-1159 of user's p2g2p, ported verbatim.
    # ==================================================================
    def _phase_A_pre_stress(self, frame, step, dt, device, smplx_dt, is_3d_measure,
                            f, sim_params, grid_size_nsub, grid_size_subj):
        # 1, zero grid
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size_nsub),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        if self.mpm_model.use_separable_contact == 1:
            wp.launch(
                kernel=zero_grid_separable,
                dim=grid_size_subj,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        # Human Velocity (only at substep 0 of each frame in user's code)
        human_step = int(round(self.time / smplx_dt))
        if step == 0:
            self.human_step = human_step
            if human_step == 0:
                lbs_temp = torch.zeros(
                    [self.mpm_state.particle_x.shape[0], 55], device=device
                )
            for k in range(len(self.human_modify_changer)):
                with wp.ScopedTimer("to_torch_particle_x", synchronize=True, print=False, dict=self.time_profile):
                    particle_x = warp.torch.to_torch(self.mpm_state.particle_x)
                if self.human_modify_model[k].model_type == 'animatable_gaussians':
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM, rot_mats_next_total = compute_animatable_gaussians_velocity_rel(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f
                        )
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM, rot_mats_next_total = compute_animatable_gaussians_velocity_tgt(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f
                        )
                    elif self.human_modify_model[k].velocity_type == 'gt':
                        velocity, rel_rot_mats, lbs, vSM = compute_animatable_gaussians_velocity_gt(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f
                        )
                elif self.human_modify_model[k].model_type == 'smplx':
                    rot_mats_next_total = None
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_rel(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params
                        )
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_tgt(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params
                        )
                    elif self.human_modify_model[k].velocity_type == 'gt':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_gt(
                            self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params
                        )
                else:
                    assert False, "No velocity function"
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                with wp.ScopedTimer("from_torch", synchronize=True, print=False, dict=self.time_profile):
                    new_human_velocity = wp.array(velocity.detach().cpu().numpy(), dtype=wp.vec3)
                    rel_rot_mats = wp.array(rel_rot_mats.detach().cpu().numpy(), dtype=wp.mat33)
                    vSM = wp.array(vSM.detach().cpu().numpy(), dtype=wp.vec3)
                    if rot_mats_next_total is not None:
                        rot_mats_next_total = wp.array(rot_mats_next_total.detach().cpu().numpy(), dtype=wp.mat33)
                wp.synchronize()
                wp.launch(
                    kernel=self.human_modify_changer[k],
                    dim=self.human_modify_model[k].human_n_particles,
                    inputs=[self.mpm_state, self.human_modify_params[k], new_human_velocity, rel_rot_mats, 0, vSM, frame],
                    device=device,
                )
                wp.synchronize()
            if human_step == 0:
                with wp.ScopedTimer("LBS", synchronize=True, print=False, dict=self.time_profile):
                    self.mpm_state.particle_LBS = wp.array(lbs_temp.detach().cpu().numpy(), dtype=wp.float32, ndim=2)

        # 2, pre-p2g operations (impulse modifiers)
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )

        # 3, dirichlet particle velocity modifiers
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )

    # ==================================================================
    # Phase B : compute stress = stress(returnMap(F_trial))
    # Lines 1172-1185 of user's p2g2p.
    # ==================================================================
    def _phase_B_stress(self, dt, device):
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True, print=False, dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )

    # ==================================================================
    # Phase C : P2G (APIC mass+momentum + stress-divergence force)
    # Lines 1202-1216 of user's p2g2p.
    # ==================================================================
    def _phase_C_p2g(self, dt, device):
        with wp.ScopedTimer(
            "p2g", synchronize=True, print=False, dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
            wp.synchronize()

    # ==================================================================
    # Phase D : grid normalize + gravity + damping + collider + separable resolve
    # Lines 1222-1342 of user's p2g2p.
    # ==================================================================
    def _phase_D_grid_update(self, dt, device, grid_size, grid_size_nsub,
                             grid_size_subj, grid_size_3d):
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile,
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size_nsub),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
            if self.mpm_model.use_separable_contact == 1:
                wp.launch(
                    kernel=grid_normalization_and_gravity_separable,
                    dim=grid_size_subj,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )

        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile,
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # separable-contact resolve
        if self.mpm_model.use_separable_contact == 1:
            wp.launch(
                kernel=compute_vcm_and_resolve,
                dim=grid_size_3d,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )
            wp.launch(
                kernel=apply_bounding_box_separable,
                dim=grid_size_subj,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

            # OPTIONAL extra constraint: project cloth subject's resolved grid
            # velocity off the kinematic SMPL-X body mesh. Stops body MPM from
            # leaking through cloth (separable contact alone is no-tension only).
            if self.body_mesh_collider is not None:
                self.body_mesh_collider.launch(self.mpm_model)
                self.body_mesh_collider.collide(self.mpm_state)

    # ==================================================================
    # Phase E : G2P
    # Lines 1350-1361 of user's p2g2p.
    # ==================================================================
    def _phase_E_g2p(self, dt, device):
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile,
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
            wp.synchronize()

    # ==================================================================
    # Phase F : post-G2P (bone shape matching when enabled, time advance)
    # Lines 1480-1707 + 1726-1728 of user's p2g2p.
    # User's code has shape_matching=False locally as default, so this is
    # mostly a no-op; we mirror the structure faithfully in case it's enabled
    # by editing user's code.
    # ==================================================================
    def _phase_F_post_g2p(self, frame, step, dt, device):
        shape_matching = False  # mirror user's local default

        if shape_matching:
            with wp.ScopedTimer(
                "zero_bone", synchronize=True, print=False, dict=self.time_profile,
            ):
                wp.launch(
                    kernel=zero_bone,
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )

            with wp.ScopedTimer(
                "shape_matching_center", synchronize=True, print=False, dict=self.time_profile,
            ):
                wp.launch(
                    kernel=shape_matching1_reduce,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()

            with wp.ScopedTimer(
                "shape_matching_solve", synchronize=True, print=False, dict=self.time_profile,
            ):
                wp.launch(
                    kernel=shape_matching2_solve,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()

            with wp.ScopedTimer(
                "shape_matching_particle", synchronize=True, print=False, dict=self.time_profile,
            ):
                wp.launch(
                    kernel=shape_matching3_Rw,
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()

            with wp.ScopedTimer(
                "shape_matching_bone_particle", synchronize=True, print=False, dict=self.time_profile,
            ):
                wp.launch(
                    kernel=shape_matching4_bone_particle,
                    dim=(self.n_particles),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()

        # advance time, mirror user's hun_time bookkeeping
        self.time = self.time + dt
        if step == self.hun_time:
            self.hun_time += 1
