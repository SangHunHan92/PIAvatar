"""Thin inner-shell variant of particle filling.

Instead of volumetrically filling the body cavity (filling_warp.py), this module
adds a SINGLE inner layer of particles by offsetting each canonical surface
vertex inward along its mesh normal by a configurable thickness. The result:

  - 1:1 with surface particles (same count as N_surface)
  - LBS weights inherited 1:1 from the corresponding surface vertex (no k-NN
    blending — anatomically perfect)
  - Forward-LBS using the same A_first_55 hack convention as the velocity
    pipeline so positions_now_total_pos[interior] matches particle_x_ori[interior]
    by construction

Why this exists: full filling (filling_warp.py) restored dent successfully but
overcoupled bones via the MPM grid (kabsch-drift feedback). Thin shell adds
through-thickness stress propagation with O(N_surface) extra particles instead
of O(volume) — fewer particles → lighter coupling → less drift, while still
giving the surface a "wall to push against" when impacted.

Drop-in compatible with fill_particles_subjects_warp's stash convention:
  smplx_model._interior_canonical, smplx_model._interior_lbs

Triggered by setting `"shell_mode": "thin"` in the subject's `particle_filling`
block. Default (no shell_mode key, or "volumetric") falls through to the
existing volumetric Warp filling.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .filling_warp import _propagate_attrs_via_knn  # reuse attr propagation


def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals (outward, assuming faces are CCW from outside)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_n = torch.cross(v1 - v0, v2 - v0, dim=-1)  # area-weighted face normal

    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], face_n)
    vn.index_add_(0, faces[:, 1], face_n)
    vn.index_add_(0, faces[:, 2], face_n)
    return torch.nn.functional.normalize(vn, dim=-1, eps=1e-12)


def _build_A_first_55(human_seq, device):
    """Mirror the A_now_55 hack convention used in compute_smplx_velocity_tgt.
    Same as filling_warp._fill_human_canonical:579-610. Centralized so both
    fillers stay in sync.
    """
    smplx_model = human_seq["smplx_model"]
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
    A55 = torch.zeros((55, 4, 4), device=device, dtype=live_smpl.A.dtype)
    A55[:22] = live_smpl.A[0, :22]
    A55[22:25] = live_smpl.A[0, 15]
    A55[25:40] = live_smpl.A[0, 20] @ only_finger.A[0, 25:40]
    A55[40:55] = live_smpl.A[0, 21] @ only_finger.A[0, 40:55]
    return A55


def _thin_shell_human_canonical(human_seq, subject_param, sim_params, device='cuda'):
    """Build a 1:1 inner shell for a human subject.

    Returns:
      interior_first_frame: [N_surface, 3]
      interior_canonical:   [N_surface, 3]
      interior_lbs:         [N_surface, 55]   (= lbs_weights, 1:1 inherited)
    """
    fp = subject_param["particle_filling"]
    thickness = float(fp.get("shell_thickness", 0.02))  # canonical-space metres

    if "faces_only_smplx" not in subject_param:
        empty = torch.empty((0, 3), device=device, dtype=torch.float32)
        empty_lbs = torch.empty((0, 55), device=device, dtype=torch.float32)
        print("[thin_shell] no faces_only_smplx — skipping shell")
        return empty, empty, empty_lbs

    cano_pts = human_seq["cano_pts"].to(device).float()      # [N_surface, 3]
    faces = subject_param["faces_only_smplx"].to(device).long()
    smplx_model = human_seq["smplx_model"]
    lbs_full = smplx_model.lbs_weights.to(device).float()    # [N_surface, 55]
    surface_n = cano_pts.shape[0]
    if lbs_full.shape[0] != surface_n:
        raise RuntimeError(
            f"[thin_shell] lbs_weights ({lbs_full.shape[0]}) != cano_pts ({surface_n}); "
            "1:1 inheritance assumption broken."
        )

    # Inward offset along outward vertex normal
    vn_out = _compute_vertex_normals(cano_pts, faces)         # outward
    interior_canonical = cano_pts - thickness * vn_out         # inward shift
    interior_lbs = lbs_full.contiguous()                       # 1:1 inheritance

    # Forward-LBS to first frame using A_first_55 (same convention as velocity compute)
    A55 = _build_A_first_55(human_seq, device)
    pt_mats = torch.einsum('nj,jxy->nxy', interior_lbs, A55)
    interior_first_frame = (
        torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], interior_canonical)
        + pt_mats[..., :3, 3]
    )

    print(f"[thin_shell] surface_n={surface_n}, thickness={thickness}, "
          f"|interior|={interior_canonical.shape[0]}")
    return interior_first_frame, interior_canonical, interior_lbs


def thin_shell_subjects(subjects, subject_params, sim_params, device='cuda',
                        human_sequences=None):
    """Drop-in alternative to fill_particles_subjects_warp for thin-shell mode.

    For each subject with `particle_filling.shell_mode == 'thin'`:
      - generates an inner layer (1:1 with surface)
      - appends to subject['pos'] and propagates cov/opacity/shs/index/screen_points
        by 1:1 copy from the corresponding surface particle (k=1 nearest is exact
        because each interior particle has a unique surface partner)
      - stashes _interior_canonical / _interior_lbs on smplx_model

    Subjects without `particle_filling`, or with shell_mode != 'thin', pass through
    unchanged (cov padded to [N, 6] for parity with filling_new).
    """
    init_gs_nums = []
    for i, (subject, param) in tqdm(list(enumerate(zip(subjects, subject_params)))):
        gs_num = subject['pos'].shape[0]
        init_gs_nums.append(gs_num)

        fp = param.get("particle_filling", None)
        is_thin = (
            fp is not None
            and str(fp.get("shell_mode", "")).lower() == "thin"
            and param.get("human", False)
            and human_sequences is not None
            and human_sequences[i] is not None
            and "smplx_model" in human_sequences[i]
        )
        if not is_thin:
            # parity with filling_new: pad cov to [N, 6] if needed
            mpm_init_cov = torch.zeros((gs_num, 6), device=device)
            mpm_init_cov[:gs_num] = subject['cov']
            subject['cov'] = mpm_init_cov
            continue

        interior_first_frame, interior_canonical, interior_lbs = (
            _thin_shell_human_canonical(human_sequences[i], param, sim_params, device=device)
        )
        smplx_model = human_sequences[i]["smplx_model"]
        smplx_model._interior_canonical = interior_canonical
        smplx_model._interior_lbs = interior_lbs

        n_interior = interior_first_frame.shape[0]
        if n_interior == 0:
            continue

        # 1:1 attribute inheritance from the matching surface particle.
        # subject['pos'] is laid out as [bone, surface] with bone_n bone particles
        # at the head and surface_n=cano_pts.shape[0] surface particles after.
        # Interior is surface-only (1:1 with surface), so we copy attrs from the
        # SURFACE PORTION ONLY — taking [bone_n:] of each. Otherwise propagated
        # length would be bone_n+surface_n while pos only grows by surface_n,
        # leading to subject['index'] being longer than pos and corrupting
        # particle_id assignment in MPM.
        bone_n = human_sequences[i]["bone_cano"].shape[0]
        surface_pos = subject['pos'].to(device)
        attrs = {
            "cov":     subject["cov"].to(device),
            "opacity": subject["opacity"].to(device),
            "shs":     subject["shs"].to(device),
            "index":   subject["index"].to(device),
        }
        propagated = {k: v[bone_n:].clone() for k, v in attrs.items()}

        new_pos = torch.cat([surface_pos, interior_first_frame], dim=0)
        subject["pos"]     = new_pos
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

        if sim_params.get("debug", False):
            from mpm_solver_warp.engine_utils import particle_position_tensor_to_ply
            particle_position_tensor_to_ply(
                subject['pos'],
                f"./log/{sim_params['name']}/thin_shell_particles_{i:02d}.ply",
            )

    return subjects, init_gs_nums
