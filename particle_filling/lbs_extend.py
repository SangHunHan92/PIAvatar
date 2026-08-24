"""LBS extension for volumetrically-filled human particles.

After particle_filling appends N_interior new positions to subject['pos'] for
human subjects, those new particles have no LBS weights and no canonical
reference position — so kinematic velocity from the SMPL-X / AnimatableGaussians
animation pipeline would be zero, freezing them while the surface skin moves.

This module assigns LBS to each new interior particle by k-NN against the
surface vertices in first-frame-posed space (where positions live before
rotation_subjects() runs). The same k-NN indices select both:
  - lbs_weights of neighbouring surface vertices  → interior_lbs
  - cano_pts of the same neighbours               → interior_canonical
With k=4 and inverse-distance weighting (Bardenhagen-style), the result is a
smooth LBS field over the volume that matches the surface where they meet.

The function then **monkey-patches**:
  - smplx_model.lbs_weights (or avatar_net.lbs) → extended [surface + interior, 55]
  - smplx_model._interior_canonical (or avatar_net._interior_canonical)
                                       ← extended canonical positions
  - human_seq["cano_pts"]              → extended canonical positions

so that compute_*_velocity_* in mpm_human_utils_separable_contact.py picks
them up automatically. The companion change in compute_smplx_velocity_* is
documented in mpm_human_utils_separable_contact.py.
"""

from __future__ import annotations
import torch


def extend_lbs_for_filled_particles(
    subjects,
    subject_params,
    human_sequences,
    init_gs_nums,
    k: int = 4,
    eps: float = 1e-8,
    device: str = "cuda",
):
    """Compute and install LBS / canonical for filled interior particles.

    Must be called AFTER fill_particles_subjects_warp and BEFORE rotation_subjects /
    set_human_model_to_boundary_conditions, so that:
      - subject['pos'][gs_num:] is in first-frame-posed frame (matches pos_pts), and
      - extended cano_pts / lbs_weights propagate into modify_smplx via the bc dict.
    """
    from pytorch3d.ops import knn_points

    for i, (subject, param, human_seq) in enumerate(zip(subjects, subject_params, human_sequences)):
        if not param.get("human", False) or human_seq is None:
            continue
        if "particle_filling" not in param:
            continue

        gs_num = init_gs_nums[i]
        n_total = subject["pos"].shape[0]
        n_interior = n_total - gs_num
        if n_interior <= 0:
            continue

        # If filling already ran in canonical space (sets ._interior_lbs and
        # ._interior_canonical on smplx_model / avatar_net), skip — those values
        # are mathematically consistent with the forward-LBS-derived first-frame
        # interior positions in subject['pos']. Recomputing here would replace
        # them with weaker first-frame-posed k-NN estimates.
        smplx_model = human_seq.get("smplx_model")
        avatar_net = human_seq.get("avatar_net")
        already_set = (
            (smplx_model is not None and getattr(smplx_model, "_interior_lbs", None) is not None)
            or (avatar_net is not None and getattr(avatar_net, "_interior_lbs", None) is not None)
        )
        if already_set:
            param["particle_end"] = param["particle_start"] + subject["pos"].shape[0]
            for j in range(i + 1, len(subject_params)):
                subject_params[j]["particle_start"] += n_interior
                subject_params[j]["particle_end"] += n_interior
            continue

        bone_cano = human_seq["bone_cano"]
        cano_pts = human_seq["cano_pts"]
        bone_n = bone_cano.shape[0]
        surface_n = cano_pts.shape[0]
        if gs_num != bone_n + surface_n:
            raise RuntimeError(
                f"[lbs_extend] subject {i}: gs_num={gs_num} != bone_n+surface_n={bone_n+surface_n}. "
                f"Filling layout assumption violated."
            )

        # Interior positions are in first-frame-posed frame at this point in the pipeline.
        interior_pos = subject["pos"][gs_num:].to(device).contiguous()

        # Surface positions in first-frame-posed frame, 1:1 with cano_pts and lbs.
        if "pos_pts" in human_seq and human_seq["pos_pts"] is not None:
            pos_pts_full = human_seq["pos_pts"].to(device)
            surface_pos = pos_pts_full[bone_n : bone_n + surface_n].contiguous()
        else:
            # Fall back to subject['pos'][:gs_num] which is also [bone_pose, pos_pts]
            surface_pos = subject["pos"][bone_n:gs_num].to(device).contiguous()

        # Pass 1: k-NN in first-frame-posed space → rough interior_canonical estimate
        # This is needed because at this point the interior particles only exist in
        # first-frame-posed frame (the fill ran on subject['pos']). The k-NN here
        # finds anatomical neighbours, but in posed frames with limb folds, neighbours
        # can be wrong (e.g. interior chest particle sees a folded arm vertex as nearest).
        knn1 = knn_points(
            interior_pos.unsqueeze(0),
            surface_pos.unsqueeze(0),
            K=min(k, surface_n),
        )
        knn1_idx = knn1.idx[0]                            # [N_interior, k]
        knn1_d2 = knn1.dists[0].clamp_min(0.0)
        w1 = 1.0 / (knn1_d2.sqrt() + eps)
        w1 = w1 / w1.sum(dim=-1, keepdim=True)            # [N_interior, k]

        # Pick the right LBS source
        smplx_model = human_seq.get("smplx_model")
        avatar_net = human_seq.get("avatar_net")
        if smplx_model is not None:
            lbs_full = smplx_model.lbs_weights.to(device)
        elif avatar_net is not None:
            lbs_full = avatar_net.lbs.to(device)
        else:
            raise RuntimeError(
                f"[lbs_extend] subject {i}: no smplx_model or avatar_net in human_seq"
            )

        cano_pts_dev = cano_pts.to(device)

        # Pass 1 → rough interior_canonical (weighted average of cano_pts at first-frame-posed knn)
        cano_neighbors_1 = cano_pts_dev[knn1_idx]         # [N_interior, k, 3]
        interior_canonical_rough = (w1.unsqueeze(-1) * cano_neighbors_1).sum(dim=1)

        # Pass 2: re-do k-NN in canonical space using the rough estimate.
        # In canonical (T-pose) the limbs are spread out, so Euclidean k-NN
        # respects anatomical adjacency much better than in posed frame.
        knn2 = knn_points(
            interior_canonical_rough.unsqueeze(0).contiguous(),
            cano_pts_dev.unsqueeze(0).contiguous(),
            K=min(k, surface_n),
        )
        knn_idx = knn2.idx[0]
        knn_d2 = knn2.dists[0].clamp_min(0.0)
        weights = 1.0 / (knn_d2.sqrt() + eps)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        lbs_neighbors = lbs_full[knn_idx]                 # [N_interior, k, 55]
        cano_neighbors = cano_pts_dev[knn_idx]            # [N_interior, k, 3]
        interior_lbs = (weights.unsqueeze(-1) * lbs_neighbors).sum(dim=1).to(lbs_full.dtype)
        interior_canonical = (weights.unsqueeze(-1) * cano_neighbors).sum(dim=1).to(cano_pts.dtype)

        # ---- stash interior LBS / canonical without disturbing SMPL-X internals --
        # Do NOT monkey-patch smplx_model.lbs_weights — SMPL-X's forward uses
        # lbs_weights together with v_template / v_posed (both 10475 verts) and
        # would shape-mismatch if we resized lbs_weights. Instead, save interior
        # LBS / canonical as attributes for compute_smplx_velocity_* to splice in
        # at the einsum site.
        if smplx_model is not None:
            smplx_model._interior_lbs = interior_lbs
            smplx_model._interior_canonical = interior_canonical
        if avatar_net is not None:
            avatar_net._interior_lbs = interior_lbs
            avatar_net._interior_canonical = interior_canonical

        # ---- extend canonical reference in human_seq ----------------------------
        human_seq["cano_pts"] = torch.cat([cano_pts.to(device), interior_canonical], dim=0)
        # also extend pos_pts with the interior first-frame-posed positions, so
        # any code reading pos_pts after this point sees aligned shapes
        if "pos_pts" in human_seq and human_seq["pos_pts"] is not None:
            human_seq["pos_pts"] = torch.cat(
                [human_seq["pos_pts"].to(device), interior_pos], dim=0
            )

        # ---- particle index bookkeeping ------------------------------------------
        param["particle_end"] = param["particle_start"] + subject["pos"].shape[0]
        for j in range(i + 1, len(subject_params)):
            subject_params[j]["particle_start"] += n_interior
            subject_params[j]["particle_end"] += n_interior

    return subjects, subject_params, human_sequences
