"""Body loader for a1_s1 (MPMAvatar assets), parallel to utils/dataset.load_smplx
without modifying the original.

Produces the same return signature as load_smplx (subject_param model="smplx")
so that downstream load_gaussian_subjects + set_human_model_to_boundary_conditions
+ modify_smplx work unchanged. No particle_filling.

Body model = SMPL-X v_shaped (10475 verts, this actor's betas).
Bones      = OSSO GLB at subject_param["osso_path"] (same as get_smplx_mesh).
Pose seq   = MPMAvatar's per-frame .pth at smplx_fitted/{frame:06d}.pth, latents
             decoded via VPoser.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


class _A1PoseDataset:
    """Duck-typed pose_dataset matching what mpm_human_utils.modify_smplx expects.

    Attributes accessed downstream:
      - pose_list: list[int], frame indices (we use 0..N-1 local index)
      - smpl_model: SMPL-X torch module (for forward/parents)
      - smpl_shape: betas tensor of shape (num_betas,)
      - body_poses[i]: tensor (66,) = orient(3) || body_pose(63)
      - transl[i]:     tensor (3,)
      - left_hand_pose[i]:  tensor (45,)
      - right_hand_pose[i]: tensor (45,)
      - cano_smpl: dict with key 'A' -> tensor (1, 55, 4, 4)
    """

    def __init__(self, smpl_model, smpl_shape, body_poses, transl,
                 left_hand_pose, right_hand_pose, cano_A):
        self.smpl_model = smpl_model
        self.smpl_shape = smpl_shape
        self.body_poses = body_poses
        self.transl = transl
        self.left_hand_pose = left_hand_pose
        self.right_hand_pose = right_hand_pose
        self.pose_list = list(range(body_poses.shape[0]))
        self.cano_smpl = {"A": cano_A}


def _build_pose_dataset(subject_param, smplx_model, vposer, device):
    """Decode MPMAvatar pose data into a pose_dataset duck object.

    Two source formats supported:
      (a) per-frame .pth (default): smplx_fitted/{frame:06d}.pth with `latent`
          (VPoser-encoded body pose). Uses subject_param["start_frame"], "end_frame".
      (b) packed .npz (e.g., a1_sitting.npz): subject_param["pose_npz"] points
          at it. Has direct `body_pose` (no VPoser needed). All N frames used.

    MPMAvatar's pose data is Y-up (SMPL-X canonical, head at +Y); our framework
    is Z-up. A fixed +90 deg X-axis rotation is composed onto every frame's
    global_orient and applied to transl so the avatar stands upright in world.
    """
    import pytorch3d.transforms as p3t

    R_stand = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=device,
    )

    body_poses_l = []
    transl_l = []
    lh_l = []
    rh_l = []
    betas_l = []
    expr_l = []

    pose_npz = subject_param.get("pose_npz", None)
    if pose_npz is not None:
        # ---- (b) packed npz (e.g., a1_sitting.npz, AMASS converted) ----
        # AMASS poses already encode Y->Z stand-up in `orient`; setting
        # `pose_npz_already_zup: true` in subject_param skips the R_stand compose.
        already_zup = bool(subject_param.get("pose_npz_already_zup", False))
        d = np.load(pose_npz)
        n_npz = d["body_pose"].shape[0]
        start = int(subject_param.get("start_frame", 0))
        end = min(int(subject_param.get("end_frame", n_npz)), n_npz)
        n_frames = end - start
        for i in range(start, end):
            body_aa = torch.from_numpy(d["body_pose"][i]).to(device).float()      # (63,)
            orient = torch.from_numpy(d["orient"][i]).to(device).float()          # (3,)
            if already_zup:
                orient_new = orient
                trans_new = torch.from_numpy(d["trans"][i]).to(device).float()
            else:
                R_orient = p3t.rotation_conversions.axis_angle_to_matrix(orient[None])
                R_new = R_stand @ R_orient[0]
                orient_new = p3t.rotation_conversions.matrix_to_axis_angle(R_new[None])[0]
                trans_orig = torch.from_numpy(d["trans"][i]).to(device).float()
                trans_new = R_stand @ trans_orig
            full66 = torch.cat([orient_new, body_aa])                              # (66,)
            body_poses_l.append(full66)
            transl_l.append(trans_new)
            lh_l.append(torch.from_numpy(d["left_hand_pose"][i]).to(device).float())
            rh_l.append(torch.from_numpy(d["right_hand_pose"][i]).to(device).float())
            betas_l.append(torch.from_numpy(d["beta"][i]).to(device).float())
            expr_l.append(torch.from_numpy(d["expr"][i]).to(device).float())
    else:
        # ---- (a) per-frame .pth (default) ----
        start = int(subject_param["start_frame"])
        end = int(subject_param["end_frame"])
        n_frames = end - start
        base_dir = "third_party/MPMAvatar/data/a1_s1/smplx_fitted"
        for frame in range(start, end):
            pth_path = os.path.join(base_dir, f"{frame:06d}.pth")
            if not os.path.exists(pth_path):
                raise FileNotFoundError(f"MPMAvatar pose file missing: {pth_path}")
            pose_d = torch.load(pth_path, map_location="cpu")
            latent = pose_d["latent"].to(device)
            body_aa = p3t.rotation_conversions.matrix_to_axis_angle(
                vposer.decode(latent).view(-1, 3, 3)
            ).view(latent.shape[0], -1)                              # (1, 63)
            orient = pose_d["orient"].to(device)                     # (1, 3) axis-angle
            R_orient = p3t.rotation_conversions.axis_angle_to_matrix(orient)
            R_new = R_stand @ R_orient[0]
            orient_new = p3t.rotation_conversions.matrix_to_axis_angle(R_new[None])
            full66 = torch.cat([orient_new, body_aa], dim=1).squeeze(0)
            body_poses_l.append(full66)
            trans_orig = pose_d["trans"].to(device).squeeze(0)
            transl_l.append(R_stand @ trans_orig)
            lh_l.append(pose_d["left_hand_pose"].to(device).squeeze(0))
            rh_l.append(pose_d["right_hand_pose"].to(device).squeeze(0))
            betas_l.append(pose_d["beta"].to(device).squeeze(0))
            expr_l.append(pose_d["expr"].to(device).squeeze(0))

    body_poses = torch.stack(body_poses_l)          # (n_frames, 66)
    transl = torch.stack(transl_l)                  # (n_frames, 3)
    left_hand_pose = torch.stack(lh_l)              # (n_frames, 45)
    right_hand_pose = torch.stack(rh_l)             # (n_frames, 45)

    # ---- pre-pose modifiers (mirror smpl_mesh.get_smplx_mesh) ----
    # Apply to body_pose[:, 3:66] (skip global orient at [:, :3]). Joint indexing
    # within body_pose: joint i has axes at body_pose[:, (i-1)*3 + axis]
    # because joint 0 (pelvis) is the root_orient and not in body_pose.
    # Wait actually body_pose convention: index k of body_pose = (joint_index k+1)
    # i.e. body_pose[k, 0:3] = joint k+1 axis-angle. So L_Collar=joint13 -> body_pose[12,:].
    # But existing get_smplx_mesh uses body_poses[:, 13*3+axis] which means
    # body_poses includes global_orient at [:, :3] then body_pose at [:, 3:66].
    # So body_poses[:, 13*3+axis] = body_poses[:, 39+axis] = joint13 axis (L_Collar). OK.
    import math as _math
    a_y_arm = _math.radians(float(subject_param.get("wide_y_arm", 0)))
    a_z_arm = _math.radians(float(subject_param.get("wide_z_arm", 0)))
    a_z_leg = _math.radians(float(subject_param.get("wide_z_leg", 0)))
    if a_y_arm != 0:
        body_poses[:, 13 * 3 + 1] += a_y_arm   # L_Collar y
        body_poses[:, 14 * 3 + 1] -= a_y_arm   # R_Collar y
    if a_z_arm != 0:
        body_poses[:, 13 * 3 + 2] += a_z_arm   # L_Collar z
        body_poses[:, 14 * 3 + 2] -= a_z_arm   # R_Collar z
    if a_z_leg != 0:
        body_poses[:, 1 * 3 + 2] += a_z_leg    # L_Hip z
        body_poses[:, 2 * 3 + 2] -= a_z_leg    # R_Hip z

    # Use first frame's beta as the actor shape (constant across sequence in MPMAvatar)
    smpl_shape = betas_l[0]                          # (300,)
    expression0 = expr_l[0]                          # (100,)

    # Optional: override betas via subject_param["betas_override"] (list of floats,
    # padded with zeros to 300 dims). Useful to swap MPMAvatar's 300-dim fitted
    # betas for the original ActorsHQ 10-dim betas.
    bo = subject_param.get("betas_override", None)
    if bo is not None and len(bo) > 0:
        bo_arr = np.asarray(bo, dtype=np.float32).flatten()
        smpl_shape_new = torch.zeros(300, dtype=torch.float32, device=device)
        smpl_shape_new[: min(300, len(bo_arr))] = torch.from_numpy(bo_arr[:300]).to(device)
        smpl_shape = smpl_shape_new
        print(f"[a1_loader] betas_override applied: first 10 = {bo_arr[:10].tolist()}")

    # Canonical (T-pose) A matrix for this actor - used by modify_smplx via
    # inv_cano_jnt_mats = torch.linalg.inv(pose_dataset.cano_smpl['A']).
    cano_out = smplx_model.forward(
        betas=smpl_shape[None],
        global_orient=torch.zeros(1, 3, device=device),
        transl=torch.zeros(1, 3, device=device),
        body_pose=torch.zeros(1, 63, device=device),
        left_hand_pose=torch.zeros(1, 45, device=device),
        right_hand_pose=torch.zeros(1, 45, device=device),
        expression=expression0[None],
        jaw_pose=torch.zeros(1, 3, device=device),
        leye_pose=torch.zeros(1, 3, device=device),
        reye_pose=torch.zeros(1, 3, device=device),
    )
    cano_A = cano_out.A.detach()                     # (1, 55, 4, 4)

    pd = _A1PoseDataset(
        smpl_model=smplx_model,
        smpl_shape=smpl_shape,
        body_poses=body_poses,
        transl=transl,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        cano_A=cano_A,
    )
    return pd, n_frames


def _build_bones(subject_param, live_smpl_A, device):
    """Load OSSO bones from subject_param['osso_path'] (same convention as get_smplx_mesh).

    Returns (bone_cano, bone_pose, bone_rot, bone_scales, bone_colors,
             bone_index, bone_faces_cat, bone_opacity).
    """
    import pytorch3d
    bone_path = os.path.join(subject_param["osso_path"],
                             "osso_per_parts", "part_split_meshes.glb")
    bone = trimesh.load(bone_path)

    bone_cano = torch.empty(0, 3, device=device)
    bone_scales = torch.empty(0, 3, device=device)
    bone_colors = torch.empty(0, 3, device=device)
    bone_index = [0]
    bone_faces = []
    bone_faces_idx_base = 0

    smpl_index = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  1, 4, 2, 5, 7, 8]                              # 20 parts

    for i, (key, val) in enumerate(bone.geometry.items()):
        if i == 7:
            continue
        val.vertices = (val.vertices - val.centroid) * 0.82 + val.centroid
        bone_cano = torch.cat(
            [bone_cano, torch.from_numpy(val.vertices).float().to(device)], 0
        )
        edge_lengths = np.linalg.norm(
            val.vertices[val.edges[:, 0]] - val.vertices[val.edges[:, 1]], axis=1
        )
        sigma = 0.05 * np.mean(edge_lengths)
        bone_scales = torch.cat(
            [bone_scales,
             torch.ones(val.vertices.shape[0], 3, device=device) * sigma], 0
        )
        bone_colors = torch.cat(
            [bone_colors,
             torch.from_numpy(val.visual.vertex_colors[:, :3] / 255).to(device)], 0
        )
        bone_index.append(bone_index[-1] + val.vertices.shape[0])
        bone_faces.append(val.faces + bone_faces_idx_base)
        bone_faces_idx_base += val.vertices.shape[0]

    bone_pose = torch.empty(0, 3, device=device)
    bone_rot = torch.empty(0, 4, device=device)
    for i in range(len(bone_index) - 1):
        joint_idx = smpl_index[i]
        Ai = live_smpl_A[0, joint_idx]                           # (4, 4)
        seg = bone_cano[bone_index[i]: bone_index[i + 1]]
        bp = seg @ Ai[:3, :3].T + Ai[:3, 3]
        bone_pose = torch.cat([bone_pose, bp])
        br = pytorch3d.transforms.matrix_to_quaternion(Ai[:3, :3])
        bone_rot = torch.cat(
            [bone_rot, br.unsqueeze(0).repeat(bp.shape[0], 1)]
        )

    bone_rotations = torch.zeros((bone_cano.shape[0], 4), device=device)
    bone_rotations[:, 0] = 1
    bone_opacity = torch.zeros(bone_cano.shape[0], 1, device=device)
    bone_faces_cat = np.concatenate(bone_faces)
    return (bone_cano, bone_pose, bone_rot, bone_scales, bone_colors,
            bone_index, bone_faces_cat, bone_opacity, bone_rotations)


def _get_a1_mesh(subject_param):
    """Build (posed_gaussians, human_sequence, faces, faces_only_smplx,
    vertex_colors) for an a1_s1 subject. Mirrors get_smplx_mesh's outputs."""
    from AnimatableGaussians import smplx as ag_smplx
    from human_body_prior.train.vposer_smpl import VPoser
    from scene.gaussian_model import GaussianModel

    device = "cuda"

    # AnimatableGaussians' SMPLX exposes .A (per-joint transform) on the output,
    # which modify_smplx requires. shapedirs in MPMAvatar's body model is 400-dim
    # → supports 300 betas + 100 expression.
    smplx_model = ag_smplx.SMPLX(
        model_path="third_party/MPMAvatar/data/body_models/smplx",
        ext="npz", gender="neutral",
        num_betas=300, num_expression_coeffs=100,
        use_face_contour=False, use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).eval().to(device)

    vposer = VPoser(512, 32, [3, 21]).eval().to(device)
    vposer.load_state_dict(torch.load(
        "third_party/MPMAvatar/data/body_models/TR00_E096.pt",
        map_location="cpu",
    ))

    # Per-frame poses
    pose_dataset, n_frames = _build_pose_dataset(
        subject_param, smplx_model, vposer, device
    )
    first_idx = pose_dataset.pose_list[0]

    # First-frame live SMPL-X (for initial posed positions)
    live_smpl = smplx_model.forward(
        betas=pose_dataset.smpl_shape[None],
        global_orient=pose_dataset.body_poses[first_idx, :3][None],
        transl=pose_dataset.transl[first_idx][None],
        body_pose=pose_dataset.body_poses[first_idx, 3:66][None],
        left_hand_pose=pose_dataset.left_hand_pose[first_idx][None],
        right_hand_pose=pose_dataset.right_hand_pose[first_idx][None],
    )

    # Canonical SMPL-X (this actor's shape, T-pose). Preferentially load from
    # a1_canonical_assets.npz (bit-identical with un-posed cloth). But if betas
    # were overridden via subject_param, recompute cano_pts to match the new
    # shape (otherwise body would be the OLD shape, mismatching pose dynamics).
    can_npz_path = "third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz"
    if os.path.exists(can_npz_path) and subject_param.get("betas_override", None) is None:
        can = np.load(can_npz_path)
        cano_pts = torch.from_numpy(can["smplx_v_canonical"]).float().to(device)
    else:
        cano_smpl = smplx_model.forward(
            betas=pose_dataset.smpl_shape[None],
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
        )
        cano_pts = cano_smpl.vertices[0]
    smplx_faces = smplx_model.faces                              # (20908, 3) int

    N = cano_pts.shape[0]
    pos_pts = live_smpl.vertices[0]                              # (10475, 3)

    # cano_J (first 22 joints, padded as 4-vec like get_smplx_mesh)
    cano_J = live_smpl.J[0, :22]
    cano_J = F.pad(cano_J, (0, 1), mode="constant", value=0)

    # Default per-vertex scale (Gaussian sigma) - same heuristic as get_smplx_mesh
    cano_pts_np = cano_pts.detach().cpu().numpy()
    edge_lengths = np.linalg.norm(
        cano_pts_np[smplx_faces[:, 0]] - cano_pts_np[smplx_faces[:, 1]], axis=1
    )
    sigma = 0.15 * np.mean(edge_lengths)
    scales = torch.ones(N, 3, device=device) * sigma
    rotations = torch.zeros((N, 4), device=device)
    rotations[:, 0] = 1
    opacity = torch.ones(N, 1, device=device)

    # Bones (OSSO) - placed using live SMPL-X A
    (bone_cano, bone_pose, bone_rot, bone_scales, bone_colors,
     bone_index, bone_faces_cat, bone_opacity, bone_rotations) = _build_bones(
        subject_param, live_smpl.A, device,
    )

    # Body color (random per-actor, same scheme as load_smplx)
    g = torch.Generator(device=device)
    g.manual_seed(int(subject_param.get("index", 0)) * 10007 + 1)
    body_color = torch.rand(3, generator=g, device=device)
    body_color = (body_color + 1.0) / 2.0
    colors = body_color.repeat(N, 1)                             # (N, 3)

    # GaussianModel (bones come first, then body - matches get_smplx_mesh order)
    from scene.gaussian_model import GaussianModel
    gaussian_vals = {
        "positions":    torch.cat([bone_pose,      pos_pts]),
        "opacity":      torch.cat([bone_opacity,   opacity]),
        "scales":       torch.cat([bone_scales,    scales]),
        "rotations":    torch.cat([bone_rotations, rotations]),
        "colors":       torch.cat([bone_colors, torch.flip(colors, dims=(1,))]),
        "max_sh_degree": 3,
    }
    gaussian_vals["opacity"] = torch.clamp(
        gaussian_vals["opacity"], min=1e-4, max=1.0 - 1e-4
    )
    posed_gaussians = GaussianModel(sh_degree=gaussian_vals["max_sh_degree"], device=device)
    posed_gaussians.create_from_values(gaussian_vals)

    bone_faces_idx = bone_faces_cat.shape[0]
    faces = torch.from_numpy(
        np.concatenate([bone_faces_cat, smplx_faces + bone_cano.shape[0]])
    ).to(device)
    faces_only_smplx = torch.from_numpy(smplx_faces.astype(np.int64)).to(device)
    vertex_colors = torch.cat([bone_colors, colors])

    human_sequence = dict()
    human_sequence["pose_dataset"] = pose_dataset
    human_sequence["smplx_model"] = smplx_model
    human_sequence["pos_pts"] = gaussian_vals["positions"]
    human_sequence["cano_pts"] = cano_pts
    human_sequence["colors"] = gaussian_vals["colors"]
    human_sequence["cano_J"] = cano_J
    human_sequence["bone_cano"] = bone_cano
    human_sequence["bone_index"] = bone_index
    human_sequence["bone_faces_idx"] = bone_faces_idx
    human_sequence["betas"] = pose_dataset.smpl_shape[None]      # (1, 300)

    return posed_gaussians, human_sequence, faces, faces_only_smplx, vertex_colors


def load_smplx_a1(subject_param, mesh_id):
    """Drop-in replacement for utils.dataset.load_smplx, but reads MPMAvatar's
    a1_s1 assets (per-frame .pth + OSSO + a1_canonical_assets.npz). Original
    load_smplx is left untouched. No particle_filling needed.

    Returns: (posed_gaussians, human_sequence, subject_param) - same as load_smplx.
    """
    from threedgrut_playground.engine import PBRMaterial
    from AnimatableGaussians import config

    posed_gaussians, human_sequence, faces, faces_only_smplx, vertex_colors = \
        _get_a1_mesh(subject_param)

    material_id = mesh_id
    g = torch.Generator(device=config.device)
    g.manual_seed(material_id * 10007)
    color = torch.rand(3, generator=g, device=config.device)
    color = (color + 1.0) / 2.0
    material = PBRMaterial(
        material_id=material_id,
        diffuse_map=torch.tensor(
            torch.cat([color, torch.tensor([1.0], device=config.device)]),
            device=config.device, dtype=torch.float32,
        ).expand(2, 2, 4),
        diffuse_factor=torch.ones(4, device=config.device, dtype=torch.float32),
        emissive_factor=torch.zeros(3, device=config.device, dtype=torch.float32),
        metallic_factor=0.0,
        roughness_factor=0.0,
        transmission_factor=0.0,
        ior=1.0,
    )

    human_sequence["rotation_degree"] = subject_param["rotation_degree"]
    human_sequence["rotation_axis"] = subject_param["rotation_axis"]
    human_sequence["center"] = subject_param["center"]
    human_sequence["scale"] = subject_param["scale"]

    subject_param["faces"] = faces
    subject_param["faces_only_smplx"] = faces_only_smplx
    subject_param["vertex_colors"] = vertex_colors
    subject_param["render_material"] = material
    subject_param["render_material_index"] = mesh_id
    subject_param["render_material_name"] = "smplx_a1_" + str(mesh_id)
    subject_param["pytorch3d_diffuse_map"] = material.diffuse_map
    num_verts_smplx = int(faces_only_smplx.max().item()) + 1
    subject_param["pytorch3d_verts_uvs"] = torch.ones(
        num_verts_smplx, 2, device=faces_only_smplx.device, dtype=torch.float32
    ) * 0.5
    subject_param["pytorch3d_faces_uvs"] = faces_only_smplx

    return posed_gaussians, human_sequence, subject_param
