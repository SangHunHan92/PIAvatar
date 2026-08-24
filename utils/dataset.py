import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "gaussian-splatting"))

import torch
import numpy as np
import glob
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud
from utils.human_utils import HumanModel
from utils.transformation_utils import *
from mpm_solver_warp.engine_utils import *
from utils.smpl_mesh import get_smplx_mesh
from kaolin.rep import SurfaceMesh
from kaolin.io import gltf
from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes, PBRMaterial
from PIL import Image

# gaussians = load_checkpoint(model_path)
# human_model, human_gaussians = load_human(model_path, human_params, device) # Initialize SMPL-X Gaussians from 1st frame
# bc_params = human_model.moving_human(human_params, time_params, bc_params) # add smpl velocity to bc_params

def load_subjects(
    in_subjects, subject_params, sim_params, pipe, scaling_modifier=1.0, device="cuda"
):        
    out_subjects = []    
        
    for s, (subject, param) in enumerate(zip(in_subjects, subject_params)):
        
        means3D = None
        means2D = None
        shs = None
        opacity = None
        scales = None
        rotations = None
        cov3D = None
        index = None
        
        # "crop area" is defined in original 3D Gaussians space        
        mask = torch.ones(subject.get_xyz.shape[0], dtype=torch.bool).to(device=device) # [N]
        
        if param['human'] is False:
            if param["crop_area"] is not None:
                assert len(param["crop_area"]) == 6            
                for i in range(3):
                    mask = torch.logical_and(mask, subject.get_xyz[:, i] > param["crop_area"][2 * i])
                    mask = torch.logical_and(mask, subject.get_xyz[:, i] < param["crop_area"][2 * i + 1])
                    
            # "opaticity threshold"
            mask = torch.logical_and(mask, subject.get_opacity[:, 0] > param["opacity_threshold"])
        
        # Gaussian splatting        
        means3D = subject.get_xyz[mask, :]
        opacity = subject.get_opacity[mask, :]
        index = torch.ones_like(subject.get_xyz[mask, 0:1], dtype=torch.int, device=device) * param["index"]
        shs = subject.get_features[mask, :]
            
        if pipe.compute_cov3D_python:
            cov3D = subject.get_covariance(scaling_modifier)[mask, :] # [N, 6]
        else:
            scales = subject.get_scaling[mask, :]      # used ?
            rotations = subject.get_rotation[mask, :]  # used ?
        
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=device) + 0
        try:
            means2D.retain_grad()
        except:
            pass

        assert all(num == means3D.shape[0] for num in [means2D.shape[0], opacity.shape[0], index.shape[0]])
        
        out_subjects.append({
            "pos": means3D, # ([N, 3])
            "screen_points": means2D,
            "shs": shs,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "cov": cov3D,
            "index": index
        })
        
        if s == 0:
            param['particle_start'] = 0
            param['particle_end'] = means3D.shape[0]
        else:
            param['particle_start'] = subject_params[s-1]['particle_end']
            param['particle_end'] = subject_params[s-1]['particle_end'] + means3D.shape[0]
        # print(param['particle_start'], param['particle_end'])
        
    if sim_params["debug"]:
        # if not os.path.exists(f"./log/{sim_params['name']}"):
        #     os.makedirs(f"./log/{sim_params['name']}")
        os.makedirs(f"./log/{sim_params['name']}", exist_ok=True)
        for i, subject_temp in enumerate(out_subjects):
            particle_position_tensor_to_ply(
                subject_temp["pos"],
                f"./log/{sim_params['name']}/init_particles_{i:02d}.ply",
            )
    
        
    return out_subjects, subject_params

def load_gs_subjects(
    gaussian_subjects, subject_params, sim_params, pipe, scaling_modifier=1.0, device="cuda"
):        
    subjects = []    
        
    for gaussian, subject in zip(gaussian_subjects, subject_params):
        
        means3D = None
        means2D = None
        shs = None
        opacity = None
        scales = None
        rotations = None
        cov3D = None
        index = None
        
        # "crop area" is defined in original 3D Gaussians space        
        mask = torch.ones(gaussian.get_xyz.shape[0], dtype=torch.bool).to(device=device) # [N]
        
        if subject['human'] is False:
            if subject["crop_area"] is not None:
                assert len(subject["crop_area"]) == 6            
                for i in range(3):
                    mask = torch.logical_and(mask, gaussian.get_xyz[:, i] > subject["crop_area"][2 * i])
                    mask = torch.logical_and(mask, gaussian.get_xyz[:, i] < subject["crop_area"][2 * i + 1])
                    
            # "opaticity threshold"
            mask = torch.logical_and(mask, gaussian.get_opacity[:, 0] > subject["opacity_threshold"])
        
        # Gaussian splatting        
        means3D = gaussian.get_xyz[mask, :]
        opacity = gaussian.get_opacity[mask, :]
        index = torch.ones_like(gaussian.get_xyz[mask, 0:1], dtype=torch.int, device=device) * subject["index"]
        shs = gaussian.get_features[mask, :]
            
        if pipe.compute_cov3D_python:
            cov3D = gaussian.get_covariance(scaling_modifier)[mask, :] # [N, 6]
        else:
            scales = gaussian.get_scaling[mask, :]      # used ?
            rotations = gaussian.get_rotation[mask, :]  # used ?
        
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=device) + 0
        try:
            means2D.retain_grad()
        except:
            pass

        assert all(num == means3D.shape[0] for num in [means2D.shape[0], opacity.shape[0], index.shape[0]])
        
        subjects.append({
            "pos": means3D, # ([N, 3])
            "screen_points": means2D,
            "shs": shs,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "cov": cov3D,
            "index": index
        })
    
    if sim_params["debug"]:
        # if not os.path.exists(f"./log/{sim_params['name']}"):
        #     os.makedirs(f"./log/{sim_params['name']}")
        os.makedirs(f"./log/{sim_params['name']}", exist_ok=True)
        for i, subject in enumerate(subjects):
            particle_position_tensor_to_ply(
                subject["pos"],
                f"./log/{sim_params['name']}/init_particles_{i:02d}.ply",
            )
        
    return subjects

def rotation_subjects(subjects, subject_params, sim_params):
    
    for i, (subject, param) in enumerate(zip(subjects, subject_params)):
        # if subject["rotation_degree"] is not None:
        rotation_matrices = generate_rotation_matrices( # rotate all gaussians
            torch.tensor(param["rotation_degree"]),
            param["rotation_axis"],
        )
        param["rot_mats"] = rotation_matrices
        
        rotated_pos = apply_rotations(subject["pos"], rotation_matrices)
        subject["pos"] = rotated_pos        
        subject["cov"] = apply_cov_rotations(subject["cov"], rotation_matrices) # [N, 6]
        subject["cov"] = param["scale"] * param["scale"] * subject["cov"]
        
        if sim_params["debug"]:
            particle_position_tensor_to_ply(
                subject["pos"],
                f"./log/{sim_params['name']}/rotated_particles_{i:02d}.ply",
            )
            
    return subjects, subject_params

def shift_scale_subjects(subjects, subject_params, sim_params):
    
    for i, (subject, param) in enumerate(zip(subjects, subject_params)):
        # remember inv_scale & transform2origin
        # transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos) # [N, 3], [1], [3]
        subject["pos"], original_mean_pos = transform2center(subject["pos"], param["scale"], param["center"])        
        param["ori_mean"] = original_mean_pos
        # undotransform2center(scene["pos"], original_mean_pos, subject["scale"], subject["center"]) # Check !        
        
        if sim_params["debug"]:
            particle_position_tensor_to_ply(
                subject["pos"],
                f"./log/{sim_params['name']}/transformed_particles_{i:02d}.ply",
            )
        
    return subjects, subject_params

def merge_subjects(subjects, init_gs_nums, pipe, device='cuda'):
        
    def merge(multiple, single):
        if multiple == None:
            return single
        else:
            return torch.cat((multiple, single))
    
    # merged = {}
    pos = None
    vol = None
    cov = None
    shs = None
    opacity = None
    screen_points = None
    index = None
    scales = None
    rotations = None
    
    for subject, init_gs_num in zip(subjects, init_gs_nums):
        pos = merge(pos, subject["pos"][:init_gs_num])
        screen_points = merge(screen_points, subject["screen_points"][:init_gs_num])
        opacity = merge(opacity, subject["opacity"][:init_gs_num])
        shs = merge(shs, subject["shs"][:init_gs_num])
        index = merge(index, subject["index"][:init_gs_num])
        
        if pipe.compute_cov3D_python:
            cov = merge(cov, subject["cov"][:init_gs_num])
        else:
            scales = merge(scales, subject["scales"][:init_gs_num])              # used ?
            rotations = merge(rotations, subject["rotations"][:init_gs_num])     # used ?
    
    # filled particles should be located at the end
    for subject, init_gs_num in zip(subjects, init_gs_nums):
        pos = merge(pos, subject["pos"][init_gs_num:])
        screen_points = merge(screen_points, subject["screen_points"][init_gs_num:])
        opacity = merge(opacity, subject["opacity"][init_gs_num:])
        shs = merge(shs, subject["shs"][init_gs_num:])
        index = merge(index, subject["index"][init_gs_num:])
        
        if pipe.compute_cov3D_python:
            cov = merge(cov, subject["cov"][init_gs_num:])
        else:
            scales = merge(scales, subject["scales"][init_gs_num:])              # used ?
            rotations = merge(rotations, subject["rotations"][init_gs_num:])     # used ?
            
    return {
        "pos": pos,
        "cov": cov,
        "shs": shs,
        "opacity": opacity,
        "screen_points": screen_points,
        "index": index[:, 0],
        "scales": scales,
        "rotations": rotations,
    }

def load_gaussian_subjects(subject_params, sim_params, device="cuda"):
        
    gaussian_subjects = []
    human_sequences = []
    particle_start = 0
    mesh_id = 2
    for subject_param in subject_params:
        if subject_param["human"]:
            if subject_param["model"] == "animatable_gaussians":
                subject, human_sequence = load_ag(subject_param) # gaussians                
            elif subject_param["model"] == "smplx":
                subject, human_sequence, subject_param = load_smplx(subject_param, mesh_id) # mesh
                mesh_id += 1
            elif subject_param["model"] == "smplx_a1":
                # MPMAvatar a1_s1 assets (Option A): SMPL-X body + OSSO bones,
                # per-frame poses from third_party/MPMAvatar/data/a1_s1/smplx_fitted/
                from hetero_cloth.a1_body_loader import load_smplx_a1
                subject, human_sequence, subject_param = load_smplx_a1(subject_param, mesh_id)
                mesh_id += 1
            else:
                assert False, "Implemented only for AnimatableGaussians, smplx, smplx_a1"
            human_sequence["particle_start"] = particle_start
            subject_param["particle_start"] = particle_start
            subject_param["particle_end"] = particle_start + human_sequence["cano_pts"].shape[0] + human_sequence["bone_cano"].shape[0]
            if subject_param["model"] in ("smplx", "smplx_a1"):
                subject_param["index_mesh"] = torch.arange(subject_param["particle_start"], subject_param["particle_end"])
                subject_param['index_mesh_only_smplx'] = torch.arange(subject_param["particle_start"] + human_sequence["bone_cano"].shape[0], subject_param["particle_end"])
            particle_start += human_sequence["cano_pts"].shape[0] + human_sequence["bone_cano"].shape[0]
        else:
            if subject_param["model"] == "gaussian":
                subject = load_checkpoint(subject_param["model_path"])
                human_sequence = None
            elif subject_param["model"] == "mesh_glb":
                subject, subject_param = load_glb(subject_param["model_path"], subject_param, mesh_id)
                human_sequence = None
                mesh_id += 1
            else:
                assert False, "Implemented only for Mesh, Gaussian"
            subject_param["particle_start"] = particle_start
            subject_param["particle_end"] = particle_start + subject.get_xyz.shape[0]
            subject_param["index_mesh"] = torch.arange(subject_param["particle_start"], subject_param["particle_end"])
            particle_start += subject.get_xyz.shape[0]
        gaussian_subjects.append(subject)
        human_sequences.append(human_sequence)
        
    if sim_params["debug"]:
        for i, (subject, human_param) in enumerate(zip(gaussian_subjects, human_sequences)):
            if isinstance(subject, GaussianModel):
                subject.save_ply(f"./log/{sim_params['name']}/loaded_gaussian_{i:02d}.ply")
            if human_param is not None :
                if 'faces' in human_param :
                    save_path = f"./log/{sim_params['name']}/loaded_mesh_{i:02d}.ply"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    mesh = trimesh.Trimesh(vertices=human_param["pos_pts"].detach().cpu().numpy(),
                                        faces=human_param["faces"],
                                        colors=human_param["colors"].detach().cpu().numpy(),
                                        process=False)
                    mesh.export(save_path)
                
    return gaussian_subjects, human_sequences

# Not Used !!
def load_human_sequences(subject_params, human_params, device="cuda"):
    
    # return human vertices sequence or LBS sequence    
    human_sequences = []
    for subject_param in subject_params:
        if subject_param["human"]:
            if subject_param["human_model"] == "animatable_gaussians":
                human_sequence = load_ag_sequence(subject_param["config_path"])
        else:
            human_sequence = None
        human_sequences.append(human_sequence)
    return human_sequences

def load_checkpoint(model_path, sh_degree=3, iteration=-1, device="cuda"):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"        
    )   
    # Load guassians
    gaussians = GaussianModel(sh_degree, device=device)
    gaussians.load_ply(checkpt_path)
    return gaussians

def load_glb(model_path, subject_param, mesh_id, device="cuda"):
    
    name = os.path.basename(model_path)
    glb_path = os.path.join(model_path, f"{name}.glb")
    if not os.path.isfile(glb_path):
        glb_path = glob.glob(os.path.join(model_path, "*.glb"))
    
    mesh = gltf.import_mesh(glb_path).to(device)   
    N = mesh.vertices.shape[0]
            
    pos = mesh.vertices
    pos_np = pos.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    opacity = torch.ones(N, 1).to(device)
    
    rotations = torch.zeros((N, 4), device=device)
    rotations[:, 0] = 1
        
    # if hasattr(mesh.visual, 'vertex_colors'):
    #     colors = torch.from_numpy(mesh.visual.vertex_colors).to(device)
    # elif hasattr(mesh.visual, 'uv'):
    #     colors = torch.from_numpy(mesh.face_colors).to(device)
    
    edge_lengths = np.linalg.norm(pos_np[faces[:,0]] - pos_np[faces[:,1]], axis=1)
    sigma = 0.05 * np.mean(edge_lengths)
    scales = torch.ones(pos.shape[0], 3).to(device) * sigma
    
    mesh_ = trimesh.load(glb_path, force='mesh')
    ##########
    # colors = mesh_.visual.to_color().vertex_colors # Temp color, 3dgrut render mesh with diffuse_map
    if hasattr(mesh.materials[0], 'diffuse_texture'):
        tex_np = mesh.materials[0].diffuse_texture.detach().cpu().numpy()  # [H, W, 3]
        out_ply_path = './test_results/ball.ply'
        colors = make_vertex_colors_from_texture(glb_path, tex_np, out_ply_path)
        
    elif hasattr(mesh_.visual, 'vertex_colors') and mesh_.visual.vertex_colors is not None:
        colors = mesh_.visual.vertex_colors

    # 2. face_colors만 존재할 경우 (vertex로 확장)
    elif hasattr(mesh_.visual, 'face_colors') and mesh_.visual.face_colors is not None:
        colors = mesh_.visual.face_colors.repeat(mesh_.faces.shape[1], axis=0)

    # 3. 텍스처 기반 (UV + material)
    elif hasattr(mesh_.visual, 'material') and hasattr(mesh_.visual.material, 'image') and mesh_.visual.material.image is not None:
        # UV 좌표를 사용해 텍스처 색 샘플링
        # mesh.materials[0].diffuse_texture
        uv = mesh_.visual.uv
        tex = np.array(mesh_.visual.material.image)
        h, w = tex.shape[:2]
        uv_pix = np.clip(uv * [w - 1, h - 1], 0, [w - 1, h - 1]).astype(int)
        colors = tex[h - 1 - uv_pix[:, 1], uv_pix[:, 0]]

    # 4. 없을 경우 기본 회색
    else:
        colors = np.ones((len(mesh_.vertices), 4), dtype=np.uint8) * 200
    #########
    
    colors = torch.from_numpy(colors[:, :3]) / 255.0
    
    gaussian_vals = {
        'positions'     : pos,        # [N, 3]
        'opacity'       : opacity,    # [N, 1]
        'scales'        : scales,     # [N, 3]
        'rotations'     : rotations, # [N, 4]
        'colors'        : colors, # RGBtoBGR, [N, 3]
        # 'colors'        : torch.cat([bone_colors, torch.flip(colors, dims=(1,))]), # RGBtoBGR, [N, 3]
        'max_sh_degree' : 3
    }
    gaussian_vals['opacity'] = torch.clamp(gaussian_vals['opacity'], min=1e-4, max=1.0 - 1e-4)

    from scene.gaussian_model import GaussianModel
    posed_gaussians = GaussianModel(sh_degree=gaussian_vals['max_sh_degree'], device=device)
    posed_gaussians.create_from_values(gaussian_vals)
    
    from threedgrut_playground.utils.mesh_io import load_materials
    materials = load_materials(mesh, device)
    material_id = mesh_id
    for mat in materials: # PLEASE 1 material
        # material_id = len(engine.primitives.registered_materials)
        material = PBRMaterial(
                        material_id=material_id,
                        diffuse_map=mat['diffuse_map'].to(device),
                        # emissive_map=mat['emissive_map'],
                        # metallic_roughness_map=mat['metallic_roughness_map'],
                        # normal_map=mat['normal_map'],
                        diffuse_factor=mat['diffuse_factor'].to(device),                        
                        emissive_factor= mat.get('emissive_factor', torch.zeros(3, device=device, dtype=torch.float32)),
                        metallic_factor=mat['metallic_factor'],
                        roughness_factor=mat['roughness_factor'],
                        # alpha_mode=mat['alpha_mode'],
                        # alpha_cutoff=mat['alpha_cutoff'],
                        transmission_factor=mat['transmission_factor'],
                        ior=mat['ior'],
                    )
        continue
        
    # posed_mesh = dict()
    # posed_mesh["pos_pts"] = cano_pts
    # posed_mesh["faces"] = faces
    # posed_mesh["vertex_colors"] = vertex_colors
    # posed_mesh["face_uvs"] = face_uvs
    # posed_mesh["face_colors"] = face_colors
    # posed_mesh["scales"] = scales
    
    subject_param["faces"] = mesh.faces
    subject_param["uvs"] = mesh.uvs
    subject_param["face_uvs_idx"] = mesh.face_uvs_idx
    subject_param["render_material"] = material
    subject_param["render_material_index"] = mesh_id
    subject_param["render_material_name"] = mat['material_name'] + "_" + str(mesh_id)
    # PyTorch3D TexturesUV 전용: diffuse map + UV (render_utils에서 여기만 사용)
    subject_param["pytorch3d_diffuse_map"] = material.diffuse_map  # (H, W, 4) e.g. (2, 2, 4)
    subject_param["pytorch3d_verts_uvs"] = mesh.uvs
    subject_param["pytorch3d_faces_uvs"] = mesh.face_uvs_idx

    return posed_gaussians, subject_param

def load_human(model_path, human_params, device, sh_degree=3):
    human_model = HumanModel(model_path, human_params, device=device)
    human_vertices = human_model.load_smplx() # [10475, 3]
    
    human_gaussians = GaussianModel(sh_degree, device=device)
    pcd = BasicPointCloud(
        points=human_vertices, 
        colors=np.zeros_like(human_vertices),        
        # colors=np.random.rand(human_vertices.shape[0], human_vertices.shape[1]),
        normals=np.zeros_like(human_vertices)
    )
    human_gaussians.create_from_pcd(pcd, 0.5) # "spatial_lr_scale" should be parameterized
    return human_model, human_gaussians

def load_ag(subject_param): #, model_path):
    
    from AnimatableGaussians.main_avatar_phys import AvatarTrainer    
    from AnimatableGaussians import config
    config.load_global_opt(subject_param["config_path"])
    config.opt['test']['pose_data']['data_path'] = subject_param["pose_path"]
    config.opt['test']['pose_data']['frame_range'] = [subject_param["start_frame"], subject_param["end_frame"]]
    ag_trainer = AvatarTrainer(config.opt)
    
    # Full version of AnimatableGaussians, posing with lbs and network
    posed_gaussians, avatar_net, pose_dataset, cano_pts, cano_rot, cano_J, joint_mat, A_mat, extr, knn_indices, bone_cano, bone_index, bone_faces \
        = ag_trainer.get_avatars(subject_param)
    human_sequence = dict()
    human_sequence["avatar_net"] = avatar_net
    human_sequence["pose_dataset"] = pose_dataset
    human_sequence["cano_pts"] = cano_pts
    human_sequence["cano_rot"] = cano_rot
    human_sequence["cano_J"] = cano_J
    human_sequence["joint_mat"] = joint_mat
    human_sequence["A_mat"] = A_mat
    human_sequence["knn_indices"] = knn_indices
    human_sequence["bone_cano"]  = bone_cano
    human_sequence["bone_index"] = bone_index
    human_sequence["bone_faces"] = bone_faces    
    
    human_sequence["rotation_degree"] = subject_param["rotation_degree"]
    human_sequence["rotation_axis"] = subject_param["rotation_axis"]
    human_sequence["center"] = subject_param["center"]
    human_sequence["scale"] = subject_param["scale"]    
    
    return posed_gaussians, human_sequence # human_sequence에 ag_trainer 넣기, human 수마다 network...?

def load_smplx(subject_param, mesh_id): #, model_path):
    
    from AnimatableGaussians.main_avatar_phys import AvatarTrainer
    from AnimatableGaussians import config
    config.load_global_opt(subject_param["config_path"])
    config.opt['test']['pose_data']['data_path'] = subject_param["pose_path"]
    config.opt['test']['pose_data']['frame_range'] = [subject_param["start_frame"], subject_param["end_frame"]]
    ag_trainer = AvatarTrainer(config.opt)
    
    # posed_mesh, pos_pts, cano_pts, smplx_faces, cano_rot, colors, cano_J, joint_mat, A_mat, bone_cano, bone_index, bone_faces_idx
    posed_gaussians, human_sequence, faces, faces_only_smplx, vertex_colors = get_smplx_mesh(subject_param, ag_trainer)
    
    # human_sequence = dict()
    # human_sequence["pos_pts"] = pos_pts
    # human_sequence["cano_pts"] = cano_pts
    # human_sequence["smplx_faces"] = smplx_faces
    # human_sequence["cano_rot"] = cano_rot
    # human_sequence["colors"] = colors
    # human_sequence["cano_J"] = cano_J
    # human_sequence["joint_mat"] = joint_mat
    # human_sequence["A_mat"] = A_mat
    # # human_sequence["knn_indices"] = knn_indices
    # human_sequence["bone_cano"]  = bone_cano
    # human_sequence["bone_index"] = bone_index
    # human_sequence["bone_faces_idx"] = bone_faces_idx
    
    material_id = mesh_id
    
    g = torch.Generator(device=config.device)
    g.manual_seed(material_id * 10007)
    color = torch.rand(3, generator=g, device=config.device)
    color = (color + 1.0) / 2.0
    material = PBRMaterial(
        material_id=material_id,
        # diffuse_map=torch.tensor([255 / 255.0, 193 / 255.0, 130 / 255.0, 1.0],
        diffuse_map=torch.tensor(torch.cat([color, torch.tensor([1.0], device=config.device)]),
                                    device=config.device, dtype=torch.float32).expand(2, 2, 4),
        diffuse_factor=torch.ones(4, device=config.device, dtype=torch.float32),
        emissive_factor=torch.zeros(3, device=config.device, dtype=torch.float32),
        metallic_factor=0.0,
        roughness_factor=0.0,
        transmission_factor=0.0,
        ior=1.0
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
    subject_param["render_material_name"] = "smplx_" + str(mesh_id)
    # PyTorch3D TexturesUV 전용: diffuse map + 더미 UV (smplx는 실제 UV 없음 → 2x2 맵 중심 샘플링으로 단색)
    subject_param["pytorch3d_diffuse_map"] = material.diffuse_map  # (2, 2, 4)
    num_verts_smplx = int(faces_only_smplx.max().item()) + 1
    subject_param["pytorch3d_verts_uvs"] = torch.ones(num_verts_smplx, 2, device=faces_only_smplx.device, dtype=torch.float32) * 0.5
    subject_param["pytorch3d_faces_uvs"] = faces_only_smplx

    return posed_gaussians, human_sequence, subject_param # subject = posed_mesh

#################################################################################################

def bilinear_sample(img_np, uv):
    H, W = img_np.shape[:2]
    img = img_np.astype(np.float32)
    if img.max() > 1.0: img /= 255.0

    u = np.clip(uv[..., 0], 0.0, 1.0)
    v = np.clip(uv[..., 1], 0.0, 1.0)
    v_img = 1.0 - v  # 이미지 좌표계 보정 (행 0이 top)

    x = u * (W - 1)
    y = v_img * (H - 1)
    x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, H - 1)

    wx = x - x0; wy = y - y0
    c00 = img[y0, x0]; c10 = img[y0, x1]; c01 = img[y1, x0]; c11 = img[y1, x1]
    c0 = (1 - wx)[..., None] * c00 + wx[..., None] * c10
    c1 = (1 - wx)[..., None] * c01 + wx[..., None] * c11
    return (1 - wy)[..., None] * c0 + wy[..., None] * c1

def make_vertex_colors_from_texture(glb_path, tex_np, out_ply_path):
    scene = trimesh.load(glb_path)
    mesh = list(scene.geometry.values())[0] if isinstance(scene, trimesh.Scene) else scene

    if mesh.visual.uv is None:
        raise RuntimeError("No UVs on mesh.")

    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    V = mesh.vertices.shape[0]

    uv_fv = uv[faces]  # [F,3,2]
    colors_fv = bilinear_sample(tex_np, uv_fv)  # [F,3,3] float [0,1]

    vertex_colors = np.zeros((V, 3), dtype=np.float32)
    counts = np.zeros((V,), dtype=np.int32)
    for f in range(faces.shape[0]):
        for j in range(3):
            vid = faces[f, j]
            vertex_colors[vid] += colors_fv[f, j]
            counts[vid] += 1
    counts[counts == 0] = 1
    vertex_colors /= counts[:, None]
    vc_u8 = (np.clip(vertex_colors, 0.0, 1.0) * 255).astype(np.uint8)

    # out_mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
    #                            faces=mesh.faces.copy(),
    #                            vertex_colors=vc_u8)
    # out_mesh.export(out_ply_path)
    
    return vc_u8