import torch
import torch.nn.functional as F
import trimesh
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AnimatableGaussians import smplx, config
from AnimatableGaussians.dataset.dataset_pose import PoseDataset
import importlib
import numpy as np
import pytorch3d
import math

# SMPL_DIR = '/workspace/physics/PhysGaussian_stress/smpl_models/smplx'
SMPL_DIR = './smpl_models/smplx'

def get_smplx_mesh(subject_param, ag_trainer):
    
    with torch.no_grad():
        
        device = config.device
        color = subject_param['color'] if 'color' in subject_param else torch.tensor([0.8, 0.8, 0.8], device=device) # [3]
        gender = subject_param["gender"]
        betas = torch.tensor(subject_param["betas"], device=device).unsqueeze(0) # [10]
        
        dataset_module = ag_trainer.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        training_dataset = MvRgbDataset(ag_trainer.opt['train']['data']['data_dir'], training = False)
        dataset = PoseDataset(ag_trainer.opt['test']['pose_data']['data_path'], smpl_shape = training_dataset.smpl_data['betas'][0], device=config.device)
        smplx_model = smplx.SMPLX(SMPL_DIR, 
                        gender = gender, 
                        use_pca = False, 
                        num_pca_comps = 45, 
                        flat_hand_mean = True, 
                        batch_size = 1).to(device)
        smplx_faces = smplx_model.faces
        # betas = torch.tensor(subject_param["betas"], device=device).unsqueeze(0) # [1, 10]
        
        dataset.pose_list = dataset.pose_list[subject_param['start_frame']:subject_param['end_frame']]
        
        first_idx = dataset.pose_list[0]
        angle_z_arm = subject_param.get('wide_z_arm', 0.0) # degree
        dataset.body_poses[:, 13*3+2] += math.radians(angle_z_arm)
        dataset.body_poses[:, 14*3+2] -= math.radians(angle_z_arm)
        angle_z_leg = subject_param.get('wide_z_leg', 0.0) # degree
        dataset.body_poses[:, 1*3+2] += math.radians(angle_z_leg)
        dataset.body_poses[:, 2*3+2] -= math.radians(angle_z_leg)
        angle_y_arm = subject_param.get('wide_y_arm', 0.0) # degree
        dataset.body_poses[:, 13*3+1] += math.radians(angle_y_arm)
        dataset.body_poses[:, 14*3+1] -= math.radians(angle_y_arm)
        betas = torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, 10)
        if 0:
            cano_test_pose = torch.zeros([1, 66], device=device)
            cano_test_pose[:, 13*3+2] += math.radians(angle_z_arm)
            cano_test_pose[:, 14*3+2] -= math.radians(angle_z_arm)
            cano_test_pose[:, 1*3+2] += math.radians(angle_z_leg) # 벌리는건 이게 맞다
            cano_test_pose[:, 2*3+2] -= math.radians(angle_z_leg)
            cano_test_pose[:, 13*3+1] += math.radians(angle_y_arm)
            cano_test_pose[:, 14*3+1] -= math.radians(angle_y_arm)
            cano_test_smpl = smplx_model.forward(betas = betas, # [1, 10]
                                    global_orient = torch.zeros([1, 3], device=device), # [1, 3]
                                    transl = torch.zeros([1, 3], device=device), # [1, 3]   
                                    body_pose = cano_test_pose[:, 3:66], # [1, 63]
                                    left_hand_pose = torch.zeros([1, 45], device=device), # [1, 45]
                                    right_hand_pose = torch.zeros([1, 45], device=device) # [1, 45]
                                    )
            cano_test_smpl = trimesh.Trimesh(vertices=cano_test_smpl.vertices[0].detach().cpu().numpy(), faces=smplx_model.faces)
            cano_test_smpl.export('./test_results/cano_smpl_mesh.ply')
        cano_smpl = smplx_model.forward(betas = betas, # [1, 10]
                                global_orient = torch.zeros([1, 3], device=device), # [1, 3]
                                transl = torch.zeros([1, 3], device=device), # [1, 3]   
                                body_pose = torch.zeros([1, 63], device=device), # [1, 63]
                                left_hand_pose = torch.zeros([1, 45], device=device), # [1, 45]
                                right_hand_pose = torch.zeros([1, 45], device=device) # [1, 45]
                                )
        live_smpl = smplx_model.forward(betas = betas,
                                        global_orient = dataset.body_poses[first_idx, :3][None], # [1, 3]
                                        transl = dataset.transl[first_idx][None], # [1, 3]   
                                        body_pose = dataset.body_poses[first_idx, 3: 66][None], # [1, 63]
                                        left_hand_pose = dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
                                        )       
        
        '''
        body_pose_temp = torch.zeros([1, 66], device=device)
        angle_leg_temp = 0
        # body_pose_temp[:, 1*3+2] += math.radians(angle_leg_temp)
        # body_pose_temp[:, 2*3+2] -= math.radians(angle_leg_temp)
        angle_arm_temp = 10
        body_pose_temp[:, 13*3+1] += math.radians(angle_arm_temp)
        body_pose_temp[:, 14*3+1] -= math.radians(angle_arm_temp)
        A_smpl = smplx_model.forward(betas = betas, # [1, 10]
                                global_orient = torch.zeros([1, 3], device=device), # [1, 3]
                                transl = torch.zeros([1, 3], device=device), # [1, 3]   
                                body_pose = body_pose_temp[:, 3:66], # [1, 63]
                                left_hand_pose = torch.zeros([1, 45], device=device), # [1, 45]
                                right_hand_pose = torch.zeros([1, 45], device=device) # [1, 45]
                                )
        
        cano_smpl_mesh = trimesh.Trimesh(vertices=cano_smpl.vertices[0].detach().cpu().numpy(), faces=smplx_model.faces)
        cano_smpl_mesh.export('./test_results/cano_smpl_mesh.ply')
        A_smpl_mesh = trimesh.Trimesh(vertices=A_smpl.vertices[0].detach().cpu().numpy(), faces=smplx_model.faces)
        A_smpl_mesh.export('./test_results/A_smpl_mesh.ply')
        '''
        
        N = cano_smpl.vertices.shape[1] # [1, 10475, 3]
        pose_pts = live_smpl.vertices[0]
        cano_pts = cano_smpl.vertices[0] # [10475, 3]
        lbs = smplx_model.lbs_weights # [10475, 55]
        A_mat = live_smpl.A[0] # [55, 4, 4]
        inv_cano_jnt_mats = torch.linalg.inv(dataset.cano_smpl['A'])

        pt_mats = torch.einsum('nj,jxy->nxy', lbs, A_mat) # posed human xyz
        pos_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]        
        cano_rot = torch.tensor([[1, 0, 0, 0]], device=device).repeat(N, 1)  # [N, 4] [1, 0, 0, 0]
        cano_J = live_smpl.J[0, :22] # [55, 3]
        cano_J = F.pad(cano_J, (0, 1), mode='constant', value=0)   
        
        # import trimesh
        # testmesh = trimesh.Trimesh(vertices=cano_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/cano_smpl_load.ply')
        # testmesh = trimesh.Trimesh(vertices=pos_pts.detach().cpu().numpy())
        # testmesh.export('./test_results/pos_pts_load.ply')
        
        ###
        
        rotations = torch.zeros((N, 4), device=config.device)
        rotations[:, 0] = 1
        opacity = torch.ones(N, 1).to(device) # Avatar opacity
        
        cano_pts_np = cano_pts.detach().cpu().numpy()
        edge_lengths = np.linalg.norm(cano_pts_np[smplx_faces[:,0]] - cano_pts_np[smplx_faces[:,1]], axis=1)
        sigma = 0.15 * np.mean(edge_lengths)
        scales = torch.ones(cano_pts.shape[0], 3).to(config.device) * sigma
        
        # 2. posed_vertices with bone
        # bone_cano
        # bone_index
        # bone_faces
        bone_cano   = torch.empty(0, 3).to(config.device)
        bone_scales = torch.empty(0, 3).to(config.device)
        bone_colors = torch.empty(0, 3).to(config.device)
        bone_index = [0]
        # 뼈 smpl에 fitting 다시 하기
        bone_path = os.path.join(subject_param["osso_path"], 'osso_per_parts', 'part_split_meshes.glb')
        bone = trimesh.load(bone_path)
        bone_faces = []
        bone_faces_idx_base = 0
        for i, (key, val) in enumerate(bone.geometry.items()):
            if i == 7:
                continue
            else:
                val.vertices = (val.vertices - val.centroid) * 0.82 + val.centroid # bone scale
                bone_cano = torch.cat([bone_cano, torch.from_numpy(val.vertices).float().to(config.device)], 0)
                
                edge_lengths = np.linalg.norm(val.vertices[val.edges[:,0]] - val.vertices[val.edges[:,1]], axis=1)
                sigma = 0.05 * np.mean(edge_lengths)
                bone_scales = torch.cat([bone_scales, torch.ones(val.vertices.shape[0], 3).to(config.device) * sigma])
                
                bone_colors = torch.cat([bone_colors, torch.from_numpy(val.visual.vertex_colors[:, :3] / 255).to(config.device)], 0)
                                
                bone_index.append(bone_index[-1] + val.vertices.shape[0])
                bone_faces.append(val.faces + bone_faces_idx_base)
                bone_faces_idx_base += val.vertices.shape[0]
                # print(val.vertices.shape)
                
                # scales.sort(axis=1).values.mean(axis=0)
                # 뼈대 cano랑 rotation 따로 만들어야함
        
        bone_opacity   = torch.zeros(bone_cano.shape[0], 1).to(config.device)  # 투명 뼈           
        # bone_opacity   = torch.ones(bone_cano.shape[0], 1).to(config.device) # 색갈 뼈
        # opacity = torch.zeros_like(opacity) # Avatar 투명
        
        bone_rotations = torch.zeros((bone_cano.shape[0], 4), device=config.device)
        bone_rotations[:, 0] = 1

        bone_pose  = torch.empty(0, 3).to(config.device)
        bone_rot   = torch.empty(0, 4).to(config.device)
        smpl_index = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1, 4, 2, 5, 7, 8]       # 20
        for i in range(len(bone_index)-1):
            bone_pose_i = bone_cano[bone_index[i] : bone_index[i+1]] @ live_smpl.A[0, smpl_index[i], :3, :3].T + live_smpl.A[0, smpl_index[i], :3, 3]
            bone_pose = torch.cat([bone_pose, bone_pose_i])
            bone_rot_i = pytorch3d.transforms.matrix_to_quaternion(live_smpl.A[0, smpl_index[i], :3, :3]).unsqueeze(0).repeat(bone_pose_i.shape[0], 1)
            bone_rot  = torch.cat([bone_rot, bone_rot_i])
        
        # pos_pts   = torch.cat([bone_pose,      pos_pts  ])
        # cano_pts  = torch.cat([bone_cano,      cano_pts ])
        # rotations = torch.cat([bone_rotations, rotations])
        # scales    = torch.cat([bone_scales,    scales   ])
        # colors    = torch.cat([bone_colors, color.repeat(cano_pts.shape[0], 1)], 0) # [N, 3]    
        colors    = color.repeat(cano_pts.shape[0], 1) # [N, 3]    

        gaussian_vals = {
                'positions'     : torch.cat([bone_pose,      pos_pts ]),        # [N, 3]
                'opacity'       : torch.cat([bone_opacity,   opacity  ]),    # [N, 1]
                'scales'        : torch.cat([bone_scales,    scales   ]),     # [N, 3]
                'rotations'     : torch.cat([bone_rotations, rotations]), # [N, 4]            
                'colors'        : torch.cat([bone_colors, torch.flip(colors, dims=(1,))]), # RGBtoBGR, [N, 3]
                'max_sh_degree' : 3 # self.avatar_net.max_sh_degree
            }
        gaussian_vals['opacity'] = torch.clamp(gaussian_vals['opacity'], min=1e-4, max=1.0 - 1e-4)
        
        from scene.gaussian_model import GaussianModel
        posed_gaussians = GaussianModel(sh_degree=gaussian_vals['max_sh_degree'], device=config.device)
        posed_gaussians.create_from_values(gaussian_vals)

        bone_faces_cat = np.concatenate(bone_faces)
        bone_faces_idx = bone_faces_cat.shape[0]
        
        human_sequence = dict()
        human_sequence["pose_dataset"] = dataset
        human_sequence["smplx_model"] = smplx_model
        human_sequence["pos_pts"] = gaussian_vals['positions'] # pos_pts
        human_sequence["cano_pts"] = cano_pts
        human_sequence["colors"] = gaussian_vals['colors']
        # human_sequence["cano_rot"] = cano_rot
        human_sequence["cano_J"] = cano_J
        # human_sequence["joint_mat"] = joint_mat
        # human_sequence["A_mat"] = A_mat
        human_sequence["bone_cano"]  = bone_cano
        human_sequence["bone_index"] = bone_index
        human_sequence["bone_faces_idx"] = bone_faces_idx
        human_sequence["betas"] = betas
        
        faces = torch.from_numpy(np.concatenate([bone_faces_cat, smplx_faces + bone_cano.shape[0]])).to(device) # smplx_faces -> faces
        faces_only_smplx = torch.from_numpy(smplx_faces.astype(np.int64)).to(device)
        vertex_colors = torch.concat([bone_colors, torch.rand(N, 3, device=device)])
        
        # mesh = trimesh.Trimesh(vertices=pos_pts.detach().cpu().numpy(), faces=smplx_faces, process=False)
        # mesh.export("./test_results/pose_smpl.ply")        
        # mesh = trimesh.Trimesh(vertices=human_sequence["pos_pts"].detach().cpu().numpy()[:bone_index[-1]],faces=faces[:bone_faces_idx].detach().cpu().numpy(),process=False)
        # mesh.export("./test_results/pose_bone.ply")        
        # mesh = trimesh.Trimesh(vertices=bone_cano.detach().cpu().numpy(),faces=faces[:bone_faces_idx].detach().cpu().numpy(),process=False)
        # mesh.export("./test_results/cano_bone.ply")        
        # mesh = trimesh.Trimesh(vertices=cano_pts.detach().cpu().numpy(),faces=smplx_faces,process=False)
        # mesh.export("./test_results/cano_smpl.ply")        
        # mesh = trimesh.Trimesh(vertices=cano_J[:, :3].detach().cpu().numpy(),process=False)
        # mesh.export("./test_results/cano_J.ply")        
        # pose_J = torch.einsum('nxy,ny->nx', A_mat[:22, :3, :3], cano_J[:, :3]) + A_mat[:22, :3, 3]
        # mesh = trimesh.Trimesh(vertices=pose_J[:, :3].detach().cpu().numpy(),process=False)
        # mesh.export("./test_results/pose_J.ply")
        # print("export")

    return posed_gaussians, human_sequence, faces, faces_only_smplx, vertex_colors