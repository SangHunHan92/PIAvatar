import argparse
import math
import cv2
import torchvision
import json
import copy
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

import os, sys
sys.path.append("gaussian-splatting")
import smplx
import numpy as np
import torch
import torch.nn.functional as F
from AnimatableGaussians import config
import trimesh

class HumanModel:
    def __init__(self, model_path, human_params, device='cuda'):
        self.smpl_model = None
        self.vertices = None
        self.faces = None
        self.gaussian = None
        self.device = device
        self.dataset = model_path.split('/')[2]
        self.human_params = human_params 
        self.model_path = model_path
        self.smpl_output = None
        
    def load_smplx(self):
        
        dataset = self.model_path.split('/')[2]
        if dataset == 'actorshq':
            smplx_data_path = os.path.join(self.model_path, 'smpl_params.npz')
            smpl_data = np.load(smplx_data_path)
            smpl_data = dict(smpl_data)
            
        if 'Actor01' in self.model_path:
            smpl_data['body_pose'][:, 6*3: 7*3] = 0.
            smpl_data['body_pose'][:, 7*3: 8*3] = 0.
            
        smpl_data = {k: torch.from_numpy(v).to(torch.float32).to(self.device) for k, v in smpl_data.items()}
        frame_num = smpl_data['body_pose'].shape[0]
        # self.body_poses = torch.zeros((frame_num, 72), dtype = torch.float32)
        self.global_orient = smpl_data['global_orient']
        self.body_poses = smpl_data['body_pose']
        self.transl = smpl_data['transl']
        self.expression = smpl_data['expression']
        self.jaw_pose = smpl_data['jaw_pose']
        
        self.betas = smpl_data['betas'].repeat(frame_num, 1)
        
        if 'left_hand_pose' in smpl_data:
            self.left_hand_pose = smpl_data['left_hand_pose']
        else:
            self.left_hand_pose = config.left_hand_pose[None].expand(self.body_poses.shape[0], -1)
        if 'right_hand_pose' in smpl_data:
            self.right_hand_pose = smpl_data['right_hand_pose']
        else:
            self.right_hand_pose = config.right_hand_pose[None].expand(self.body_poses.shape[0], -1)
            
        self.pose_list = list(range(0, self.body_poses.shape[0], 1))
                
        if self.dataset == 'actorshq':
            self.smpl_model = smplx.create(model_path = self.human_params['smplx_path'], 
                                           model_type='smplx',
                                        gender = 'neutral', 
                                        use_pca = False, 
                                        num_pca_comps = 45, 
                                        batch_size = frame_num,
                                        # flat_hand_mean = True, 
                                        ).to(self.device)
        else:
            self.smpl_model = smplx.create(
                model_path = self.human_params['smplx_path'],
                model_type = 'smplx',
                gender = 'neutral',
                ext = "pkl",
            ).to(self.device)
        
        self.smpl_output = self.smpl_model.forward(
                            betas = self.betas,
                            expression=self.expression,
                            global_orient = self.global_orient,
                            transl = self.transl,
                            body_pose = self.body_poses,
                            left_hand_pose = self.left_hand_pose,
                            right_hand_pose = self.right_hand_pose,                                      
                            # jaw_pose=self.jaw_pose,
                            )
        
        
        v_temp = self.smpl_output.vertices[0]
        min_pos = torch.min(v_temp, 0)[0]                        # transform2origin
        max_pos = torch.max(v_temp, 0)[0]
        max_diff = torch.max(max_pos - min_pos)
        original_mean_pos = (min_pos + max_pos) / 2.0
        scale = 1.0 / max_diff
        tensor111 = torch.tensor([1.0, 1.0, -1.0], device="cuda") # shift2center111
        
        # for i in range(frame_num):
        #     v_s = self.smpl_output.vertices[i]            
        #     v_s = (v_s - original_mean_pos) * scale
        #     v_s = v_s + tensor111
        #     v_s = v_s.cpu().detach().numpy()            
        #     f_s = self.smpl_model.faces
        #     m_s = trimesh.Trimesh(vertices=v_s, faces=f_s)
        #     os.makedirs('./test', exist_ok=True)
        #     m_s.export('./test/smpl_{0:06d}.obj'.format(i))
        
        # test
        # v_s = smpl_output.vertices[0].cpu().detach().numpy()
        # f_s = self.smpl_model.faces
        # m_s = trimesh.Trimesh(vertices=v_s, faces=f_s)
        # os.makedirs('./test', exist_ok=True)
        # m_s.export('./test/smpl_test.obj')
        
        return self.smpl_output.vertices[0].cpu().detach().numpy()
    
    def moving_human(self, human_params, time_params, bc_params):
        
        fps = human_params["smplx_fps"]          # 30
        # substep_dt = time_params["substep_dt"]   # 5e-05        
        # self.smpl_output.vertices # (N, 10475, 3)       
        
        # substep_dt = time_params["substep_dt"]
        # frame_dt = time_params["frame_dt"]
        # frame_num = time_params["frame_num"]
        # step_per_frame = int(frame_dt / substep_dt)
        
        for i in range(len(self.smpl_output.vertices) - 1):
            if i == 20:
                break
            bc_param = {
                "type": "enforce_smpl_velocity",
                "start_time" : i / fps,
                "end_time"   : (i + 1) / fps,
                # "velocity"   : (self.smpl_output.vertices[i+1] - self.smpl_output.vertices[i]).detach().cpu().numpy() * substep_dt / fps, # 1.0 / smpl_fps초 동안 움직인 거리
                "velocity"   : (self.smpl_output.vertices[i+1] - self.smpl_output.vertices[i]).detach().cpu().numpy() / fps, # 1.0 / smpl_fps초 동안 움직인 거리
            }
            bc_params.append(bc_param)
            
        # bc_params[1]['velocity'] / substep_dt * fps
        # self.smpl_output.vertices[1] - self.smpl_output.vertices[0]
        
        
        return bc_params
    
    def moving_human2(self, human_params, time_params, bc_params):
        
        fps = human_params["smplx_fps"]          # 30
        substep_dt = time_params["substep_dt"]   # 5e-05        
        # self.smpl_output.vertices # (N, 10475, 3)       
        
        # substep_dt = time_params["substep_dt"]
        # frame_dt = time_params["frame_dt"]
        # frame_num = time_params["frame_num"]
        # step_per_frame = int(frame_dt / substep_dt)
        
        for i in range(len(self.smpl_output.vertices) - 1):
            if i == 100:
                break
            bc_param = {
                "type": "enforce_smpl_velocity",
                "start_time" : i / fps,
                "end_time"   : (i + 1) / fps,
                # "velocity"   : (self.smpl_output.vertices[i+1] - self.smpl_output.vertices[i]).detach().cpu().numpy() * substep_dt / fps, # 1.0 / smpl_fps초 동안 움직인 거리
                "velocity"   : (self.smpl_output.vertices[i+1] - self.smpl_output.vertices[i]).detach().cpu().numpy() / fps / substep_dt, # 1.0 / smpl_fps초 동안 움직인 거리
            }
            bc_params.append(bc_param)
            
        # bc_params[1]['velocity'] / substep_dt * fps
        # self.smpl_output.vertices[1] - self.smpl_output.vertices[0]
        
        
        return bc_params

def load_params_from_gs_and_human(
    pc: GaussianModel, human: GaussianModel, pipe, scaling_modifier=1.0, override_color=None, device="cuda"
):
    
    # init_pos = params["pos"]
    # init_cov = params["cov3D_precomp"]
    # init_screen_points = params["screen_points"]
    # init_opacity = params["opacity"]
    # init_shs = params["shs"]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            torch.concat([pc.get_xyz, human.get_xyz]), dtype=pc.get_xyz.dtype, requires_grad=True, device=device # Assume pc.get_xyz.dtype == human.get_xyz.dtype
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = torch.concat([pc.get_xyz, human.get_xyz])
    means2D = screenspace_points
    opacity = torch.concat([pc.get_opacity, human.get_opacity])

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.get_xyz.shape[0] == 0:
            cov3D_precomp = human.get_covariance(scaling_modifier)
        else:
            cov3D_precomp = torch.concat([pc.get_covariance(scaling_modifier), human.get_covariance(scaling_modifier)]) # (N, 6)
    else:
        scales = torch.concat([pc.get_scaling, human.get_scaling])
        rotations = torch.concat([pc.get_rotation, human.get_rotation])

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = torch.concat([pc.get_features, human.get_features]) # Not need to requires_grad?
    else:
        colors_precomp = override_color

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    # Human particle mask
    human_particle = torch.cat([torch.zeros(pc.get_xyz.shape[0]), torch.ones(human.get_xyz.shape[0])], dim=0).to(device) # (N, )

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
        "human_particle": human_particle
    }

def merge_gaussians(pc1: GaussianModel, pc2: GaussianModel, pipe, scaling_modifier=1.0, override_color=None, device="cuda"):
    
    new_gs = GaussianModel(sh_degree=3, device=device)
    
    new_gs._xyz = torch.concat([pc1.get_xyz, pc2.get_xyz])
    new_gs._features_dc = torch.concat([pc1._features_dc, pc2._features_dc])
    new_gs._features_rest = torch.concat([pc1._features_rest, pc2._features_rest])
    new_gs._opacity = torch.concat([pc1._opacity, pc2._opacity])
    new_gs._scaling = torch.concat([pc1._scaling, pc2._scaling])
    new_gs._rotation = torch.concat([pc1._rotation, pc2._rotation])

    return new_gs
