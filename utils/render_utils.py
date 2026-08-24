import sys
sys.path.append("gaussian-splatting")
import argparse
import math
import cv2
import torchvision
import torch
import os
import numpy as np
import json
import copy
from tqdm import tqdm
import kaolin
from kaolin.rep import SurfaceMesh
import trimesh
import time

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

import torch.nn.functional as nnF
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    TexturesUV,
    BlendParams,
    Materials,
)
from torchvision.transforms.functional import to_pil_image
from pytorch3d.structures import Meshes, join_meshes_as_scene

# PyTorch3D 메시 색상 방식:
# - TexturesVertex: 버텍스당 색 [1,V,3]. 보간으로 면 내부 색 결정. (현재 이 방식, 단색 0.8)
# - TexturesUV: diffuse map 이미지(H,W,3) + verts_uvs(V,2) + faces_uvs(F,3). UV로 텍스처 샘플링.
# simulation_pytorch3d.py 423~483: 3DGRUT 엔진이 render_material.diffuse_map + 메시 UV로 diffuse 렌더링.


def initialize_resterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterize = GaussianRasterizer(raster_settings=raster_settings)
    return rasterize

def load_params_from_gs(
    pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None, device="cuda"
):
    
    # init_pos = params["pos"]
    # init_cov = params["cov3D_precomp"]
    # init_screen_points = params["screen_points"]
    # init_opacity = params["opacity"]
    # init_shs = params["shs"]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.

    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
    }

def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].unsqueeze(2)).squeeze(2)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp

@torch.no_grad()
def metrics_from_F(F, prev_eq=None, prev_abslogJ=None, dt=None, eps=1e-12):
    S = torch.linalg.svdvals(F)                 # [N,3], S>=0
    logS = torch.log(S.clamp_min(eps))          # [N,3]
    m = logS.mean(dim=-1, keepdim=True)         # [N,1]
    dev = logS - m                              # [N,3]
    eq = ((2.0/3.0) * (dev*dev).sum(dim=-1)).sqrt()  # [N]
    logJ = logS.sum(dim=-1)                          # [N]

    if (prev_eq is not None) and (prev_abslogJ is not None) and (dt is not None) and dt > 0:
        shock = ((eq - prev_eq).abs() + (logJ.abs() - prev_abslogJ).abs()) / dt
    else:
        shock = torch.zeros_like(eq)

    return eq, logJ, shock

@torch.no_grad()
def overlay_simple(
    F_,
    base_rgb,
    prev_eq=None, prev_abslogJ=None, dt=None,
    eq_scale=0.30, vol_scale=0.30, shock_scale=0.50,
    # ---- 데드존 임계값들 ----
    eq_tol=0.0001,          # 전단(Hencky 등가변형) 데드존
    vol_tol=0.0001,         # 체적(logJ의 절댓값) 데드존
    shock_tol=0.0005,       # 쇼크(시간 변화율) 데드존
    # ---- 행렬 자체 데드존 ----
    use_F_deadzone=True,
    F_tol=2e-5,           # ||F - I||_F 임계
):
    """
    base_rgb: [N,3] in [0,1]
    컬러 매핑: R=팽창(+vol), G=전단(eq), B=압축(-vol)
    데드존을 넣어 작은 변형에는 색이 거의/전혀 변하지 않게 함.
    """
    # 입력 보존 (연산 중 수정되지 않도록)
    F = F_.detach()

    # 메트릭 계산
    eq, logJ, shock = metrics_from_F(F, prev_eq, prev_abslogJ, dt)

    # --------- 데드존 적용(하드 컷오프) ----------
    # relu(x - tol)로 임계치 이후부터만 증가
    # 전단
    eq_eff = torch.relu(eq - eq_tol)
    # 체적(+/- 따로 쓰므로 |logJ|에 tol 적용 후 다시 부호 분기)
    abs_logJ = logJ.abs()
    abs_logJ_eff = torch.relu(abs_logJ - vol_tol)
    # 부호 복원용 마스크
    pos = (logJ > 0).float()
    neg = 1.0 - pos
    # 쇼크
    shock_eff = torch.relu(shock - shock_tol)

    # --------- 정규화 ----------
    eq_n    = (eq_eff / max(eq_scale, 1e-12)).clamp(0, 1)
    vol_mag = (abs_logJ_eff / max(vol_scale, 1e-12)).clamp(0, 1)
    vol_p   = vol_mag * pos         # 팽창 → R
    vol_n   = vol_mag * neg         # 압축 → B
    shock_n = (shock_eff / max(shock_scale, 1e-12)).clamp(0, 1)

    # --------- 행렬 자체 데드존(옵션) ----------
    if use_F_deadzone:
        # Frobenius norm of (F - I)
        N = F.shape[0]
        I = torch.eye(3, device=F.device, dtype=F.dtype).expand(N, 3, 3)
        Fdiff = (F - I).reshape(N, -1)
        Fmask = (Fdiff.norm(dim=-1) > F_tol).float().unsqueeze(-1)  # [N,1]
        # 매우 작은 변형(노이즈 수준)은 전부 0으로
        eq_n    = eq_n.unsqueeze(-1) * Fmask
        vol_p   = vol_p.unsqueeze(-1) * Fmask
        vol_n   = vol_n.unsqueeze(-1) * Fmask
        shock_n = shock_n.unsqueeze(-1) * Fmask

        # 다시 [N]로 모양 복원
        eq_n    = eq_n.squeeze(-1)
        vol_p   = vol_p.squeeze(-1)
        vol_n   = vol_n.squeeze(-1)
        shock_n = shock_n.squeeze(-1)

    # 오버레이 알파 (가중 합)
    alpha = (0.7*eq_n + 0.4*(vol_p + vol_n).clamp(0, 1) + 0.8*shock_n).clamp(0, 1)

    # 임팩트 컬러: [R,G,B] = [팽창, 전단, 압축]
    impact = torch.stack([vol_p, eq_n, vol_n], dim=-1)  # [N,3]

    out_rgb = (1 - alpha).unsqueeze(-1) * base_rgb + alpha.unsqueeze(-1) * impact
    return out_rgb.clamp(0, 1), eq.detach(), logJ.abs().detach()

'''
@torch.no_grad()
def metrics_from_F(F, prev_eq=None, prev_abslogJ=None, dt=None, eps=1e-12):
    """
    F: [N,3,3] float32 권장
    반환: eq [N], logJ [N], shock [N] (없으면 0)
    """
    # SVD 값만 사용 (U,V 불필요) -> 메모리/시간 절약
    S = torch.linalg.svdvals(F)                  # [N,3], S>=0
    logS = torch.log(S.clamp_min(eps))          # [N,3]

    m = logS.mean(dim=-1, keepdim=True)         # [N,1]
    dev = logS - m                              # [N,3]
    eq = ((2.0/3.0) * (dev*dev).sum(dim=-1)).sqrt()  # Hencky 등가변형 [N]
    logJ = logS.sum(dim=-1)                          # 체적변형 log(det F) [N]

    if (prev_eq is not None) and (prev_abslogJ is not None) and (dt is not None) and dt>0:
        shock = ((eq - prev_eq).abs() + (logJ.abs() - prev_abslogJ).abs()) / dt
    else:
        shock = torch.zeros_like(eq)

    return eq, logJ, shock

@torch.no_grad()
def overlay_simple(F, base_rgb, prev_eq=None, prev_abslogJ=None, dt=None,
                   eq_scale=0.30, vol_scale=0.30, shock_scale=0.50):
    """
    base_rgb: [N,3] in [0,1]
    아주 단순한 오버레이: 빨강(팽창) / 초록(전단) / 파랑(압축)
    """
    eq, logJ, shock = metrics_from_F(F, prev_eq, prev_abslogJ, dt)

    eq_n   = (eq / eq_scale).clamp(0,1)
    vol_p  = ( torch.relu(logJ)   / vol_scale ).clamp(0,1)  # +vol (팽창) -> R
    vol_n  = ( torch.relu(-logJ)  / vol_scale ).clamp(0,1)  # -vol (압축) -> B
    shock_n= (shock / shock_scale).clamp(0,1)

    # 오버레이 세기(알파): 전단/체적/충격을 간단히 결합
    alpha = (0.7*eq_n + 0.4*(vol_p+vol_n).clamp(0,1) + 0.8*shock_n).clamp(0,1)

    # 임팩트 컬러: [R,G,B] = [팽창, 전단, 압축]
    impact = torch.stack([vol_p, eq_n, vol_n], dim=-1)      # [N,3]

    out_rgb = (1 - alpha).unsqueeze(-1)*base_rgb + alpha.unsqueeze(-1)*impact
    return out_rgb.clamp(0,1), eq.detach(), logJ.abs().detach()    
'''

@torch.no_grad()
def overlay_simple2(F, base_rgb, prev_eq=None, prev_abslogJ=None, dt=None,
                   eq_scale=0.30, vol_scale=0.30, shock_scale=0.50):
    """
    base_rgb: [N,3] in [0,1]
    아주 단순한 오버레이: 빨강(팽창) / 초록(전단) / 파랑(압축)
    """
    eq, logJ, shock = metrics_from_F(F, prev_eq, prev_abslogJ, dt)

    eq_n   = (eq / eq_scale).clamp(0,1)
    vol_p  = ( torch.relu(logJ)   / vol_scale ).clamp(0,1)  # +vol (팽창) -> R
    vol_n  = ( torch.relu(-logJ)  / vol_scale ).clamp(0,1)  # -vol (압축) -> B
    shock_n= (shock / shock_scale).clamp(0,1)

    # 오버레이 세기(알파): 전단/체적/충격을 간단히 결합
    alpha1 = (1.0*vol_p + 0.8*shock_n).clamp(0,1)
    alpha2 = (1.0*eq_n + 0.8*shock_n).clamp(0,1)
    alpha3 = (1.0*vol_n + 0.8*shock_n).clamp(0,1)

    # 임팩트 컬러: [R,G,B] = [팽창, 전단, 압축]
    zeros = torch.zeros_like(vol_p)
    impact1 = torch.stack([vol_p, zeros, zeros], dim=-1)      # [N,3]
    impact2 = torch.stack([zeros, eq_n, zeros], dim=-1)      # [N,3]
    impact3 = torch.stack([zeros, zeros, vol_n], dim=-1)      # [N,3]

    out_rgb1 = (1 - alpha1).unsqueeze(-1)*base_rgb + alpha1.unsqueeze(-1)*impact1
    out_rgb2 = (1 - alpha2).unsqueeze(-1)*base_rgb + alpha2.unsqueeze(-1)*impact2
    out_rgb3 = (1 - alpha3).unsqueeze(-1)*base_rgb + alpha3.unsqueeze(-1)*impact3

    return out_rgb1.clamp(0,1), out_rgb2.clamp(0,1), out_rgb3.clamp(0,1), eq.detach(), logJ.abs().detach()


def create_init_mesh(device):
    """ Creates a procedurally generated mesh. """
    MS = 0.00001
    MZ = 0.0 
    v0 = [-MS, -MS, MZ]
    v1 = [-MS, +MS, MZ]
    v2 = [+MS, -MS, MZ]
    v3 = [+MS, +MS, MZ]
    c0 = [1.0, 0.0, 1.0]                # PLEASE check here !!
    c1 = [0.0, 1.0, 1.0] 
    c2 = [1.0, 0.0, 0.0]
    c3 = [0.0, 0.0, 1.0] 
    faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
    vertex_uvs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    mesh = kaolin.rep.SurfaceMesh(
        vertices=torch.tensor([v0, v1, v2, v3]), 
        faces=faces, 
        face_uvs=vertex_uvs[faces].contiguous(), 
        vertex_colors=torch.tensor([c0, c1, c2, c3])) 
    mesh.vertex_tangents = torch.zeros([len(mesh.vertices), 3], dtype=torch.bool)
    # mesh.material_assignments = torch.zeros([len(mesh.faces)], device=device)
    mesh.material_assignments = torch.ones([len(mesh.faces)], device=device)
    return mesh.to(device)

def create_render_mesh(pos, subject_param, render_smpl_only=True, render_bone_only=False, device='cuda'):
    
    if subject_param['model'] in ('smplx', 'smplx_a1'):
        index = subject_param['index_mesh'] # [84971]
        faces = subject_param['faces'] # [169452, 3]        
        if render_smpl_only==True:
            index = subject_param['index_mesh_only_smplx'] # [10475]
            faces = subject_param['faces_only_smplx'] # [20908, 3]
        elif render_bone_only==True:
            index = subject_param['index_mesh'][:-subject_param['index_mesh_only_smplx'].shape[0]]
            faces = subject_param['faces'][:-subject_param['faces_only_smplx'].shape[0]]
        assign = torch.ones((faces.shape[0],), dtype=torch.int16, device=device) * subject_param['render_material_index']
        F = faces.shape[0]
        base_uv = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], device=device)  # (3,2)
        face_uvs = base_uv.unsqueeze(0).repeat(F, 1, 1).contiguous()
        mesh = SurfaceMesh(
            vertices=pos[index],
            faces=faces,
            face_uvs=face_uvs, 
            material_assignments=assign,
        )
        # mesh.vertex_tangents = torch.zeros([len(mesh.vertices), 3], dtype=torch.bool, device=device)                        
    
    elif subject_param['model'] == 'mesh_glb':
        # verts_list[0][:, 0] +=  0.01521 * 2
        # verts_list[0][:, 1] +=  0.01401 * 2
        # verts_list[0][:, 2] += -0.00061 * 2
        index = subject_param['index_mesh']
        assign = torch.ones((subject_param["faces"].shape[0],), dtype=torch.int16, device=device) * subject_param['render_material_index']
        mesh = SurfaceMesh(
            # vertices=pos[index],
            vertices=pos[index] + torch.tensor([0.01521*2, 0.01401*2, -0.00061*2], device=pos.device),
            faces=subject_param["faces"],
            uvs=subject_param["uvs"],
            face_uvs_idx=subject_param["face_uvs_idx"],
            material_assignments=assign,
        )
    return mesh

def my_autoscale_no_unit(scene_scale, mesh, transform, scale_of_new_mesh_to_small_scene=0.5):
    # 1단계 유닛 크기 정규화 제거
    if scene_scale.max() > 5.0:    # 큰 씬이면 아무것도 안 함
        return
    adjusted_scale = scale_of_new_mesh_to_small_scene * scene_scale.to(transform.device)
    largest_axis_scale = adjusted_scale.max()
    transform.scale(largest_axis_scale)
    
def get_lookat_camera(current_camera):
    
    view_matrix = torch.eye(4)
    view_matrix[:3, :3] = torch.tensor(current_camera.R).T 
    view_matrix[:3, 3] = torch.tensor(current_camera.T)
    align_transform = torch.tensor([
        [1.,  0.,  0.,  0.],
        [0., -1.,  0.,  0.], 
        [0.,  0.,  -1.,  0.],
        [0.,  0.,  0.,  1.]
    ], device=view_matrix.device)

    view_matrix = align_transform @ view_matrix
    
    aspect = current_camera.image_width / current_camera.image_height
    FoVy = 2 * math.atan(math.tan(current_camera.FoVx / 2) / aspect)
                
    return view_matrix, FoVy


########################### pytorch3d ###########################
def _fov_to_focal_px(fov_rad: float, size_px: int) -> float:
    return 0.5 * float(size_px) / math.tan(0.5 * float(fov_rad))


def inria_camera_to_pytorch3d(cam, device):
    """
    cam: 너의 Camera 클래스 인스턴스
    """
    H = int(cam.image_height)
    W = int(cam.image_width)

    R_in = cam.R
    T_in = cam.T
    if not torch.is_tensor(R_in):
        R_in = torch.tensor(R_in, device=device, dtype=torch.float32)
    else:
        R_in = R_in.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(T_in):
        T_in = torch.tensor(T_in, device=device, dtype=torch.float32)
    else:
        T_in = T_in.to(device=device, dtype=torch.float32)

    # Gaussian: getWorld2View2에서 W2C 회전 = R^T, PyTorch3D: X_cam = X_world R + T → 열벡터로 p_cam = R^T p_world + T.
    # 따라서 PyTorch3D에 넘길 R은 R_in 그대로 (transpose 하면 시점이 다른 축 기준으로 회전함).
    R_view = R_in.unsqueeze(0) if R_in.dim() == 2 else R_in  # [1,3,3]
    T_view = T_in.unsqueeze(0) if T_in.dim() == 1 else T_in  # [1,3]

    fx = _fov_to_focal_px(cam.FoVx, W)
    fy = _fov_to_focal_px(cam.FoVy, H)
    focal_length = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)

    cx = float(getattr(cam, "cx", 0.0))
    cy = float(getattr(cam, "cy", 0.0))
    principal_point = torch.tensor([[W * 0.5 + cx, H * 0.5 + cy]],
                                   device=device, dtype=torch.float32)

    cameras = PerspectiveCameras(
        device=device,
        R=R_view,
        T=T_view,
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
    )
    return cameras


# PyTorch3D: 1 + (max_image_size - 1) // bin_size < kMaxFacesPerBin(22) 이어야 함.
def _pytorch3d_bin_size_for_image(H, W, bin_size):
    if bin_size is None or bin_size <= 0:
        return bin_size
    max_side = max(int(H), int(W))
    min_bin = int(math.ceil((max_side - 1) / 21.0))
    return max(bin_size, min_bin)


# ---------- Renderer (create once, reuse) ----------
def make_pytorch3d_renderer(device, H, W, bg_rgb=(0.0, 0.0, 0.0),
                            light_loc=(0.0, 0.0, 1.0),
                            specular_color=(0.15, 0.15, 0.15),
                            shininess=32.0,
                            faces_per_pixel=1,
                            bin_size=32,
                            max_faces_per_bin=100000,
                            ambient_color=(0.38, 0.38, 0.38),
                            diffuse_color=(0.52, 0.52, 0.52)):
    """
    faces_per_pixel: 1=빠름, 4=서브픽셀 품질 향상(느림). 품질이 중요하면 4 사용.
    bin_size: 0=naive(느림), 32 등 양수=coarse-to-fine. 고해상도에서는 22 미만 빈 개수 제한을 위해 자동으로 증가.
    max_faces_per_bin: coarse 단계 한 빈당 최대 면 수. None이면 PyTorch3D 휴리스틱 사용. 간헐적 overflow 경고 시 20만 등으로 키우기.
    light_loc: 조명 1개 위치 (x,y,z).
    광택: specular_color=하이라이트 색(0~1), shininess=날카로움(0=없음, 24=약간, 64~128=강함).
    ambient_color: 앰비언트(0~1). 디퓨즈보다 낮게 두면 굴곡 음영 유지, 올리면 전체 밝아짐.
    diffuse_color: 디퓨즈(0~1). 올리면 밝은 부분이 밝아짐. 앰비언트보다 크게 두면 윤곽 대비 유지.
    """
    bin_size = _pytorch3d_bin_size_for_image(H, W, bin_size)
    # coarse 단계 overflow 방지: 휴리스틱(10000, F/5)이 특정 프레임에서 부족할 수 있으므로 기본값을 크게 둠
    mfp = max_faces_per_bin if (max_faces_per_bin is not None and max_faces_per_bin > 0) else None
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=mfp,
        cull_backfaces=False,  # avoid dark holes from inconsistent face winding (e.g. SMPL)
    )
    lights = PointLights(
        device=device,
        location=(light_loc,),
        ambient_color=(ambient_color,),
        diffuse_color=(diffuse_color,),
    )
    blend_params = BlendParams(background_color=bg_rgb)
    materials = Materials(
        device=device,
        specular_color=(specular_color,),
        shininess=shininess,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )
    return renderer


# ---------- Build multiple meshes from your create_render_mesh logic ----------
def _compute_vertex_normals(verts, faces, device):
    """Area-weighted vertex normals for consistent shading (reduces artifacts)."""
    v = verts.to(device=device, dtype=torch.float32)
    f = faces.to(device=device).long()
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    face_normals = torch.nn.functional.normalize(
        torch.cross(e1, e2, dim=1), dim=1
    )
    face_areas = 0.5 * torch.cross(e1, e2, dim=1).norm(dim=1, keepdim=True).clamp(min=1e-8)
    weighted = face_normals * face_areas
    verts_normals = torch.zeros_like(v)
    verts_normals.index_add_(0, f[:, 0], weighted)
    verts_normals.index_add_(0, f[:, 1], weighted)
    verts_normals.index_add_(0, f[:, 2], weighted)
    return torch.nn.functional.normalize(verts_normals, dim=1)


def _surface_mesh_to_pytorch3d_mesh(surface_mesh, device, default_rgb=0.8, fix_face_winding=False,
                                    use_vertex_normals=True, subject_param=None):
    """
    surface_mesh: create_render_mesh가 리턴하는 SurfaceMesh (3DGRUT와 동일).
    PyTorch3D TexturesUV는 3DGRUT가 material 위해 쓰는 UV를 surface_mesh에서 그대로 사용.
    """
    verts = surface_mesh.vertices.to(device=device, dtype=torch.float32)
    faces = surface_mesh.faces.to(device=device).long()
    if fix_face_winding:
        faces = faces[:, [0, 2, 1]]  # flip winding → outward normals

    V = verts.shape[0]
    textures = None
    # 3DGRUT와 동일 소스: diffuse_map은 subject_param, UV는 create_render_mesh가 쓴 surface_mesh에서 추출
    diffuse_map = None
    if subject_param is not None:
        diffuse_map = subject_param.get("pytorch3d_diffuse_map")
        if diffuse_map is None:
            mat = subject_param.get("render_material")
            diffuse_map = getattr(mat, "diffuse_map", None) if mat is not None else None

    verts_uvs = None
    faces_uvs = None
    if hasattr(surface_mesh, "uvs") and surface_mesh.uvs is not None and hasattr(surface_mesh, "face_uvs_idx") and surface_mesh.face_uvs_idx is not None:
        # mesh_glb: create_render_mesh가 넣은 uvs, face_uvs_idx 그대로 사용
        verts_uvs = surface_mesh.uvs.to(device=device, dtype=torch.float32)
        faces_uvs = surface_mesh.face_uvs_idx.to(device=device).long()
    elif hasattr(surface_mesh, "face_uvs") and surface_mesh.face_uvs is not None:
        # smplx: create_render_mesh의 base_uv (0,0),(1,0),(0,1) → verts_uvs 3개, faces_uvs [0,1,2] 반복
        face_uvs = surface_mesh.face_uvs.to(device=device, dtype=torch.float32)
        F = face_uvs.shape[0]
        verts_uvs = face_uvs[0:1, :, :].squeeze(0)  # (3, 2)
        faces_uvs = torch.arange(3, device=device).long().unsqueeze(0).expand(F, 3)

    if diffuse_map is not None and verts_uvs is not None and faces_uvs is not None:
        # (H, W, 4) -> (1, H, W, 3) RGB
        map_rgb = diffuse_map.to(device=device).float()
        if map_rgb.dim() == 2:
            map_rgb = map_rgb.unsqueeze(-1).expand(*map_rgb.shape, 3)
        elif map_rgb.shape[-1] == 4:
            map_rgb = map_rgb[..., :3]
        if map_rgb.dim() == 3:
            map_rgb = map_rgb.unsqueeze(0)  # (1, H, W, 3)
        if verts_uvs.dim() == 2:
            verts_uvs = verts_uvs.unsqueeze(0)
        if faces_uvs.dim() == 2:
            faces_uvs = faces_uvs.unsqueeze(0)
        try:
            textures = TexturesUV(
                maps=map_rgb,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
            )
        except Exception:
            textures = None
    if textures is None:
        verts_rgb = torch.full((1, V, 3), float(default_rgb), device=device, dtype=torch.float32)
        textures = TexturesVertex(verts_features=verts_rgb)

    verts_normals_list = None
    if use_vertex_normals:
        verts_normals_list = [_compute_vertex_normals(verts, faces, device)]

    return Meshes(verts=[verts], faces=[faces], textures=textures,
                 verts_normals=verts_normals_list)


def build_scene_meshes_from_subject_params(pos, subject_params, device,
                                          render_smpl_only=True, render_bone_only=False,
                                          fix_face_winding=False,
                                          use_vertex_normals=False):
    """
    pos: [N,3] particle positions (torch)
    subject_params: list of dicts (create_render_mesh에서 쓰는 구조)
    fix_face_winding: True면 면 winding 뒤집기 (일부 메시에서만 필요).
    use_vertex_normals: True면 면적 가중 버텍스 법선 사용 (아티팩트 완화).

    복수 메시: for문으로 verts_list, faces_list, TexturesUV(maps, verts_uvs, faces_uvs) 리스트를 채운 뒤
    Meshes(verts=verts_list, faces=faces_list, textures=TexturesUV(...)) 한 번에 생성.
    전부 TexturesUV가 아니면 기존처럼 concat + TexturesVertex fallback.
    """
    mesh_subjects = [sp for sp in subject_params if sp.get("type", "mesh") == "mesh"]
    meshes_list = []
    for sp in mesh_subjects:
        sm = create_render_mesh(
            pos, sp,
            render_smpl_only=render_smpl_only,
            render_bone_only=render_bone_only,
            device=device,
        )
        meshes_list.append(_surface_mesh_to_pytorch3d_mesh(
            sm, device=device,
            default_rgb=0.8,
            fix_face_winding=fix_face_winding,
            use_vertex_normals=use_vertex_normals,
            subject_param=sp,
        ))


    # 복수 메시: 전부 TexturesUV면 리스트로 합쳐 한 번에 Meshes(TexturesUV) 생성 후
    # 한 장면으로 합침 (PyTorch3D는 메시 N개면 이미지 N장 생성 → join_meshes_as_scene으로 1장에 두 명 모두 렌더)
    if all(isinstance(m.textures, TexturesUV) for m in meshes_list):
        verts_list = [m.verts_list()[0] for m in meshes_list]
        # verts_list[0][:, 0] +=  0.01521 * 2
        # verts_list[0][:, 1] +=  0.01401 * 2
        # verts_list[0][:, 2] += -0.00061 * 2
        faces_list = [m.faces_list()[0] for m in meshes_list]
        maps_list = [m.textures.maps_list()[0] for m in meshes_list]
        verts_uvs_list = [m.textures.verts_uvs_list()[0] for m in meshes_list]
        faces_uvs_list = [m.textures.faces_uvs_list()[0] for m in meshes_list]
        textures = TexturesUV(
            maps=maps_list,
            verts_uvs=verts_uvs_list,
            faces_uvs=faces_uvs_list,
        )
        verts_normals_arg = None
        if use_vertex_normals and all(m.has_verts_normals() for m in meshes_list):
            verts_normals_arg = [m.verts_normals_list()[0] for m in meshes_list]
        scene = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=textures,
            verts_normals=verts_normals_arg,
        )
        return join_meshes_as_scene(scene)


# ---------- Main: render multiple meshes together ----------
def render_mesh_pytorch3d(
    viewpoint_camera,            # INRIA Camera
    pos,                         # [N,3] torch, already transformed to render coords
    subject_params,              # list of mesh params
    out_dir,                     # output folder
    frame_idx,                   # int
    device="cuda",
    renderer=None,               # optional prebuilt renderer for speed
    render_smpl_only=True,
    render_bone_only=False,
    debug_timing=False,
):
    """
    Renders multiple meshes together in one scene using PyTorch3D.
    Saves RGB PNG (and alpha PNG).
    Returns: rgb(H,W,3), alpha(H,W,1) on GPU.
    """
    device = torch.device(device)
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    if renderer is None: # not None, pass
        renderer = make_pytorch3d_renderer(device, H, W, bg_rgb=(0.0, 0.0, 0.0), faces_per_pixel=1)

    cameras = inria_camera_to_pytorch3d(viewpoint_camera, device=device)

    if debug_timing:
        torch.cuda.synchronize()
        t0 = time.time()
    scene = build_scene_meshes_from_subject_params(
        pos=pos,
        subject_params=subject_params,
        device=device,
        render_smpl_only=render_smpl_only,
        render_bone_only=render_bone_only,
    )
    if debug_timing:
        torch.cuda.synchronize()
        print(f"  pytorch3d scene build: {time.time() - t0:.3f}s")

    if debug_timing:
        t0 = time.time()
    images = renderer(scene, cameras=cameras)   # [1,H,W,4]
    if debug_timing:
        torch.cuda.synchronize()
        print(f"  pytorch3d rasterize+shade: {time.time() - t0:.3f}s")

    rgba = images[0].clamp(0.0, 1.0)
    # PyTorch3D NDC: +Y up → image row 0 at bottom; flip vertically for standard image coords (row 0 = top)
    rgba = rgba.flip(0)
    rgba = rgba.flip(1)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]

    os.makedirs(out_dir, exist_ok=True)
    if debug_timing:
        t0 = time.time()
    to_pil_image(rgb.permute(2, 0, 1).detach().cpu()).save(
        os.path.join(out_dir, f"{frame_idx:08d}.png")
    )
    if debug_timing:
        print(f"  pytorch3d save PNG: {time.time() - t0:.3f}s")
    # to_pil_image(alpha.permute(2, 0, 1).detach().cpu()).save(
    #     os.path.join(out_dir, f"{frame_idx:08d}_alpha.png")
    # )

    return rgb, alpha