import sys
sys.path.append("gaussian-splatting")
import argparse
import math
import cv2
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import shutil
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
from utils.graphics_utils import focal2fov, BasicPointCloud

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
# from mpm_solver_warp.mpm_solver_warp_hair import MPM_Simulator_WARP_hair
import warp as wp
wp.config.mode = 'debug'
wp.config.verify_cuda = True

# Particle filling dependencies
from particle_filling.filling_new import *

# Utils
from utils.decode_param import *
# from utils.decode_param_hair import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import * # initialize_resterize, convert_SH

# dataset loader
from utils.dataset import *
# from utils.dataset_hair import *
# from utils.human_utils import *
from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes, PBRMaterial
from kaolin.render.camera import Camera
import torchvision.transforms.functional as F


os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
os.environ["TORCH_USE_CUDA_DSA"]= "1"
device = "cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
# os.environ["TI_VISIBLE_DEVICES"] = device.split(":")[1]
# os.environ["TI_ENABLE_CUDA"] = device.split(":")[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["TI_VISIBLE_DEVICES"] = "1"
# os.environ["TI_ENABLE_CUDA"] = "1"

wp.init()
wp.set_device(device)
# wp.set_device("cuda:1")
wp.config.verify_cuda = True

# ti.init(arch=ti.cuda, device_memory_GB=8.0)
ti.init(arch=ti.cuda, device_memory_GB=1.0)

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="ours", choices=["vanila", "ours"])
    args = parser.parse_args()

    # if not os.path.exists(args.model_path):
    #     AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")

    # load scene config
    print("Loading scene config...")
    (
        sim_params, 
        subject_params, # load할 gaussians_model_path, rotation, translation, scale, particle_filling, initial velocity,
                        # (sim_area, density, E, nu, material)
        bc_params,
        time_params,
        camera_params,
        human_params
    ) = decode_param_json(args.config) # './config/ficus_config.json'    
    if args.output_path is not None:
        sim_params["output_path"] = args.output_path
        
    if sim_params["output_path"] is not None and not os.path.exists(sim_params["output_path"]):
        os.makedirs(sim_params["output_path"])
        os.chmod(sim_params["output_path"], 0o777)        
    if sim_params["output_path"] is not None and not os.path.exists(os.path.join(sim_params["output_path"], "3dgrut")):
        os.makedirs(os.path.join(sim_params["output_path"], "3dgrut"))
        os.chmod(os.path.join(sim_params["output_path"], "3dgrut"), 0o777)    
    if sim_params["output_path"] is not None:
        shutil.copy(args.config, os.path.join(sim_params["output_path"], "config.json"))
        copy_files = ["mpm_solver_warp/mpm_solver_warp.py", "mpm_solver_warp/warp_utils.py", "mpm_solver_warp/mpm_utils.py",
                      "mpm_solver_warp/mpm_human_utils.py", "utils/decode_param.py", "utils/dataset.py", "utils/render_utils.py",
                      "AnimatableGaussians/main_avatar_phys.py", "simulation.py"]
        for file in copy_files:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(sim_params["output_path"], file.split("/")[-1]))

    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        if sim_params["white_bg"]
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    )
    
    # load gaussians
    print("Loading gaussians...")
    # TODO : load human from T-pose or A-pose, it is human vertices from 1st frame now
    # subject_params : 가우시안 모델, human_sequences : canonical, lbs, joint_mats
    gaussian_subjects, human_sequences = load_gaussian_subjects(subject_params, sim_params, device=device) # load gaussians of canonical human & object
    # human_sequences   = load_human_sequences(subject_params, human_params, device=device) # load human vertices frames
    
    # init the scene
    # list format of each gaussiansubjects
    print("Initializing scene and pre-processing...")
    subjects, subject_params = load_subjects(gaussian_subjects, subject_params, sim_params, pipeline, device=device)    
    # rotate and translate object
    subjects, subject_params = rotation_subjects(subjects, subject_params, sim_params) # rotate each scenes
        
    # shift to each center & scale to each size
    # modify covariance matrix accordingly
    subjects, subject_params = shift_scale_subjects(subjects, subject_params, sim_params) # shift each scenes        

    # bc_params = set_human_sequences_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params) # add smpl velocity to bc_params
    bc_params = set_human_model_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params) # modify posed gaussian via net

    # fill particles if needed, 
    subjects, init_gs_nums = fill_particles_subjects(subjects, subject_params, sim_params, device=device)
    
    # mpm volume
    mpm_vol = get_particle_volume_from_subjects(subjects, subject_params, sim_params, device=device)
    
    # merge all subject gaussians
    mpm_params = merge_subjects(subjects, init_gs_nums, pipeline, device=device)
    mpm_params["vol"] = mpm_vol
    # mpm_gs_num = mpm_params["pos"].shape[0]

    
    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch( # particle_selection, init velocity, g, E, nu, material, dencity 등등
        mpm_params["pos"],
        mpm_params["vol"], # pos, n_grid
        mpm_params["index"], # particle_id
        mpm_params["cov"], # [N, 6], particle_cov
        n_grid=sim_params["n_grid"],
        grid_lim=sim_params["grid_lim"],
        n_subjects=len(subjects),
        n_humans=sum(1 for item in human_sequences if item is not None) # Num of humans
    )

    mpm_solver.set_simulater_parameters(sim_params, args.method) # MPM space gravity
    mpm_solver.set_subjects_parameters(subject_params) # set up material parameters, subject gravity, density, object init velocity
    # mpm_solver.set_parameters_dict(material_params)
    # mpm_solver.mpm_state.particle_x.numpy().shape              # (341002, 3)
    # mpm_solver.mpm_state.particle_selection.numpy().shape      # (341002,)
    # np.unique(mpm_solver.mpm_state.particle_selection.numpy()) # 0    

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params, device) # human init velocity

    mpm_solver.finalize_mu_lam()


    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"], device=device).reshape((1, 3)) #.cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"], device=device)
        .reshape((1, 3))
        #.cuda()
    )
    
    # rotation matrix of 1st subject
    degrees = get_rotation_degree(subject_params)
    axes    = get_rotation_axis(subject_params)    
    rotation_matrices = generate_rotation_matrices(
        degrees,
        axes
    )
    
    # rotation_matrix = rotation_matrices[1]
    # scale_origin = subject_params[0]["scale"]
    # center = subject_params[0]["center"]
    # ori_mean = subject_params[0]["ori_mean"]
    
    # rotation_matrix = rotation_matrices[1][None]
    scale_origin = subject_params[0]["scale"]
    ori_mean = subject_params[0]["ori_mean"]
    center = subject_params[0]["center"]
        
    (
        viewpoint_center_worldspace,     # array([-2.2237003e-04,  9.0204656e-01,  2.4339089e-02], dtype=float32)
        observant_coordinates,           # array([[ 7.07e-01, -7.07e-01,  0.0e+00], [ 7.07e-01,  7.07e-01,  6.60e-08], [ 4.21e-08,  4.67e-08, -1.0e+00]])
    ) = get_center_view_worldspace_and_observant_coordinate_arbitrary_center(
        mpm_space_viewpoint_center,      # tensor([[1, 1, 1]], device='cuda:0')
        mpm_space_vertical_upward_axis,  # tensor([[0, 1, 0]], device='cuda:0')
        rotation_matrices,               # tensor([[ 1.0,  0.0,  0.0], [ 0.0,  0.0, -1.0], [ 0.0,  1.0,  0.0]], device='cuda:0')
        scale_origin,                    # 0.5543
        ori_mean,               # tensor([-2.2237e-04, -2.4339e-02,  9.0205e-01])
        center,
    )
    
    # run the simulation
    if sim_params["output_ply"] or sim_params["output_h5"]:
        directory_to_save = os.path.join(sim_params["output_path"], "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=sim_params["output_ply"],
            save_to_h5=sim_params["output_h5"],
        )

    substep_dt = time_params["substep_dt"]
    smplx_dt = time_params["smplx_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = mpm_params["opacity"]
    # shs_render = mpm_params["shs"]
    height = None
    width = None
    
    # from types import SimpleNamespace
    # state = SimpleNamespace()
    
    gs_object = "./data/render/gs_seed/point_cloud.ply"  # any 3DGS ply; only used to boot the 3DGRUT engine
    gs_object2 = None
    mesh_assets_folder = "./data/render/mesh_assets"
    default_config = "apps/colmap_3dgrt.yaml"
    default_config2 = "./3dgrut/configs/apps/colmap_3dgrt.yaml"
        
    # 3dgrut renderer
    engine = Engine3DGRUT(
        gs_object=gs_object,
        mesh_assets_folder=mesh_assets_folder,
        default_config=default_config
    )
    engine.force_white_bg = True
    
    # Remove initial glass sphere from scene
    for mesh_name in list(engine.primitives.objects.keys()):
        engine.primitives.remove_primitive(mesh_name)
        
    # init gaussian index, mesh index, mesh face
    is_mesh = False
    is_gaussian = False
    is_3d_measure = False
    gs_index = []
    for subject_param in subject_params:
        if subject_param['type'] == 'gaussian':
            gs_index.append((subject_param['particle_start'], subject_param['particle_end']))
            is_gaussian = True
        elif subject_param['type'] == 'mesh':
            if "render_material" in subject_param:
                engine.primitives.registered_materials[subject_param["render_material_name"]] = subject_param["render_material"]
            is_mesh = True
        else:
            assert "Not mesh or gaussian"
    ranges = [torch.arange(start, end) for start, end in gs_index]
    gs_index = torch.cat(ranges) if ranges else torch.tensor([0], dtype=torch.long)
    
    for frame in tqdm(range(frame_num), leave=True, ncols=60): # time sequence
        current_camera = get_camera_view(
            "./data/render/",  # directory containing cameras.json
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace, # view target point
            observant_coordinates=observant_coordinates, # camera의 기준 좌표축
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )

        ##################################
        # f = open(os.path.join(sim_params["output_path"], "kabsch_log.txt"), "a")
        f = None
        for step in range(step_per_frame):            
            mpm_solver.p2g2p(frame, step, substep_dt, mpm_params, device=device, smplx_dt=smplx_dt, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params)
            # mpm_solver.p2g2p_base(frame, step, substep_dt, mpm_params, device=device, smplx_dt=smplx_dt, is_3d_measure=is_3d_measure)
            
            # if step == 0 and frame == 0:
            # if frame == 0:
            # if step == 0:
            if 0:
                mpm_solver.human_modify_model[0].human_n_particles
                init_gs_num = sum(init_gs_nums)
                pos = mpm_solver.export_particle_x_to_torch()[:init_gs_num].to(device) # mpm_state.particle_x
                # cov3D = mpm_solver.export_particle_cov_to_torch()                      # compute_cov_from_F, gaussian render에 필요한 cov를 갱신, F안에 R, S가 내포됨, 따로 S를 추출해서 갱신하지 않음
                rot = mpm_solver.export_particle_R_to_torch()                          # compute_R_from_F, F = R*S, not use particle_R
                # cov3D = cov3D.view(-1, 6)[:init_gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:init_gs_num].to(device)

                pos = apply_inverse_rotations(
                    undotransform2center(pos, ori_mean, scale_origin, center),
                    rotation_matrices,
                    )
                shs = mpm_params["shs"][:init_gs_num]
                
                colors_precomp = convert_SH(shs, current_camera, gaussian_subjects[0], pos, rot)
                
                os.makedirs(os.path.join(sim_params["output_path"], "pc"), exist_ok=True)
                pc = trimesh.Trimesh(vertices=pos.detach().cpu().numpy(),
                                    vertex_colors=colors_precomp.detach().cpu().numpy())
                pc.export(os.path.join(sim_params["output_path"], "pc", f"{frame:04d}_{step:04d}.ply"))
        
        ##################################       
        
        rasterize = initialize_resterize(
            current_camera, gaussian_subjects[0], pipeline, background
        )

        if sim_params["output_ply"] or sim_params["output_h5"]:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=sim_params["output_ply"],
                save_to_h5=sim_params["output_h5"],
            )
         
        if sim_params["output_ply"] and is_mesh:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=sim_params["output_ply"],
                save_to_h5=sim_params["output_h5"],
            )

        # rotation 개별, 합한 gaussian 따로 하나 생성
        init_gs_num = sum(init_gs_nums)
        if sim_params["render_img"]:
            pos = mpm_solver.export_particle_x_to_torch()[:init_gs_num].to(device) # mpm_state.particle_x            
            rot = mpm_solver.export_particle_R_to_torch()                          # compute_R_from_F, F = R*S, not use particle_R            
            rot = rot.view(-1, 3, 3)[:init_gs_num].to(device)            
            pos = apply_inverse_rotations(
                undotransform2center(pos, ori_mean, scale_origin, center),
                rotation_matrices,
                )            
            opacity = opacity_render[:init_gs_num]
            shs = mpm_params["shs"][:init_gs_num]
                        
            # if is_mesh: # mesh 3dgrut render, mesh가 없으면 사용 안함
                        
            if 1: # mesh 3dgrut render, mesh가 없으면 사용 안함
                _, scale, rot_ = mpm_solver.export_particle_quat_scale_to_torch()
                if (rot - rot_).abs().max() > 1e-7 :
                    print("rot is wrong : {0:06f}".format((rot - rot_).abs().max()))
                rot_ = apply_inverse_rot_rotations(rot_, rotation_matrices)
                quat = rotmat_to_quat(rot_, ordering="wxyz", orthonormalize=True)
                scale = scale / scale_origin
                
                end = time.time()                
                gaussians_3dgrut = engine.scene_mog
                gaussians_3dgrut.positions          = torch.nn.Parameter(pos[gs_index]) # [N, 3]
                gaussians_3dgrut.features_albedo    = torch.nn.Parameter(shs[gs_index][:, :1].view(-1, 3)) # [N, 3]
                gaussians_3dgrut.features_specular  = torch.nn.Parameter(shs[gs_index][:, 1:].view(-1, 45)) # [N, 45]
                gaussians_3dgrut.density            = torch.nn.Parameter(gaussians_3dgrut.density_activation_inv(opacity[gs_index])) # [N, 1]
                gaussians_3dgrut.scale              = torch.nn.Parameter(gaussians_3dgrut.scale_activation_inv(scale[gs_index])) # [N, 3]
                gaussians_3dgrut.rotation           = torch.nn.Parameter(quat[gs_index]) # [N, 4]
                if is_gaussian==False:
                    gaussians_3dgrut.density =  torch.nn.Parameter(gaussians_3dgrut.density - 10.0)
                gaussians_3dgrut.validate_fields()
                engine.rebuild_bvh(gaussians_3dgrut)                
                engine.primitives.mesh_autoscale_func = my_autoscale_no_unit
                
                for mesh_name in list(engine.primitives.objects.keys()):
                    engine.primitives.remove_primitive(mesh_name)
                
                for subject_param in subject_params:
                    if subject_param['type'] == 'mesh':
                        mesh = create_render_mesh(pos, subject_param, render_smpl_only=True, render_bone_only=False, device=device)
                        # mesh = create_render_mesh(pos, subject_param, render_smpl_only=False, render_bone_only=True, device=device)                        
                        geometry_type = 'rendermesh_' + str(subject_param['index'])
                        engine.primitives.PROCEDURAL_SHAPES[geometry_type] = lambda *args, **kwargs: mesh
                        engine.primitives.add_primitive(
                            geometry_type=geometry_type,
                            primitive_type=OptixPrimitiveTypes.DIFFUSE,
                            device='cuda'
                        )
                # engine.primitives.objects['rendermesh_0 1'].vertices
                
                # create arbitary mesh
                engine.primitives.PROCEDURAL_SHAPES['testmesh1'] = create_init_mesh
                engine.primitives.add_primitive(
                    geometry_type='testmesh1',
                    primitive_type=OptixPrimitiveTypes.DIFFUSE,
                    device='cuda'
                )               
                
                engine.invalidate_materials_on_gpu()
                engine.primitives.rebuild_bvh_if_needed(True, True)
                
                view_matrix, FoVy = get_lookat_camera(current_camera)                
                camera = Camera.from_args(
                                view_matrix=view_matrix,
                                fov=FoVy,
                                width=current_camera.image_width, height=current_camera.image_height,
                        )                
                framebuffer = engine.render(camera)
                rgba_buffer = torch.cat([framebuffer['rgb'], framebuffer['opacity']], dim=-1)
                rgba_buffer = torch.clamp(rgba_buffer, 0.0, 1.0)
                chw_buffer = rgba_buffer[0].permute(2, 0, 1)
                img = F.to_pil_image(chw_buffer)
                img.save(os.path.join(sim_params["output_path"], "3dgrut", f"{frame}.png".rjust(8, "0")))
                # img.save('./test_results/test2.png')
                # print('./test_results/test2.png')
                # print(time.time()-end)
                
            # if is_gaussian:
            if 1:
                end = time.time()
                _, scale, rot_ = mpm_solver.export_particle_quat_scale_to_torch()
                if (rot - rot_).abs().max() > 1e-7 :
                    print("rot is wrong : {0:06f}".format((rot - rot_).abs().max()))
                rot_ = apply_inverse_rot_rotations(rot_, rotation_matrices)
                quat = rotmat_to_quat(rot_, ordering="wxyz", orthonormalize=True)
                scale = scale / scale_origin
                rendering, raddi = rasterize(
                    means3D=pos,
                    means2D=mpm_params["screen_points"],
                    shs=shs, # shs
                    colors_precomp=None, # rotation, export_particle_R_to_torch, compute_R_from_F
                    opacities=opacity,
                    scales=scale, # [N, 3]
                    rotations=quat, # [N, 4]
                    cov3D_precomp=None, # export_particle_cov_to_torch, compute_cov_from_F
                )
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                assert sim_params["output_path"] is not None
                cv2.imwrite(
                    os.path.join(sim_params["output_path"], f"{frame}.png".rjust(8, "0")),
                    255 * cv2_img,
                )
                # cv2.imwrite('./test_results/test3.png', 255 * cv2_img)
                # print(time.time()-end)            

    # ffmpeg -framerate 50 -i ./%04d.png -c:v libx264 -s 1022x746 -y -pix_fmt yuv420p ./output.mp4
    # ffmpeg -framerate 50 -i ./%04d.png -c:v libx264 -s 2800x2000 -y -pix_fmt yuv420p ./output.mp4
    # ffmpeg -framerate 25 -i ./%04d.png -frames:v 106 -c:v libx264 -s 2800x2000 -y -pix_fmt yuv420p ./output_.mp4
    if sim_params["render_img"] and sim_params["compile_video"]:
        if height is None or width is None:
            if is_gaussian:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            elif is_mesh:
                height = img.size[1] // 2 * 2
                width = img.size[0] // 2 * 2
        fps = int(1.0 / time_params["frame_dt"]) // 2
        output_path = sim_params["output_path"]
        os.system(
            f"ffmpeg -framerate {fps} -i {output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {output_path}/output.mp4"
        )
        # if is_mesh:
        if 1:
            output_path = os.path.join(sim_params["output_path"], "3dgrut")
            os.system(
                f"ffmpeg -framerate {fps} -i {output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {output_path}/output.mp4"
            )
