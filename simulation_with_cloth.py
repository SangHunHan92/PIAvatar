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
from mpm_solver_warp.mpm_solver_warp_separable_contact import MPM_Simulator_WARP
# from mpm_solver_warp.mpm_solver_warp_hair import MPM_Simulator_WARP_hair
# heterogeneous body+cloth: subclass that adds MPMAvatar-style cloth on top of user's solver
from hetero_cloth.cloth_solver import MPM_Simulator_WARP_with_Cloth
from hetero_cloth.cloth_init import prepare_cloth_template, assemble_cloth_particle_arrays
from hetero_cloth.cloth_integration import (
    load_cloth_obj,
    load_a1_cloth_subset,
    pose_a1_cloth_with_body,
    get_body_mesh_at_frame,
    extend_mpm_params_with_cloth,
    apply_cloth_material,
    apply_cloth_pin_top_no_gravity,
    setup_cloth_lbs_pin,
)
import warp as wp
wp.config.mode = 'debug'
wp.config.verify_cuda = True

# Particle filling dependencies
from particle_filling.filling_new import *

# Utils
from utils.decode_param_separable_contact import *
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
        copy_files = ["mpm_solver_warp/mpm_solver_warp_separable_contact.py", "mpm_solver_warp/warp_utils_separable_contact.py",
                      "mpm_solver_warp/mpm_utils_separable_contact.py", "mpm_solver_warp/mpm_human_utils_separable_contact.py",
                      "utils/decode_param_separable_contact.py", "utils/dataset.py", "utils/render_utils.py", "AnimatableGaussians/main_avatar_phys.py",
                      "simulation_with_cloth.py", "hetero_cloth/cloth_solver.py", "hetero_cloth/cloth_kernels.py"]
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

    # fill particles BEFORE rotation/shift_scale and BEFORE set_human_model_to_boundary_conditions:
    #   - subject['pos'] here is still in the gaussian's first-frame-posed frame, 1:1 with human_seq['pos_pts']
    #   - extend_lbs_for_filled_particles needs that frame to do k-NN against pos_pts/cano_pts
    #   - set_human_model_to_boundary_conditions reads human_seq['cano_pts'] / smplx_model.lbs_weights;
    #     extending those before that call propagates the extended human_n_particles count downstream
    from particle_filling.filling_warp import (
        fill_particles_subjects_warp,
        get_particle_volume_from_subjects_warp,
    )
    from particle_filling.lbs_extend import extend_lbs_for_filled_particles
    from particle_filling.thin_shell import thin_shell_subjects
    # Dispatch: if any subject's particle_filling has shell_mode="thin", use the
    # thin-shell variant (1:1 inner layer, LBS inherited from surface). Otherwise
    # fall back to the volumetric Warp filling. Mixing is not supported.
    _use_thin_shell = any(
        isinstance(p.get("particle_filling"), dict)
        and str(p["particle_filling"].get("shell_mode", "")).lower() == "thin"
        for p in subject_params
    )
    if _use_thin_shell:
        print("[filling] using thin-shell mode (1:1 inner layer)")
        subjects, init_gs_nums = thin_shell_subjects(
            subjects, subject_params, sim_params, device=device,
            human_sequences=human_sequences,
        )
    else:
        subjects, init_gs_nums = fill_particles_subjects_warp(
            subjects, subject_params, sim_params, device=device,
            human_sequences=human_sequences,
        )
    subjects, subject_params, human_sequences = extend_lbs_for_filled_particles(
        subjects, subject_params, human_sequences, init_gs_nums,
        k=4, device=device,
    )

    # rotate and translate object
    subjects, subject_params = rotation_subjects(subjects, subject_params, sim_params) # rotate each scenes

    # shift to each center & scale to each size
    # modify covariance matrix accordingly
    subjects, subject_params = shift_scale_subjects(subjects, subject_params, sim_params) # shift each scenes

    # bc_params = set_human_sequences_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params) # add smpl velocity to bc_params
    bc_params = set_human_model_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params) # modify posed gaussian via net
    
    # mpm volume — passes init_gs_nums to enforce merge_subjects ordering
    # (without it, multi-subject + filling cases silently misalign vol↔pos)
    mpm_vol = get_particle_volume_from_subjects_warp(
        subjects, subject_params, sim_params, init_gs_nums=init_gs_nums, device=device,
    )

    # Optional: scale interior (filled) particles' volume → reduces their MPM mass
    # (since mass = density · vol). Tests whether heavy interior particles accelerate
    # the kabsch-drift / LBS-feedback cycle by overcoupling bones via grid.
    interior_mass_scale = sim_params.get("interior_mass_scale", 1.0)
    if interior_mass_scale != 1.0:
        n_surface_total = sum(init_gs_nums)
        n_interior_total = mpm_vol.shape[0] - n_surface_total
        if n_interior_total > 0:
            mpm_vol[n_surface_total:] = mpm_vol[n_surface_total:] * float(interior_mass_scale)
            print(f"[interior_mass_scale={interior_mass_scale}] scaled vol of {n_interior_total} interior particles")

    # Optional: scale BONE particles' volume → makes bones much heavier than interior,
    # so interior MPM coupling can't drift bones. With kabsch (smplx_direct_A=false),
    # this stabilizes the kabsch fit while preserving paper's interaction-driven pose
    # contribution AND allowing interior_mass_scale=1.0 (full dent response).
    # In merge_subjects order, each human subject's particles start with bone particles.
    bone_mass_scale = sim_params.get("bone_mass_scale", 1.0)
    if bone_mass_scale != 1.0:
        cumul = 0
        for i, (subject, hs) in enumerate(zip(subjects, human_sequences)):
            n_surf = init_gs_nums[i]
            if hs is not None and "bone_cano" in hs:
                bone_n = hs["bone_cano"].shape[0]
                if cumul + bone_n <= mpm_vol.shape[0]:
                    mpm_vol[cumul : cumul + bone_n] = mpm_vol[cumul : cumul + bone_n] * float(bone_mass_scale)
                    print(f"[bone_mass_scale={bone_mass_scale}] subject {i}: scaled vol of {bone_n} bone particles")
            cumul += n_surf
    
    # merge all subject gaussians
    mpm_params = merge_subjects(subjects, init_gs_nums, pipeline, device=device)
    mpm_params["vol"] = mpm_vol
    # mpm_gs_num = mpm_params["pos"].shape[0]

    # ============================================================
    # CLOTH INTEGRATION (heterogeneous body+cloth experiment)
    # ============================================================
    # Configurable via sim_params["cloth"]; if absent, runs as pure body simulation
    # (and behaves identically to simulation_separable_contact.py thanks to the
    # has_cloth=False guard in MPM_Simulator_WARP_with_Cloth.p2g2p).
    cloth_cfg = sim_params.get("cloth", None)
    cloth_prep = None
    if cloth_cfg is not None:
        n_existing = mpm_params["pos"].shape[0]

        # Two cloth-loading paths:
        #   1) source="a1_subset": use cloth subset (22424 verts) from
        #      a1_canonical_assets.npz, in UNSCALED SMPL-X coord at pose460
        #      (aligned with load_smplx_a1's body coord).
        #   2) default: legacy load_cloth_obj on cloth_cfg["obj_path"].
        cloth_source = cloth_cfg.get("source", "obj")
        if cloth_source == "a1_subset":
            # Pose cloth canonical to body's first-frame pose using body's OWN
            # SMPL-X model + pose_dataset (R_stand-composed). Guarantees cloth
            # ends up in same world-coord frame as body (no SMPL-X impl mismatch).
            body_hs = human_sequences[0]
            cloth_v, cloth_f = pose_a1_cloth_with_body(
                canonical_npz=cloth_cfg.get(
                    "canonical_npz",
                    "third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz",
                ),
                pose_dataset=body_hs["pose_dataset"],
                device=device,
                outward_offset=float(cloth_cfg.get("outward_offset", 0.0)),
            )
            print(f"[cloth] a1_subset: {cloth_v.shape[0]} verts, {cloth_f.shape[0]} faces (unscaled SMPL-X coord)")
        else:
            cloth_obj_path = cloth_cfg["obj_path"]
            cloth_v, cloth_f = load_cloth_obj(
                cloth_obj_path,
                scale=1.0, offset=(0.0, 0.0, 0.0),
                rotation_deg=0.0, rotation_axis=0,
                device=device,
            )
            print(f"[cloth] loaded {cloth_v.shape[0]} verts, {cloth_f.shape[0]} faces from {cloth_obj_path}")

        # Apply body subject's transform chain so cloth ends up in SAME sim coord as body.
        # Use body subject (assumed at index 0) for transform reference.
        body_param = subject_params[0]
        body_rot_mats = generate_rotation_matrices(
            [torch.tensor(body_param["rotation_degree"])],
            [torch.tensor(body_param["rotation_axis"])],
        )
        cloth_v = apply_rotations(cloth_v, body_rot_mats)
        ori_mean = body_param["ori_mean"].to(cloth_v.device)
        body_center = torch.tensor(body_param["center"], device=cloth_v.device, dtype=cloth_v.dtype)
        cloth_v = (cloth_v - ori_mean) * float(body_param["scale"]) + body_center
        # Optional fine-tune offset on top (default zeros)
        ft_offset = cloth_cfg.get("offset", [0.0, 0.0, 0.0])
        cloth_v = cloth_v + torch.tensor(ft_offset, device=cloth_v.device, dtype=cloth_v.dtype)
        print(f"[cloth] aligned with body subject's transform "
              f"(rot_deg={body_param['rotation_degree']}, scale={body_param['scale']}, "
              f"center={body_param['center']}, fine_offset={ft_offset})")
        print(f"[cloth] final bbox: min={cloth_v.min(0).values.tolist()}, "
              f"max={cloth_v.max(0).values.tolist()}")

        cloth_prep = prepare_cloth_template(
            cloth_v, cloth_f, n_existing=n_existing,
            thickness=float(cloth_cfg.get("thickness", 1e-3)),
            density=float(cloth_cfg.get("density", 1.0)),
            gamma=float(cloth_cfg.get("gamma", 1e3)),
            kappa=float(cloth_cfg.get("kappa", 1e5)),
            friction_angle_deg=float(cloth_cfg.get("friction_angle_deg", 20.0)),
        )
        cloth_subject_id = len(subjects)  # cloth gets its own subject id at end
        cloth_x, cloth_v_, cloth_vol_, cloth_mass_ = assemble_cloth_particle_arrays(cloth_prep)
        extend_mpm_params_with_cloth(
            mpm_params, cloth_x, cloth_v_, cloth_vol_, cloth_subject_id,
        )
        print(f"[cloth] appended {cloth_prep.n_elements} elements + {cloth_prep.n_vertices} vertices "
              f"as subject_id={cloth_subject_id}; total particles = {mpm_params['pos'].shape[0]}")

    # set up the mpm solver — cloth-aware subclass; behaves identically when cloth not configured
    n_subjects_total = len(subjects) + (1 if cloth_prep is not None else 0)
    mpm_solver = MPM_Simulator_WARP_with_Cloth(10)
    mpm_solver.load_initial_data_from_torch( # particle_selection, init velocity, g, E, nu, material, dencity 등등
        mpm_params["pos"],
        mpm_params["vol"], # pos, n_grid
        mpm_params["index"], # particle_id
        mpm_params["cov"], # [N, 6], particle_cov
        n_grid=sim_params["n_grid"],
        grid_lim=sim_params["grid_lim"],
        n_subjects=n_subjects_total,
        n_humans=sum(1 for item in human_sequences if item is not None) # Num of humans
    )

    mpm_solver.set_simulater_parameters(sim_params, args.method) # MPM space gravity
    if "rpic_damping" in sim_params:
        mpm_solver.mpm_model.rpic_damping = float(sim_params["rpic_damping"])
        print(f"[sim] rpic_damping = {mpm_solver.mpm_model.rpic_damping}")
    mpm_solver.set_subjects_parameters(subject_params) # set up material parameters, subject gravity, density, object init velocity

    # cloth's per-particle E/nu/density/gravity (set_subjects_parameters didn't touch cloth slots)
    if cloth_prep is not None:
        apply_cloth_material(
            mpm_solver,
            n_existing=cloth_prep.n_existing,
            n_elements=cloth_prep.n_elements,
            n_vertices=cloth_prep.n_vertices,
            E=float(cloth_cfg.get("E", 5e4)),
            nu=float(cloth_cfg.get("nu", 0.3)),
            density=float(cloth_cfg.get("density", 1.0)),
            gravity=tuple(cloth_cfg.get("gravity", [0.0, -9.8, 0.0])),
            elements_have_mass=False,
            device=device,
        )
        # optional: pin top portion of cloth to body (zero gravity for top verts)
        pin_top_pct = float(cloth_cfg.get("pin_top_pct", 0.0))
        if pin_top_pct > 0.0:
            apply_cloth_pin_top_no_gravity(
                mpm_solver,
                canonical_npz=cloth_cfg.get(
                    "canonical_npz",
                    "third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz",
                ),
                cloth_prep=cloth_prep,
                pin_top_pct=pin_top_pct,
                device=device,
            )
    # mpm_solver.set_parameters_dict(material_params)
    # mpm_solver.mpm_state.particle_x.numpy().shape              # (341002, 3)
    # mpm_solver.mpm_state.particle_selection.numpy().shape      # (341002,)
    # np.unique(mpm_solver.mpm_state.particle_selection.numpy()) # 0    

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params, device) # human init velocity

    mpm_solver.finalize_mu_lam()

    # register cloth state (allocates ClothStateStruct, populates cloth-specific fields)
    if cloth_prep is not None:
        mpm_solver.register_cloth(cloth_prep, device=device)
        print(f"[cloth] registered. has_cloth = {mpm_solver.has_cloth}")

        # LBS pin: top cloth verts kinematically follow body each substep
        lbs_pin_pct = float(cloth_cfg.get("lbs_pin_top_pct", 0.0))
        if lbs_pin_pct > 0.0:
            n_pin, pinned_local, cloth_v_can_np, cloth_lbs_w_np = setup_cloth_lbs_pin(
                mpm_solver,
                mpm_solver.cloth_state,
                canonical_npz=cloth_cfg.get(
                    "canonical_npz",
                    "third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz",
                ),
                pin_top_pct=lbs_pin_pct,
                device=device,
            )
            if n_pin > 0:
                body_param = subject_params[0]
                body_rot_mats_pin = generate_rotation_matrices(
                    [torch.tensor(body_param["rotation_degree"])],
                    [torch.tensor(body_param["rotation_axis"])],
                )
                mpm_solver.cloth_pin_context = {
                    "pose_dataset": human_sequences[0]["pose_dataset"],
                    "cloth_v_can_np": cloth_v_can_np,
                    "cloth_lbs_w_np": cloth_lbs_w_np,
                    "pinned_local_np": pinned_local,
                    "body_rot_mats": body_rot_mats_pin,
                    "body_scale": float(body_param["scale"]),
                }

        # ---------- OPTIONAL: body-mesh collider for cloth ----------
        # Adds a kinematic SMPL-X mesh as an extra collider that prevents
        # body MPM from leaking through cloth (separable contact alone is
        # no-tension only, see body_mesh_collider.py docstring).
        # Opt-in via cloth.use_body_mesh_collider in the scenario config.
        if cloth_cfg.get("use_body_mesh_collider", False):
            body_param_bmc = subject_params[0]
            body_rot_mats_bmc = generate_rotation_matrices(
                [torch.tensor(body_param_bmc["rotation_degree"])],
                [torch.tensor(body_param_bmc["rotation_axis"])],
            )
            body_ori_mean_bmc = body_param_bmc["ori_mean"]
            body_center_bmc = torch.tensor(
                body_param_bmc["center"], device=device, dtype=torch.float32
            )
            body_scale_bmc = float(body_param_bmc["scale"])

            # initial-frame body mesh in world coord
            verts_init, faces_init = get_body_mesh_at_frame(
                pose_dataset=human_sequences[0]["pose_dataset"],
                frame_idx_in_pose_list=0,
                body_rot_mats=body_rot_mats_bmc,
                body_ori_mean=body_ori_mean_bmc,
                body_scale=body_scale_bmc,
                body_center=body_center_bmc,
                device=device,
            )
            mpm_solver.register_body_mesh_collider(
                verts=verts_init,
                faces=faces_init,
                cloth_subj_id=cloth_subject_id,
                friction=float(cloth_cfg.get("body_mesh_friction", 0.1)),
                device=device,
            )
            # stash the bits we need each frame to refresh the mesh's pose
            mpm_solver.body_mesh_collider_ctx = {
                "pose_dataset": human_sequences[0]["pose_dataset"],
                "body_rot_mats": body_rot_mats_bmc,
                "body_ori_mean": body_ori_mean_bmc,
                "body_scale": body_scale_bmc,
                "body_center": body_center_bmc,
            }
            print(f"[body_mesh_collider] enabled. {verts_init.shape[0]} verts, "
                  f"{faces_init.shape[0]} faces, friction="
                  f"{float(cloth_cfg.get('body_mesh_friction', 0.1)):.2f}")

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
        # OPTIONAL: refresh body-mesh collider with this frame's kinematic
        # SMPL-X verts + velocity (verts_next - verts_now)/frame_dt. The
        # mesh stays static within a frame; per-substep collide reads its
        # state from the per-grid scratch arrays (re-scattered each substep).
        if getattr(mpm_solver, "body_mesh_collider", None) is not None:
            ctx = mpm_solver.body_mesh_collider_ctx
            verts_now, _ = get_body_mesh_at_frame(
                pose_dataset=ctx["pose_dataset"],
                frame_idx_in_pose_list=frame,
                body_rot_mats=ctx["body_rot_mats"],
                body_ori_mean=ctx["body_ori_mean"],
                body_scale=ctx["body_scale"],
                body_center=ctx["body_center"],
                device=device,
            )
            verts_next, _ = get_body_mesh_at_frame(
                pose_dataset=ctx["pose_dataset"],
                frame_idx_in_pose_list=frame + 1,
                body_rot_mats=ctx["body_rot_mats"],
                body_ori_mean=ctx["body_ori_mean"],
                body_scale=ctx["body_scale"],
                body_center=ctx["body_center"],
                device=device,
            )
            mpm_solver.body_mesh_collider.update_pose(verts_now, verts_next, frame_dt)

        # f = open(os.path.join(sim_params["output_path"], "kabsch_log.txt"), "a")
        f = None
        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, step, substep_dt, mpm_params, device=device, smplx_dt=smplx_dt, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params)

        # cloth PLY dump per frame (vertex positions only — for visualization or post-processing)
        if cloth_prep is not None:
            cloth_dir = os.path.join(sim_params["output_path"], "cloth_ply")
            os.makedirs(cloth_dir, exist_ok=True)
            x_all = mpm_solver.mpm_state.particle_x.numpy()
            v_off = cloth_prep.n_existing + cloth_prep.n_elements
            cloth_xyz = x_all[v_off : v_off + cloth_prep.n_vertices]
            ply_path = os.path.join(cloth_dir, f"cloth_{frame:04d}.ply")
            with open(ply_path, "w") as fh:
                fh.write("ply\nformat ascii 1.0\n")
                fh.write(f"element vertex {cloth_xyz.shape[0]}\n")
                fh.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                for v in cloth_xyz:
                    fh.write(f"{v[0]} {v[1]} {v[2]}\n")
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
                        
            if sim_params.get("render_3dgrut", False): # mesh 3dgrut render, mesh가 없으면 사용 안함
                _, scale, rot_ = mpm_solver.export_particle_quat_scale_to_torch()
                # slice to surface-only particles to match `rot`/`pos` already sliced
                # (interior particles from particle_filling sit at the tail and are not rendered)
                rot_ = rot_[:init_gs_num]
                scale = scale[:init_gs_num]
                if (rot - rot_).abs().max() > 1e-7 :
                    print("rot is wrong : {0:06f}".format((rot - rot_).abs().max()))
                rot_ = apply_inverse_rot_rotations(rot_, rotation_matrices)
                quat = rotmat_to_quat(rot_, ordering="wxyz", orthonormalize=True)
                scale = scale / scale_origin
                
                end = time.time()
                # 3dgrut gaussians: body+ball only (cloth rendered as mesh primitive below).
                gs_pos     = pos[gs_index]
                gs_alb     = shs[gs_index][:, :1].view(-1, 3)
                gs_spec    = shs[gs_index][:, 1:].view(-1, 45)
                gs_opacity = opacity[gs_index]
                gs_scale   = scale[gs_index]
                gs_rot     = quat[gs_index]
                gaussians_3dgrut = engine.scene_mog
                gaussians_3dgrut.positions          = torch.nn.Parameter(gs_pos)
                gaussians_3dgrut.features_albedo    = torch.nn.Parameter(gs_alb)
                gaussians_3dgrut.features_specular  = torch.nn.Parameter(gs_spec)
                gaussians_3dgrut.density            = torch.nn.Parameter(gaussians_3dgrut.density_activation_inv(gs_opacity))
                gaussians_3dgrut.scale              = torch.nn.Parameter(gaussians_3dgrut.scale_activation_inv(gs_scale))
                gaussians_3dgrut.rotation           = torch.nn.Parameter(gs_rot)
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

                # cloth as 3dgrut MESH primitive (raytraced surface, perfect alignment
                # with body mesh). Color uses MPMAvatar's average cloth RGB.
                if cloth_prep is not None and sim_params.get("render_cloth", True):
                    from kaolin.rep import SurfaceMesh
                    n_e_loc = cloth_prep.n_elements
                    n_v_loc = cloth_prep.n_vertices
                    cv_start = cloth_prep.n_existing + n_e_loc
                    cv_end   = cloth_prep.n_existing + n_e_loc + n_v_loc
                    cv_world = mpm_solver.export_particle_x_to_torch()[cv_start:cv_end].clone()
                    cv_render = apply_inverse_rotations(
                        undotransform2center(cv_world, ori_mean, scale_origin, center),
                        rotation_matrices,
                    )
                    # cloth_prep.faces_global is global particle indices; convert to local
                    cf = (cloth_prep.faces_global.long() - (cloth_prep.n_existing + n_e_loc)).to(device)
                    n_F = cf.shape[0]
                    # register cloth material once (uses MPMAvatar avg cloth color).
                    # material_id must be the next-available slot (engine auto-assigns
                    # by len(registered_materials) elsewhere, see engine.py:484).
                    cloth_mat_name = "cloth_a1_mpmavatar"
                    if cloth_mat_name not in engine.primitives.registered_materials:
                        # Configurable via sim_params.cloth_render_color (RGB list).
                        # Default = saturated forest green for visibility.
                        _crc = sim_params.get("cloth_render_color", [0.20, 0.65, 0.30])
                        cloth_rgb = torch.tensor([_crc[0], _crc[1], _crc[2], 1.0],
                                                 device=device, dtype=torch.float32)
                        cloth_mat_id = len(engine.primitives.registered_materials)
                        cloth_material = PBRMaterial(
                            material_id=cloth_mat_id,
                            diffuse_map=cloth_rgb.expand(2, 2, 4),
                            diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                            emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                            metallic_factor=0.0,
                            roughness_factor=0.5,
                            transmission_factor=0.0,
                            ior=1.0,
                        )
                        engine.primitives.registered_materials[cloth_mat_name] = cloth_material
                    cloth_mat_id = engine.primitives.registered_materials[cloth_mat_name].material_id
                    cloth_assign = torch.full((n_F,), cloth_mat_id, dtype=torch.int16, device=device)
                    cloth_uv_base = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], device=device)
                    cloth_face_uvs = cloth_uv_base.unsqueeze(0).repeat(n_F, 1, 1).contiguous()
                    cloth_mesh = SurfaceMesh(
                        vertices=cv_render,
                        faces=cf,
                        face_uvs=cloth_face_uvs,
                        material_assignments=cloth_assign,
                    )
                    engine.primitives.PROCEDURAL_SHAPES['rendermesh_cloth'] = lambda *args, **kwargs: cloth_mesh
                    engine.primitives.add_primitive(
                        geometry_type='rendermesh_cloth',
                        primitive_type=OptixPrimitiveTypes.DIFFUSE,
                        device='cuda',
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
                # slice to surface-only (interior filled particles sit at the tail and are not rendered)
                rot_ = rot_[:init_gs_num]
                scale = scale[:init_gs_num]
                if (rot - rot_).abs().max() > 1e-7 :
                    print("rot is wrong : {0:06f}".format((rot - rot_).abs().max()))
                rot_ = apply_inverse_rot_rotations(rot_, rotation_matrices)
                quat = rotmat_to_quat(rot_, ordering="wxyz", orthonormalize=True)
                scale = scale / scale_origin

                # ---- cloth render: append cloth vertices (no SHS/opacity/scale set up
                # in mpm_params for cloth). Use a constant color via SH DC component.
                # Clone all warp-shared tensors to avoid memory aliasing across the
                # next p2g2p step. ----
                render_pos = pos.clone()
                render_shs = shs.clone()
                render_opacity = opacity.clone()
                render_scale = scale.clone()
                render_quat = quat.clone()
                render_screen = (mpm_params["screen_points"][:init_gs_num] if mpm_params["screen_points"].shape[0] > init_gs_num else mpm_params["screen_points"]).clone().detach()
                if cloth_prep is not None and sim_params.get("render_cloth", True):
                    n_e = cloth_prep.n_elements
                    n_v = cloth_prep.n_vertices
                    cloth_v_start = cloth_prep.n_existing + n_e
                    cloth_v_end   = cloth_prep.n_existing + n_e + n_v
                    cloth_pos_world = mpm_solver.export_particle_x_to_torch()[cloth_v_start:cloth_v_end].clone()
                    cloth_pos_render = apply_inverse_rotations(
                        undotransform2center(cloth_pos_world, ori_mean, scale_origin, center),
                        rotation_matrices,
                    )
                    SH_C0 = 0.28209479177387814
                    cloth_rgb = torch.tensor([0.85, 0.20, 0.20], device=device, dtype=torch.float32)  # red
                    cloth_shs_t = torch.zeros(n_v, shs.shape[1], 3, device=device, dtype=shs.dtype)
                    cloth_shs_t[:, 0, :] = (cloth_rgb - 0.5) / SH_C0
                    cloth_op  = torch.ones(n_v, 1, device=device, dtype=opacity.dtype) * 0.99
                    cloth_sigma = float(sim_params.get("cloth_render_sigma", 0.008))
                    cloth_sc  = torch.ones(n_v, 3, device=device, dtype=scale.dtype) * cloth_sigma
                    cloth_qt  = torch.zeros(n_v, 4, device=device, dtype=quat.dtype); cloth_qt[:, 0] = 1
                    cloth_screen = torch.zeros(n_v, 3, device=device, dtype=render_screen.dtype)

                    render_pos = torch.cat([render_pos, cloth_pos_render], 0)
                    render_shs = torch.cat([render_shs, cloth_shs_t], 0)
                    render_opacity = torch.cat([render_opacity, cloth_op], 0)
                    render_scale = torch.cat([render_scale, cloth_sc], 0)
                    render_quat = torch.cat([render_quat, cloth_qt], 0)
                    render_screen = torch.cat([render_screen, cloth_screen], 0)

                rendering, raddi = rasterize(
                    means3D=render_pos,
                    means2D=render_screen,
                    shs=render_shs, # shs
                    colors_precomp=None, # rotation, export_particle_R_to_torch, compute_R_from_F
                    opacities=render_opacity,
                    scales=render_scale, # [N, 3]
                    rotations=render_quat, # [N, 4]
                    cov3D_precomp=None, # export_particle_cov_to_torch, compute_cov_from_F
                )
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                _rot_kind = sim_params.get("image_rotate_kind", None)
                if _rot_kind is None and sim_params.get("image_rotate_cw_90", False):
                    _rot_kind = "cw90"
                if _rot_kind == "cw90":
                    cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
                elif _rot_kind == "ccw90":
                    cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif _rot_kind == "180":
                    cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_180)
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
