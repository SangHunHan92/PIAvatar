import json
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.engine_utils import *
import pytorch3d
from utils.transformation_utils import *
from mpm_solver_warp.mpm_human_utils import modify_animatable_gaussians, modify_smplx

def decode_param_json(json_file):
    f = open(json_file)
    total_params = json.load(f)
    
    # simualtion parameters
    if "sim_params" not in total_params.keys():
        raise ValueError("Simulation parameters should be given")
    sim_params = total_params["sim_params"]
    if "name" not in sim_params.keys():
        sim_params["name"] = "default"
        print("Simulation name is not given. Set to [default]")
    if "output_path" not in sim_params.keys():
        sim_params["output_path"] = "output/defalut/"
        print("Output path is not given. Set to [output/default/]")
    if "render_img" not in sim_params.keys():
        sim_params["render_img"] = False
    if "render_3dgrut" not in sim_params.keys():
        sim_params["render_3dgrut"] = False
    if "compile_video" not in sim_params.keys():
        sim_params["compile_video"] = False
    if "white_bg" not in sim_params.keys():
        sim_params["white_bg"] = False
    if "debug" not in sim_params.keys():
        sim_params["debug"] = False
    if "output_ply" not in sim_params.keys():
        sim_params["output_ply"] = False
    if "output_h5" not in sim_params.keys():
        sim_params["output_h5"] = False
        
    # simualtion parameters 2
    if "grid_lim" in total_params.keys():
        sim_params["grid_lim"] = total_params["grid_lim"]
    else:
        sim_params["grid_lim"] = 2.0

    if "n_grid" in total_params.keys():
        sim_params["n_grid"] = total_params["n_grid"]
    else:
        sim_params["n_grid"] = 50
        
    if "penalty_d" in total_params.keys():
        sim_params["penalty_d"] = total_params["penalty_d"]
    else:
        sim_params["penalty_d"] = 5000.0
    
    if "penalty_v" in total_params.keys():
        sim_params["penalty_v"] = total_params["penalty_v"]
    else:
        sim_params["penalty_v"] = 200.0
    
    if "penalty_th" in total_params.keys():
        sim_params["penalty_th"] = total_params["penalty_th"]
    else:
        sim_params["penalty_th"] = 1.5
        
    if "g" in total_params.keys():
        sim_params["g"] = total_params["g"]
    else:
        sim_params["g"] = [0.0, 0.0, 0.0]
        
    
    # subject conditions
    subject_params = {}
    if "subject_conditions" in total_params.keys():
        subject_params = total_params["subject_conditions"]
        
        check_human_front = True
        for subject in subject_params:
            if subject["human"] == True and check_human_front == False:
                raise ValueError("Human subjects should be located before non-human subjects in subject_conditions!")
            else:
                check_human_front = subject["human"]        
        
        for subject in subject_params:
            if subject["model"] != "smplx" and "model_path" not in subject.keys():
                raise ValueError("Model path should be given")
            if subject["model"] == "animatable_gaussians" and "config_path" not in subject.keys():
                raise ValueError("Config path should be given for animatable_gaussians")
                # raise ValueError("Config path should be given for human subjects")            
            if "index" not in subject.keys(): # index를 지정하지 않고, 자동으로 부여하게 수정?
                raise ValueError("All subjects should have each index")
            
            if "rotation_degree" not in subject.keys():
                subject["rotation_degree"] = [0.0]
            if "rotation_axis" not in subject.keys():
                subject["rotation_axis"] = [0]
            if "center" not in subject.keys():
                subject["center"] = [1, 1, 1]
            if "scale" not in subject.keys():
                subject["scale"] = 1.0
            if "initial_velocity" not in subject.keys():
                subject["initial_velocity"] = [0, 0, 0]
            if "crop_area" not in subject.keys():
                subject["crop_area"] = None
            if "density" not in subject.keys():
                subject["density"] = 200.0
            if "E" not in subject.keys():
                subject["E"] = 1e5
            if "nu" in subject.keys():
                if subject["nu"] > 0.5 or subject["nu"] < 0.0:
                    raise ValueError("Poisson's ratio should be less than 0.5")
            else:
                subject["nu"] = 0.4
                
            if "g" not in subject.keys():
                subject["g"] = [0, 0, 0]
            elif sim_params["g"] != [0.0, 0.0, 0.0] and subject["g"] != [0.0, 0.0, 0.0]:
                raise ValueError("Global gravity and local gravity cannot be applied at the same time, One of them must be set to [0, 0, 0]")
            if "opacity_threshold" not in subject.keys():
                subject["opacity_threshold"] = 0.02
            if "material" not in subject.keys():
                subject["material"] = "jelly"
            
    # boundary conditions
    bc_params = {} # It will be python list
    if "boundary_conditions" in total_params.keys():
        bc_params = total_params["boundary_conditions"]

    # time step
    time_params = {}
    if "substep_dt" in total_params.keys():
        time_params["substep_dt"] = total_params["substep_dt"]
    else:
        time_params["substep_dt"] = 1e-4

    if "frame_dt" in total_params.keys():
        time_params["frame_dt"] = total_params["frame_dt"]
    else:
        time_params["frame_dt"] = 1e-2

    if "frame_num" in total_params.keys():
        time_params["frame_num"] = total_params["frame_num"]
    else:
        time_params["frame_num"] = 100    
        
    if "smplx_dt" in total_params.keys():
        time_params["smplx_dt"] = total_params["smplx_dt"]
    else:
        time_params["smplx_dt"] = 1e-2

    # camera params
    camera_params = {}
    if "mpm_space_viewpoint_center" in total_params.keys():
        camera_params["mpm_space_viewpoint_center"] = total_params["mpm_space_viewpoint_center"]
    else:
        camera_params["mpm_space_viewpoint_center"] = [1.0, 1.0, 1.0]
    if "mpm_space_vertical_upward_axis" in total_params.keys():
        camera_params["mpm_space_vertical_upward_axis"] = total_params["mpm_space_vertical_upward_axis"]
    else:
        camera_params["mpm_space_vertical_upward_axis"] = [0, 0, 1]
    if "default_camera_index" in total_params.keys():
        camera_params["default_camera_index"] = total_params["default_camera_index"]
    else:
        camera_params["default_camera_index"] = 0
    if "show_hint" in total_params.keys():
        camera_params["show_hint"] = total_params["show_hint"]
    else:
        camera_params["show_hint"] = False
    if "init_azimuthm" in total_params.keys():
        camera_params["init_azimuthm"] = total_params["init_azimuthm"]
    else:
        camera_params["init_azimuthm"] = None
    if "init_elevation" in total_params.keys():
        camera_params["init_elevation"] = total_params["init_elevation"]
    else:
        camera_params["init_elevation"] = None
    if "init_radius" in total_params.keys():
        camera_params["init_radius"] = total_params["init_radius"]
    else:
        camera_params["init_radius"] = None
    if "delta_a" in total_params.keys():
        camera_params["delta_a"] = total_params["delta_a"]
    else:
        camera_params["delta_a"] = None
    if "delta_e" in total_params.keys():
        camera_params["delta_e"] = total_params["delta_e"]
    else:
        camera_params["delta_e"] = None
    if "delta_r" in total_params.keys():
        camera_params["delta_r"] = total_params["delta_r"]
    else:
        camera_params["delta_r"] = None
    if "move_camera" in total_params.keys():
        camera_params["move_camera"] = total_params["move_camera"]
    else:
        camera_params["move_camera"] = False
        
    human_params = {}
    if "smplx_path" in total_params.keys():
        human_params["smplx_path"] = total_params["smplx_path"]
    else:
        human_params["smplx_path"] = os.path.join(os.getcwd(), "smpl_models", "smplx")
        print("Assume smplx model is in ", human_params["smplx_path"])

    return sim_params, subject_params, bc_params, time_params, camera_params, human_params


def set_boundary_conditions(
    mpm_solver: MPM_Simulator_WARP, bc_params: dict, time_params: dict, device="cuda"
):
    for bc in bc_params:
        if bc["type"] == "cuboid":
            assert (
                "point" in bc.keys() and "size" in bc.keys() and "velocity" in bc.keys()
            )
            start_time = 0.0
            end_time = 1e3
            reset = 0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            if "end_time" in bc.keys():
                end_time = bc["end_time"]
            if "reset" in bc.keys():
                reset = bc["reset"]
            mpm_solver.set_velocity_on_cuboid(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=start_time,
                end_time=end_time,
                reset=reset,
            )

        elif bc["type"] == "particle_impulse":
            assert "force" in bc.keys()

            start_time = 0.0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            num_dt = 1
            if "num_dt" in bc.keys():
                num_dt = bc["num_dt"]
            point = [1, 1, 1]
            if "point" in bc.keys():
                point = bc["point"]
            size = [1, 1, 1]
            if "size" in bc.keys():
                size = bc["size"]

            mpm_solver.add_impulse_on_particles(
                force=bc["force"],
                dt=time_params["substep_dt"],
                point=point,
                size=size,
                num_dt=num_dt,
                start_time=start_time,
            )

        elif bc["type"] == "bounding_box":
            mpm_solver.add_bounding_box()

        elif bc["type"] == "enforce_particle_translation":
            assert "point" in bc.keys()
            assert "size" in bc.keys()
            assert "velocity" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "index" in bc.keys()

            # mpm_solver.mpm_state.particle_x.numpy().min() # 0.45877
            mpm_solver.enforce_particle_velocity_translation(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
                index=bc["index"]
            )
            
        elif bc["type"] == "surface_collider":
            assert "point" in bc.keys()
            assert "normal" in bc.keys()
            assert "surface" in bc.keys()
            assert "friction" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.add_surface_collider(
                point=bc["point"],
                normal=bc["normal"],
                surface=bc["surface"],
                friction=bc["friction"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        elif bc["type"] == "release_particles_sequentially":
            assert "normal" in bc.keys()
            assert "start_position" in bc.keys()
            assert "end_position" in bc.keys()
            assert "num_layers" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.release_particles_sequentially(
                normal=bc["normal"],
                start_position=bc["start_position"],
                end_position=bc["end_position"],
                num_layers=bc["num_layers"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
            
        elif bc["type"] == "enforce_particle_velocity_rotation":
            assert "normal" in bc.keys()
            assert "point" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "half_height_and_radius" in bc.keys()
            assert "rotation_scale" in bc.keys()
            assert "translation_scale" in bc.keys()

            mpm_solver.enforce_particle_velocity_rotation(
                point=bc["point"],
                normal=bc["normal"],
                half_height_and_radius=bc["half_height_and_radius"],
                rotation_scale=bc["rotation_scale"],
                translation_scale=bc["translation_scale"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
            
        elif bc["type"] == "enforce_smpl_velocity":
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "velocity" in bc.keys()
            
            mpm_solver.enforce_smpl_velocity(
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        
        elif bc["type"] == "enforce_joint_velocity":            
            mpm_solver.enforce_joint_velocity(
                index=bc["index"],
                canonical=bc["canonical"],
                lbs=bc["lbs"],
                joint_mats=bc["joint_mats"],
                smplx_dt=time_params["smplx_dt"],
                rot_mats=bc["rot_mats"],
                ori_mean=bc["ori_mean"],
                scale=bc["scale"],
                center=bc["center"],
                device=device                
            )
        
        elif bc["type"] == "modify_animatable_gaussians":            
            # mpm_solver.modify_posed_human(            
            modify_animatable_gaussians(mpm_solver,
                model_type = bc["model_type"],
                velocity_type = bc["velocity_type"],
                velocity_alpha = bc["velocity_alpha"],
                avatar_net=bc["avatar_net"],
                pose_dataset=bc["pose_dataset"],
                cano_pts=bc["cano_pts"],
                cano_rot=bc["cano_rot"],
                cano_J=bc["cano_J"],
                joint_mat=bc["joint_mat"],
                A_mat=bc["A_mat"],
                knn_indices=bc["knn_indices"],                
                bone_cano=bc["bone_cano"],
                bone_index=bc["bone_index"],
                bone_faces=bc["bone_faces"],
                particle_start=bc["particle_start"],
                index=bc["index"],
                rot_mats=bc["rot_mats"],
                ori_mean=bc["ori_mean"],
                scale=bc["scale"],
                center=bc["center"],
                g_time=bc["g_time"],
                device=device
            )
        
        elif bc["type"] == "modify_smplx":     
            modify_smplx(mpm_solver,
                model_type = bc["model_type"],
                velocity_type = bc["velocity_type"],
                velocity_alpha = bc["velocity_alpha"],
                # avatar_net=bc["avatar_net"],
                pose_dataset=bc["pose_dataset"],
                betas=bc["betas"],
                smplx_model = bc['smplx_model'],
                cano_pts=bc["cano_pts"],
                # cano_rot=bc["cano_rot"],
                cano_J=bc["cano_J"],
                # joint_mat=bc["joint_mat"],
                # A_mat=bc["A_mat"],
                # knn_indices=bc["knn_indices"],                
                bone_cano=bc["bone_cano"],
                bone_index=bc["bone_index"],
                # bone_faces=bc["bone_faces"],
                particle_start=bc["particle_start"],
                index=bc["index"],
                rot_mats=bc["rot_mats"],
                ori_mean=bc["ori_mean"],
                scale=bc["scale"],
                center=bc["center"],
                g_time=bc["g_time"],
                device=device
            )
            
        else:
            raise TypeError("Undefined BC type")

def set_human_sequences_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params):
    
    for i, (human, param) in enumerate(zip(human_sequences, subject_params)):
        if human is None:
            continue
        new_bc = dict()
        new_bc["type"] = "enforce_joint_velocity"
        new_bc["index"] = i
        new_bc["canonical"] = human['canonical']
        new_bc["lbs"] = human['lbs']
        new_bc["joint_mats"] = human['joint_mats']
        
        new_bc["rot_mats"] = param['rot_mats'][0]
        new_bc["ori_mean"] = param['ori_mean']
        new_bc["scale"] = param['scale']
        new_bc["center"] = torch.tensor(param['center'], device="cuda")
        
        bc_params.append(new_bc)
    
    return bc_params

def set_human_model_to_boundary_conditions(human_sequences, subject_params, time_params, bc_params):
    
    for i, (human, param) in enumerate(zip(human_sequences, subject_params)):
        
        if human is None:
            continue
        
        if param['model'] == 'animatable_gaussians':
            new_bc = dict()
            new_bc["type"] = "modify_animatable_gaussians"
            new_bc["model_type"] = "animatable_gaussians"
            new_bc["velocity_type"] = param['velocity_type']
            new_bc["velocity_alpha"] = param['velocity_alpha']
            new_bc["index"] = i
            new_bc["avatar_net"] = human['avatar_net']
            new_bc["pose_dataset"] = human['pose_dataset']        
            new_bc["cano_pts"] = human['cano_pts']
            new_bc["cano_rot"] = human['cano_rot']
            new_bc["cano_J"] = human['cano_J']
            new_bc["joint_mat"] = human['joint_mat']
            new_bc["A_mat"] = human['A_mat']
            new_bc["knn_indices"] = human['knn_indices']
            new_bc["bone_cano"] = human['bone_cano']
            new_bc["bone_index"] = human['bone_index']
            new_bc["bone_faces"] = human['bone_faces']
            new_bc["particle_start"] = human["particle_start"]
            
            new_bc["rot_mats"] = param['rot_mats'][0]
            new_bc["ori_mean"] = param['ori_mean']
            new_bc["scale"] = param['scale']
            new_bc["center"] = torch.tensor(param['center'], device="cuda")
            new_bc['g_time'] = param['g_time']
            
            bc_params.append(new_bc)
        
        elif param['model'] == 'smplx':
            new_bc = dict()
            new_bc["type"] = "modify_smplx"
            new_bc["model_type"] = "smplx"
            new_bc["velocity_type"] = param['velocity_type']
            new_bc["velocity_alpha"] = param['velocity_alpha']
            new_bc["index"] = i
            new_bc["betas"] = human['betas']
            new_bc["smplx_model"] = human['smplx_model']
            new_bc["pose_dataset"] = human['pose_dataset']
            new_bc["cano_pts"] = human['cano_pts']
            # new_bc["cano_rot"] = human['cano_rot'] #
            new_bc["cano_J"] = human['cano_J'] #
            # new_bc["joint_mat"] = human['joint_mat'] #
            # new_bc["A_mat"] = human['A_mat'] #
            new_bc["bone_cano"] = human['bone_cano']
            new_bc["bone_index"] = human['bone_index']
            # new_bc["bone_faces"] = human['bone_faces'] #
            new_bc["particle_start"] = human["particle_start"]
            
            new_bc["rot_mats"] = param['rot_mats'][0]
            new_bc["ori_mean"] = param['ori_mean']
            new_bc["scale"] = param['scale']
            new_bc["center"] = torch.tensor(param['center'], device="cuda")
            new_bc['g_time'] = param['g_time']
            
            bc_params.append(new_bc)
    
    return bc_params