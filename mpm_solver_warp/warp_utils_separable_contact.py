import warp as wp
import warp.torch
import numpy as np
import torch
from AnimatableGaussians import config, smplx
from torch.utils.data import Dataset
import torch.nn as nn

@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    material: int
    material_list: wp.array(dtype=int)
    penalty_d: float
    penalty_v: float
    penalty_th: float

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: float
    alpha: float
    gravitational_accelaration: wp.vec3
    hardening: float
    xi: float
    plastic_viscosity: float
    softening: float

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int
    
    ####### for avatar stress
    n_subjects: int
    n_humans: int
    bone_index: wp.array(dtype=int)
    # n_slot: int
    
    ######## method ########
    method: int

    ####### separable-contact (Bardenhagen 2000 multi-field MPM)
    use_separable_contact: int   # 0 = original single-field path, 1 = per-subject separable contact
    contact_eps: float           # mass / normal-length threshold


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    # particle_v_given: wp.array(dtype=wp.vec3) # human particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    particle_cov: wp.array(dtype=float)  # current covariance matrix
    particle_F_trial: wp.array(dtype=wp.mat33)  # apply return mapping on this to obtain elastic def grad    
    particle_quat: wp.array(dtype=wp.vec4f)
    particle_scale: wp.array(dtype=wp.vec3f)
    particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)
    particle_gravity: wp.array(dtype=wp.vec3)
    particle_S: wp.array(dtype=wp.mat33) # scaling matrix
    particle_selection: wp.array(dtype=int)  # only particle_selection[p] = 0 will be simulated
    
    ####### for avatar stress
    particle_id: wp.array(dtype=int)
    particle_vk: wp.array(dtype=wp.vec3)
    particle_vko: wp.array(dtype=wp.vec3)
    particle_Fe: wp.array(dtype=wp.mat33)
    particle_F_add: wp.array(dtype=wp.mat33)  # apply return mapping on this to obtain elastic def grad
    particle_Fe_trial: wp.array(dtype=wp.mat33)
    particle_Fk: wp.array(dtype=wp.mat33)
    particle_F_before: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_Ce: wp.array(dtype=wp.mat33)
    particle_Ce_trial: wp.array(dtype=wp.mat33)
    particle_Ck: wp.array(dtype=wp.mat33)
    particle_material: wp.array(dtype=int)
    avatar_offset: wp.array(dtype=int)
    particle_vSM: wp.array(dtype=wp.vec3)
    particle_LBS: wp.array(dtype=wp.float32, ndim=2)
    particle_SM_test: wp.array(dtype=wp.vec3)
    
    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3)  # grid node velocity, after grid update
        
    ####### individual stress, atomicCAS
    grid_f: wp.array(dtype=wp.vec3, ndim=4)
    grid_vk: wp.array(dtype=wp.vec3, ndim=4)  # grid node momentum
    grid_mk: wp.array(dtype=float, ndim=4)  # grid node momentum
    grid_id: wp.array(dtype=int, ndim=4)
    grid_count: wp.array(dtype=int, ndim=3)

    ####### separable-contact per-subject grid (Bardenhagen 2000)
    grid_p_in_s: wp.array(dtype=wp.vec3, ndim=4)   # per-subject momentum from P2G   [nx,ny,nz,n_subjects]
    grid_ms: wp.array(dtype=float, ndim=4)         # per-subject mass                 [nx,ny,nz,n_subjects]
    grid_v_subj: wp.array(dtype=wp.vec3, ndim=4)   # per-subject velocity (p/m + dt g)
    grid_v_resolved: wp.array(dtype=wp.vec3, ndim=4)  # per-subject velocity after contact resolve
    grid_normal: wp.array(dtype=wp.vec3, ndim=4)   # per-subject ~ grad m_k
       
    # bone    
    bone_mx:   wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
    bone_mv:   wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
    bone_m:    wp.array(dtype=wp.float32, ndim=2)     # [num of human, 20]
    bone_L:    wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
    bone_I:    wp.array(dtype=wp.mat33, ndim=2)  # [num of human, 20]
    bone_w:    wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 20]
    bone_A:    wp.array(dtype=wp.mat33, ndim=2)  # [num of human, 20]
    bone_Apose:wp.array(dtype=wp.mat44, ndim=2)  # [num of human, 20]
    bone_R:    wp.array(dtype=wp.mat33, ndim=2)  # [num of human, 20]
    bone_x0:   wp.array(dtype=wp.vec3, ndim=2)   # [num of human, 74496], bone canonical
    bone_x0cm: wp.array(dtype=wp.vec3, ndim=2) # [num of human, 20], bone canonical center
    bone_q:    wp.array(dtype=wp.vec3, ndim=2) # [num of human, 74496], mass center of bone canonical
    bone_idx:  wp.array(dtype=wp.int16)          # [num of particles]
    bone_pnum: wp.array(dtype=wp.int16)          # [20]
    
    # grid_v_in_prescribed: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum (+ v_prescribed)
    # grid_v_out_prescribed: wp.array(dtype=wp.vec3, ndim=3)  # grid node velocity, after grid update (+ v_prescribed)
    # grid_v_mean_pos: wp.array(dtype=wp.vec3, ndim=4)
    # grid_v_particle_num: wp.array(dtype=int, ndim=4)
    # grid_v_check: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum
    # grid_v_check2: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum
    # grid_v_out_human: wp.array(dtype=wp.vec3, ndim=4)  # grid node velocity, after grid update
    
    # debug
    frame: int
    step: int
    


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float

@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float
    
@wp.struct
class HumanModifier:
    particle_id: wp.array(dtype=int)
    velocity: wp.array(dtype=wp.vec3)
    angular_velocity: wp.array(dtype=wp.mat33)
    g: wp.vec3
    g_frame: int

class HumanTorchModel:
    model_type: str
    index: int
    avatar_index: slice
    avatar_net: torch.nn.Module # AvatarNet
    pose_dataset: Dataset # smplx.SMPLX
    cano_xyz: torch.tensor
    cano_rot: torch.tensor
    cano_J: torch.tensor
    joint_mat: torch.tensor
    A_mat: torch.tensor
    knn_indices: torch.tensor
    particle_start: int
    betas: torch.tensor
    smplx_model: nn.Module
    
    bone_cano: torch.tensor
    bone_index: list
    bone_faces: list
    bone2smplx: list
    
    rot_mats: torch.tensor # bc params
    ori_mean: torch.tensor
    scale: torch.tensor
    center: torch.tensor
    
    # g_frame: int
    # g: torch.tensor
    
    human_n_particles: int

@wp.struct
class SMPLVelocityModifier:
    start_time: float
    end_time: float
    velocity: wp.array(dtype=wp.vec3)
    mask: wp.array(dtype=int)
      
@wp.struct
class JointVelocityModifier:
        
    # human num
    index: int
    cano_xyz: wp.array2d(dtype=wp.vec3)
    cano_rot: wp.array2d(dtype=wp.vec4)
    lbs: wp.array2d(dtype=wp.float32)
    joint_mats: wp.array2d(dtype=wp.mat44)
    smplx_dt: float
    substep_dt: float
    cano_num_particles: int
    particle_id: wp.array(dtype=int)
    
    velocity: wp.array(dtype=wp.vec3)
    angular_velocity: wp.array(dtype=wp.mat33)
    
# @wp.struct
# class HumanVelocityModifier:
#     # position: wp.array(dtype=wp.vec3)
#     velocity: wp.array(dtype=wp.vec3)
#     angular_velocity: wp.array(dtype=wp.mat33)
#     # particle_id: wp.array(dtype=int)

@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float
    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    start_time: float
    end_time: float
    velocity: wp.vec3        
    mask: wp.array(dtype=int)
    index: int

@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])

@wp.kernel
def set_subject_value_to_float_array(
    index_array: wp.array(dtype=int), target_array: wp.array(dtype=float), index: int, value: float
    ):
    tid = wp.tid()
    if index_array[tid] == index:
        target_array[tid] = value

@wp.kernel
def set_subject_vec_to_float_array(
    index_array: wp.array(dtype=int), target_array: wp.array(dtype=wp.vec3), index: int, vec: wp.vec3
    ):
    tid = wp.tid()
    if index_array[tid] == index:
        target_array[tid] = vec

@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value
    
@wp.kernel
def set_value_to_float_array_condition(
    target_array: wp.array(dtype=float), condition_array: wp.array(dtype=int), value: float, condition: int):
    tid = wp.tid()
    if condition_array[tid] == condition:        
        target_array[tid] = value

@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3d(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float64:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3d,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
