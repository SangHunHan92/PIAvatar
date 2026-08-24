import warp as wp
from warp_utils_separable_contact import *
import numpy as np
import math
import pytorch3d
import torch
import torch.nn.functional as F
import trimesh
import time
from mpm_utils_separable_contact import overwrite_R_to_F


# FROM : main/set_boundary_conditions
def modify_animatable_gaussians(MPM_sim, model_type, velocity_type, velocity_alpha, avatar_net, pose_dataset, cano_pts, cano_rot, cano_J, joint_mat, A_mat, knn_indices,
    bone_cano, bone_index, bone_faces, particle_start, index, rot_mats, ori_mean, scale, center, g_time, device="cuda"
):
    # 여기부터 작성
    # input은 pose -> main_avatar_phys.py/get_avatars 참고해서 smpl로부터 값을 유추하기
    
    # 1. cano_xyz, cano_rot, lbs, cano_num_particles
    human_model = HumanTorchModel()
    human_model.model_type = model_type
    human_n_particles = bone_index[-1] + avatar_net.lbs.shape[0]  # [Bone Particles] + [Human Particles] 
    # human_n_particles = avatar_net.lbs.shape[0]                     # [Bone Particles] + [Human Particles] 
            
    human_model.index = index
    human_model.avatar_index = slice(bone_index[-1], human_n_particles) # particle_start 반영 필요
    human_model.avatar_net = avatar_net
    human_model.pose_dataset = pose_dataset
    human_model.cano_xyz = cano_pts                               # [Human Particles, 3], not need to define init?
    human_model.cano_rot = cano_rot
    human_model.cano_J = cano_J
    human_model.joint_mat = joint_mat                             # [55, 4, 4], joint_mat = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
    human_model.A_mat = A_mat
    human_model.knn_indices = knn_indices
    
    human_model.particle_start = particle_start
    # torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) @ ag_avatar
    # torch.matmul(live_smpl.A[0]) @ bone_cano
    
    human_model.bone_cano = bone_cano # [74496, 3]
    human_model.bone_index = bone_index
    human_model.bone_faces = bone_faces
    human_model.bone2smplx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1, 4, 2, 5, 7, 8]
    
    # human_model.lbs = avatar_net.lbs
    human_model.rot_mats = rot_mats # bc params
    human_model.ori_mean = ori_mean
    human_model.scale = scale
    human_model.center = center
    
    # human_model.add_velocity = torch.zeros_like(human_model.cano_xyz, device=device)       # [Human Particles, 3]
    # human_model.add_angular_velocity = torch.zeros_like(human_model.cano_rot, device=device)   # [Human Particles, 4]
    human_model.human_n_particles = human_n_particles              # [Human Particles]
    human_model.velocity_type = velocity_type
    human_model.velocity_alpha = velocity_alpha
    
    MPM_sim.human_modify_model.append(human_model) # torch velocity
    
    # 2.
    human_params = HumanModifier()   
    human_params.g = wp.vec3(g_time[1], g_time[2], g_time[3])    
    human_params.g_frame = g_time[0]     
    _human_idx_np = np.where(MPM_sim.mpm_state.particle_id.numpy() == index)[0]
    human_params.particle_id = wp.array(_human_idx_np, dtype=int) # MPM particle indices for this human (NOT contiguous when interior is appended after another subject)
    # store torch version on human_model so compute_*_velocity can index particle_x correctly
    MPM_sim.human_modify_model[-1].particle_id_torch = torch.from_numpy(_human_idx_np).long().to(device)       
    MPM_sim.human_modify_params.append(human_params) # warp velocity

    # 3. velocity
    @wp.kernel
    def kinematic_velocity(
        state: MPMStateStruct,
        human_params: HumanModifier,
        kinematic_v: wp.array(dtype=wp.vec3), 
        relR: wp.array(dtype=wp.mat33),
        apply_rot: int,
        vSM: wp.array(dtype=wp.vec3),
        frame: int
        # human_modify_changer: wp.array(dtype=wp.mat33)
    ): # dim = human_n_particles (avatar + bone)
        p = wp.tid()
        id = human_params.particle_id[p]        
        state.particle_v[id]   = state.particle_v[id] + kinematic_v[p] - state.particle_vk[id]
        state.particle_vk[id]  = kinematic_v[p]
        state.particle_vko[id] = kinematic_v[p]
        # state.particle_v[id]   = state.particle_v[id] + kinematic_v[p] - state.particle_vk[id] + state.particle_vSM[p]
        # state.particle_vk[id]  = kinematic_v[p] + state.particle_vSM[p]
        # state.particle_vko[id] = kinematic_v[p] + state.particle_vSM[p]        
        state.particle_F_trial[id] = relR[p] * state.particle_F_trial[id]
        # state.particle_F_trial[id] = overwrite_R_to_F(state.particle_F_trial[id], human_modify_changer[p])
        
        if frame == human_params.g_frame:
            state.particle_gravity[id] = human_params.g
        # state.particle_F[id] = relR[p] * state.particle_F[id]
        
        # bid = state.bone_idx[p]
        # if bid < 0: # if not bone(if avatar)
        # state.particle_vSM[p] = vSM[p]
        
    MPM_sim.human_modify_changer.append(kinematic_velocity)
            
    # 4. apply particle bone index
    # bone + avatar
    # MPM_sim.mpm_state.bone_idx # [N]
    ps = particle_start
    # particle_bone_idx = ps + np.array(range(bone_index[-1])) # [74496]
    particle_bone_val = np.zeros(bone_index[-1], dtype=np.int16) # [74496]
    # particle_bone_val = np.zeros_like(particle_bone_idx)
    for i in range(len(bone_index)-1):
        particle_bone_val[ bone_index[i] : bone_index[i+1] ] = i
    
    state_bone_idx = MPM_sim.mpm_state.bone_idx.numpy()
    state_bone_idx[ps:ps+bone_index[-1]] = particle_bone_val
    MPM_sim.mpm_state.bone_idx = wp.array(state_bone_idx, dtype=wp.int16, device=device)
    
    # 5. save bone cano
    bone_cano_torch = (torch.mm(bone_cano, rot_mats.T) - ori_mean) * scale + center # GT2Sim coordinate system
    bone_cano_wp    = torch2warp_vec3(bone_cano_torch, dvc=device)
    
    bone_mass_torch = wp.to_torch(MPM_sim.mpm_state.particle_mass)[ps:ps+bone_index[-1]]
    x_splits = torch.split(bone_cano_torch, MPM_sim.bone_p_num.tolist(), dim=0)
    m_splits = torch.split(bone_mass_torch, MPM_sim.bone_p_num.tolist(), dim=0)
    
    bone_cano_c_torch = torch.stack([chunk.mean(dim=0) for chunk in x_splits], dim=0) # [20, 3]
    bone_cano_c_wp    = torch2warp_vec3(bone_cano_c_torch, dvc=device)
    # bone_cano_c_wp    = wp.from_torch(bone_cano_c_torch.to(torch.double).detach(), dtype=wp.vec3d)
    
    bone_cano_q_torch = bone_cano_torch - bone_cano_c_torch[particle_bone_val]
    bone_cano_q_wp    = torch2warp_vec3(bone_cano_q_torch, dvc=device)
    # bone_cano_p_wp    = wp.from_torch(bone_cano_p_torch.to(torch.double).detach(), dtype=wp.vec3d)

    @wp.kernel
    def particle_bone_cano(arr: wp.array(dtype=wp.vec3, ndim=2), idx: int, val: wp.array(dtype=wp.vec3)):
        p = wp.tid()
        arr[idx, p] = val[p]
    
    wp.launch(
        kernel=particle_bone_cano,
        dim=bone_index[-1], # 74496
        inputs=[MPM_sim.mpm_state.bone_x0, index, bone_cano_wp],
        device=device,
    )   
    wp.launch(
        kernel=particle_bone_cano,
        dim=20, # 20
        inputs=[MPM_sim.mpm_state.bone_x0cm, index, bone_cano_c_wp],
        device=device,
    )
    wp.launch(
        kernel=particle_bone_cano,
        dim=bone_index[-1], # 20
        inputs=[MPM_sim.mpm_state.bone_q, index, bone_cano_q_wp],
        device=device,
    )
            
    @wp.kernel
    def set_bone_E_nu(
        state: MPMStateStruct, 
        model: MPMModelStruct
    ):
        p = wp.tid()
        bone_idx = state.bone_idx[p]
        if bone_idx >= 0:
            E = 1e5
            nu = 0.3
            model.E[p] = E
            model.nu[p] = nu
            model.mu[p] = E / (2.0 * (1.0 + nu))
            model.lam[p] = (
                E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            )
    
    wp.launch(
        kernel=set_bone_E_nu,
        dim=MPM_sim.n_particles,
        inputs=[MPM_sim.mpm_state, MPM_sim.mpm_model],
        device=device,
    )
    
    # 6. avatar_offset
    # For shape matching
    offset_np = MPM_sim.mpm_state.avatar_offset.numpy()
    last_offset = offset_np[-1] + human_n_particles
    new_offset_np = np.concatenate([offset_np, [last_offset]]).astype(np.int32)
    MPM_sim.mpm_state.avatar_offset = wp.array(new_offset_np, dtype=wp.int32, device=device)

@torch.no_grad()
def compute_animatable_gaussians_velocity_rel(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None):
    # list_idx -> human_idx
        
    # end = time.time()
    
    # 1. 뼈대 움직임에서 keypoint R, t 계산 -> 현재 global A-pose matrix 계산
    # 1.1 이때 vector 2개로 R 구하는 방법이 필요할 수도 있다. (contribution 2)
    # torch    
    avatar_net = MPM_sim.human_modify_model[list_idx].avatar_net
    # extr = MPM_sim.human_modify_model[list_idx].extr # [55, 4, 4]
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    knn_indices = MPM_sim.human_modify_model[list_idx].knn_indices
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center
    
    A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
    # A_mat = MPM_sim.human_modify_model[list_idx].A_mat[:22]
    
    # cano_J[:, :3] = (torch.mm(cano_J[:, :3], rot_mats.T) - ori_mean) * scale + center
    # bone_cano = (torch.mm(bone_cano, rot_mats.T) - ori_mean) * scale + center
    # avatar particle : particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
            
    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
    
    # temp test
    kabsch_A = torch.zeros_like(A_now)
    kabsch_A[:, 3, 3] = 1.0
    
    # end2 = time.time()
    for i in range(len(bone_index)-1):
        # time1 = time.time()
        R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
        
        kabsch_A[bone2smplx[i], :3, :3] = R_est # temp test
        kabsch_A[bone2smplx[i], :3,  3] = t_est # temp test
        
        joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
        # 이거 맞는지 어떻게 확인?
        # joint_gt  = pose_J[bone2smplx[i]]
        A_now[bone2smplx[i], :3, :3] = R_est
        A_now[bone2smplx[i], :3, 3] = joint_cal
        if bone2smplx[i] == 7 or bone2smplx[i] == 8: # smpl foot 7 = 10, 8 = 11
            A_now[bone2smplx[i]+3, :3, :3] = R_est
            A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
            kabsch_A[bone2smplx[i]+3, :3, :3] = R_est # temp test
            kabsch_A[bone2smplx[i]+3, :3,  3] = t_est # temp test
        # time1 = time.time() - time1; print(time1*1000, "ms")
        
        # bone_cano[bone_index[i]:bone_index[i+1]].detach().cpu().numpy()
        # particle_x[ps+bone_index[i]:ps+bone_index[i+1]].detach().cpu().numpy()
    # end2 = time.time() - end2
    
    A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
    A_now = A_now.unsqueeze(0)
    
    # kabsch_A <-> A_now[0] relation test !!
    # 완전히 똑같다, kabsch 결과를 A로 대체해도 된다
    # (A_now[0] - kabsch_A).abs().max()
    
    # 2. 현재 global A-pose matrix 에서 다음 global A-pose matrix 계산
    # torch
    # end3 = time.time()
    from AnimatableGaussians.smplx.lbs import batch_rodrigues
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        # now_frame = pose_dataset.pose_list[1]
        # next_frame = pose_dataset.pose_list[1]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        now_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        
        T_pred = A_now[0]
        T_gt = now_smpl.A[0, :22]
        parents = pose_dataset.smpl_model.parents[:22]
        
        rot_err_deg, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
        
        rot_err_deg_mean = rot_err_deg.mean()
        trans_err_mean = trans_err.mean()
        rel_rot_err_deg_mean = rel_rot_err_deg.mean()
        rel_trans_err_mean = rel_trans_err.mean()
        
        if f is not None and frame % 100 == 0:
            print(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f},\
                    rel_rot_err_deg_mean: {rel_rot_err_deg_mean.item():.6f},\
                    rel_trans_err_mean: {rel_trans_err_mean.item():.6f} \n")
            f.write(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f},\
                    rel_rot_err_deg_mean: {rel_rot_err_deg_mean.item():.6f},\
                    rel_trans_err_mean: {rel_trans_err_mean.item():.6f} \n")
        
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]
        now_transl = pose_dataset.transl[now_frame]
        
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(now_smpl.A[0, :22, :3, :3]) # rel rotation, 위의 batch_rodrigues로 대체 가능
        
        # 2.2 다음 frame의 global rotation 계산
        # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        # tgt_A[:, :3, :3] = rel_rot @ now_smpl.A[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        # now_smpl.A[0, :22], A_mat, cal_A # [22, 4, 4], the same
        
        # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        for i in range(1, parents.shape[0]):
            transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4], Eq 11
        
        # A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
        now_transl_sim = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Eq 12, mine
        rel_transforms[0, :, :3, 3] += next_transl - now_transl + now_transl_sim # [1, 22, 4, 4] #%#%#%#
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4] #%#%#%#
        A_next = rel_transforms
        
        
        # (A_next[0] - next_smpl.A[0, :22]).abs().max() # check !!, yes !!
        
        # 3. AG model network, delta_position
        # torch
        test_code = False
        if test_code or is_3d_measure:
            now_smpl_woRoot = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
                                                    body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                    # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                    # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                    left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                    right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
            )
            next_smpl_woRoot = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
                                                    body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                                    # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                                    # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                   
                                                    left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                    right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
            )
        # next_smpl_notfinger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
        #                                         global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
        #                                         transl = pose_dataset.transl[next_frame][None], # [1, 3]   
        #                                         body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
        # )
        # next_smpl_woRoot_notfinger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
        #                                         body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
        # )
        # next_smpl_notfinger.A[0]
        
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        # (A_next_woRoot - next_smpl_woRoot.A[0, :22]).abs().max() # check !!, ok !!
        
        # 3.2 delta_position
        # from AnimatableGaussians.network.avatar import get_outputs
        # network.avatar는 55가 필요하다
        # dataset에서도 다른 pose는 0으로 취급해버리자
        # 다른 pose가 0이면 A-pose matrix는 어떻게 될까? [22:]
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]
                                
        # a = torch.linalg.inv(now_smpl.A[0, 20]) @ now_smpl.A[0, 25:40]
        # b = torch.linalg.inv(next_smpl.A[0, 20]) @ next_smpl.A[0, 25:40]
        # c = torch.linalg.inv(now_smpl_woRoot.A[0, 20]) @ now_smpl_woRoot.A[0, 25:40]
        # d = torch.linalg.inv(next_smpl_woRoot.A[0, 20]) @ next_smpl_woRoot.A[0, 25:40]            
        # e = only_finger.A[0, 25:40]            
        # f = torch.linalg.inv(next_smpl_woRoot.A[0, 21]) @ next_smpl_woRoot.A[0, 40:55]
        # g = only_finger.A[0, 40:55]
        
        # cano_xyz_next, cano_rot_next, colors_next = avatar_net.get_outputs(pose_dataset, A_next_55, A_next_woRoot_55)
        cano_xyz_next, cano_rot_next = avatar_net.get_outputs(pose_dataset, A_next_55, A_next_woRoot_55)
       
        # 원래 network 결과와 같은지 test ...
        
        # 짧은 step_dt마다 R을 바로 구하므로 차이는 무시될만하다 ?
        # kinematic하게 R 구하는거 오늘 구현 필요하다, local rot 크게 안 다르겠지... 아마 ....
        
        # 4. velocity 적용
        
        # 4.1 Avatar Velocity
        # now frame
        inv_cano_jnt_mats = torch.linalg.inv(pose_dataset.cano_smpl['A'])
        cano_xyz_now = MPM_sim.human_modify_model[list_idx].cano_xyz
        cano_rot_now = MPM_sim.human_modify_model[list_idx].cano_rot
        joint_mat_now = torch.matmul(A_now_55[0], inv_cano_jnt_mats) # [55, 4, 4]
        # joint_mat_now = MPM_sim.human_modify_model[list_idx].joint_mat # [55, 4, 4], torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
        # A_now # = MPM_sim.human_modify_model[list_idx].A_mat # [55, 4, 4], 
        pt_mats_now = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now) 
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_xyz_now) + pt_mats_now[..., :3, 3]        
        rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
        
        #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.
        # pose, 거리 오차 계산
        # if human_step % 50 == 0:
        #     # 이동 거리에 비례해 particle의 오차 비율을 측정합니다.
        #     frame_now_particle = particle_x_ori[ps + bone_index[-1] : ps+human_n_particles] # [373056, 3]
        #     A_now[0] # [22, 4, 4]
        #     now_smpl.A[0, :22]
        #     data_to_save = {
        #         'frame_now_particle': frame_now_particle,
        #         'data_now_particle': positions_now,
        #         'A_now_0': A_now[0],
        #         'now_smpl_A_0_22': now_smpl.A[0, :22]
        #     }
        #     # (now_smpl.A[0, :22] - A_now[0]).abs().max()
        #     save_dir = './test_results/'
        #     file_path = os.path.join(save_dir, f'step_{human_step:04d}.pt')
        #     if not os.path.exists(file_path):
        #         torch.save(data_to_save, file_path)
        #     else:
        #         file_path = os.path.join(save_dir, f'step_{human_step:04d}_.pt')
        #         torch.save(data_to_save, file_path)
        #     print(f"Step {human_step}의 데이터가 {file_path}에 저장되었습니다.")
            
        #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.
        
        # next_frame
        
        
        joint_mat_next = torch.matmul(A_next_55[0], inv_cano_jnt_mats)
        pt_mats_next = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_next) 
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_now) + pt_mats_next[..., :3, 3]
        rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
        # positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_next) + pt_mats_next[..., :3, 3]
        # rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_next)) # [human_N, 3, 3]
        # end3 = time.time() - end3
        # print(joint_mat_next[0])
        # print(A_next_55[0, 0])
        # MPM_sim.human_modify_model[list_idx].pt_mats_next = pt_mats_next
        # MPM_sim.human_modify_model[list_idx].cano_xyz = cano_xyz_next
        # MPM_sim.human_modify_model[list_idx].cano_rot = cano_rot_next
        # MPM_sim.human_modify_model[list_idx].joint_mat = joint_mat_next
        # MPM_sim.human_modify_model[list_idx].A_mat = A_next
        
        # 4.2 Bone Velocity
        # bone_cano  = MPM_sim.human_modify_model[list_idx].bone_cano
        # bone_index = MPM_sim.human_modify_model[list_idx].bone_index
        # smpl_index = MPM_sim.human_modify_model[list_idx].bone2smplx
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
    
        # data_dir = './AnimatableGaussians/datasets/Actor01/Sequence1'
        # data_dir = './AnimatableGaussians/datasets/Actor07/Sequence1'
        # bone_path = os.path.join(data_dir, 'osso', 'osso_per_parts', 'part_split_meshes.glb')
        # bone = trimesh.load(bone_path)
        # bone_faces = MPM_sim.human_modify_model[0].bone_faces
        # bone.geometry.items()
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]            
                
        # 4.2.1 Export Bone Model
        # save_path = "/workspace/physics/PhysGaussian_org/figure/teasor/"
        # merged_mesh = []
        # j = 0
        # for i, (key, val) in enumerate(bone.geometry.items()):
        #     if i == 7:
        #         continue
        #     bone_pose_i_1 = bone_pose1[bone_index[j] : bone_index[j+1]]
        #     bone_vertices = bone_pose_i_1.cpu().numpy()
        #     bone_centroid = bone_vertices.mean(axis=0)
        #     bone_vertices = (bone_vertices-bone_centroid) / 0.82 + bone_centroid                
        #     bone_mesh = trimesh.Trimesh(vertices=bone_vertices, faces=val.faces)
        #     bone_mesh.visual.vertex_colors = val.visual.vertex_colors[:, :3]
        #     # bone_mesh.export(os.path.join(save_path, f'{i:04d}.ply'))
        #     merged_mesh.append(bone_mesh)
        #     j += 1
        # final_mesh = trimesh.util.concatenate(merged_mesh)            
        # final_mesh.export(os.path.join(save_path, f'part_split_mesh_{human_step:04d}.ply'))
            
    
        # bone 형태 유지하는 코드 추가
        # 방법 1. bone 형태 유지하는 방향으로 velocity 추가
        # 방법 2. bone 형태 유지하는 방향으로 particle 위치 조정 (흡사 강체처럼)    
        
        # 4.3 Total Velocity
        # maintain_avatar_shape
        alpha = 0.0 # must 0, alpha is only for tgt function
        alpha2 = 0.0
        alpha2 = MPM_sim.human_modify_model[list_idx].velocity_alpha
        
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total     = torch.cat([bone_pose2, positions_next])
        
        rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

        velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]
        velocity += alpha2 * (positions_next_total - positions_now_total_sim) / smplx_dt
        # velocity  = (positions_next_total - positions_now_total_pos) / smplx_dt # [373056, 3]
        # velocity += alpha * (positions_now_total_pos - positions_now_total_sim) / smplx_dt # [373056, 3]
        
        if 0:
        # if frame % 10 == 0:
            positions_next_total_ply = trimesh.Trimesh(vertices=positions_next_total.detach().cpu().numpy())
            positions_now_total_pos_ply = trimesh.Trimesh(vertices=positions_now_total_pos.detach().cpu().numpy())
            positions_now_total_sim_ply = trimesh.Trimesh(vertices=positions_now_total_sim.detach().cpu().numpy())
            positions_next_total_ply.export("./test_results/positions_next_total_{:04d}.ply".format(frame))
            positions_now_total_pos_ply.export("./test_results/positions_now_total_{:04d}.ply".format(frame))
            positions_now_total_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
        
        # velocity = ( (positions_next_total - positions_now_total_pos) + (positions_now_total_pos - positions_now_total_sim) * alpha )/ smplx_dt # [373056, 3]
        
        relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next, torch.inverse(rot_mats_now))
        
        # velocity += (positions_next - particle_x[ps : ps + human_n_particles]) / smplx_dt * 0.001 # [373056, 3], 아바타, 뼈 유지
        # velocity += (positions_now - particle_x[ps : ps + human_n_particles]) / smplx_dt * 0.001 # [373056, 3], 아바타, 뼈 유지
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        # particle_x = trimesh.Trimesh(vertices=positions_now_total[bone_index[-1]:human_n_particles].detach().cpu().numpy())
        # particle_x.export(save_path + str(human_step) + "_positions_now.ply")
        # particle_x = trimesh.Trimesh(vertices=positions_next_total[bone_index[-1]:human_n_particles].detach().cpu().numpy())
        # particle_x.export(save_path + str(human_step) + "_positions_next.ply")
        
        # cano_xyz_now = MPM_sim.human_modify_model[list_idx].cano_xyz
        # cano_rot_now = MPM_sim.human_modify_model[list_idx].cano_rot
        # joint_mat_gt_now = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4]
        # pt_mats_gt_now = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_gt_now) 
        # positions_gt_now = torch.einsum('nxy,ny->nx', pt_mats_gt_now[..., :3, :3], cano_xyz_now) + pt_mats_gt_now[..., :3, 3]
        # positions_gt_now_ = trimesh.Trimesh(vertices=positions_gt_now.detach().cpu().numpy())
        # positions_gt_now_.export(save_path + str(human_step) + "_positions_gt_now.ply")
        
        ###############################
        # 4.4 LBS (empty bone)
        # 1. 55 -> 20
        # 1.1 smpl foot 7 = 10, 8 = 11
        # 1.2 
        # 2. avatar LBS -> bone LBS 순서 뒤섞기
        # 3. zero value to bone
        # A_now_55[0, 22:25] = A_now_55[0, 15]
        # A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40] # [4, 4] @ [15, 4, 4]
        # A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        lbs = None
        if human_step == 0:
            bone_lbs = torch.zeros(bone_verts_num, 55, device=bone_cano.device)
            lbs      = torch.cat([bone_lbs, avatar_net.lbs])
        ########################################
        # 4.5 knn_indices reg
        # v1 = positions_next.unsqueeze(1)  # (N, 1, 3)
        # v2 = positions_next[knn_indices]  # (N, k, 3)
        
        # # 유클리드 거리 계산 (N, k)
        # L_target = torch.norm(v1 - v2, dim=2, keepdim=True)[:, :, 0]  # (N, k, 1), posed gt avatar에서 인접한 indices 사이의 거리, bone 제외    
        # points = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles] 
        # grad = MPM_sim.compute_manual_gradients_parallel(points, knn_indices, L_target) # Edge Length Regularization
        # move_directions = -10*grad
        # velocity[bone_index[-1]:] += move_directions            
        ########################################
        
        velocity = torch.mm(velocity, rot_mats.T) * scale
        
        ##########################################            
        # 4.6 Final, avatar shape maintain regulzation
        # alpha = 0.1
        # velocity += alpha * vSM
        ##########################################
        # end = time.time() - end
        # relative_rot_mats = torch.matmul(rot_mats, relative_rot_mats) # [human_n_particles, 3, 3]
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    ## 뼈대가 너무 크긴하고, 조금 튀어나온다
    # bone 위치따라서 step_dt당 avatar의 velocity 계산하는 코드 추가, 얼마나 보정을 할까 
    # 새로운 human velocity 계산이랑 / avatar 형태 보존이 velocity는 각각 따로다        
    # global rotation은 그렇다 치는데, 충돌하고 나서 global translation이 있는거는 조금 이상하긴 하다, 일단 하자
    
    # joints = torch.unsqueeze(now_smpl.J, dim=-1)
    # joints_homogen = F.pad(joints, [0, 0, 0, 1])
    # transforms = now_smpl.A[0].clone()
    # transforms[:, :3, 3] = now_smpl.joints[0, :55]        
    # # now_smpl.A[0]랑 같다 !!, A matrix에서 transform 만들기 가능
    # rel_transforms = transforms - F.pad(
    #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
    # # rel_transforms[0, :, :3, :3] + posed_joints = transforms      ## rel_transforms -> transforms
    
    # print()
    ############################################################################################################################
    
    if is_3d_measure and frame % 10 == 0 :
        positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
        
        cano_xyz_now_gt, _ = avatar_net.get_outputs(pose_dataset, now_smpl.A, now_smpl_woRoot.A)
        joint_mat_now_gt = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats)
        pt_mats_now_gt = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now_gt) 
        positions_now_gt = torch.einsum('nxy,ny->nx', pt_mats_now_gt[..., :3, :3], cano_xyz_now_gt) + pt_mats_now_gt[..., :3, 3]
        
        # positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
        # positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
        # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
        # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
        
        diff = positions_now_sim - positions_now_gt
        dists = torch.linalg.norm(diff, axis=1)
        
        tau1 = 0.01
        tau2 = 0.02
        tau3 = 0.03
        mae = torch.mean(dists)
        rmse = torch.sqrt(torch.mean(dists ** 2))
        acc1 = torch.mean((dists < tau1).float())
        acc2 = torch.mean((dists < tau2).float())
        acc3 = torch.mean((dists < tau3).float())
        
        print("frame", frame, 
            "mean", mae,
            "rmse", rmse,
            "acc1", acc1,
            "acc2", acc2,
            "acc3", acc3,
            )
    
    test_code = False
    if test_code:
        try:
            gt_cano_xyz_now, _ = avatar_net.get_outputs(pose_dataset, now_smpl.A, now_smpl_woRoot.A)
            gt_cano_xyz_next, _= avatar_net.get_outputs(pose_dataset, next_smpl.A, next_smpl_woRoot.A)        
            joint_mat1 = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats)
            pt_mats1 = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat1) 
            positions1 = torch.einsum('nxy,ny->nx', pt_mats1[..., :3, :3], gt_cano_xyz_now) + pt_mats1[..., :3, 3]        
            joint_mat2 = torch.matmul(next_smpl.A[0], inv_cano_jnt_mats)
            pt_mats2 = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat2) 
            positions2 = torch.einsum('nxy,ny->nx', pt_mats2[..., :3, :3], gt_cano_xyz_next) + pt_mats2[..., :3, 3]     
            
            bone_verts_num = bone_cano.shape[0]
            bone_pose_now_gt = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
            bone_pose_next_gt = torch.zeros(bone_verts_num, 3, device=bone_cano.device)        
            for i in range(len(bone_index)-1):
                bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ now_smpl.A[0, bone2smplx[i], :3, :3].T + now_smpl.A[0, bone2smplx[i], :3, 3]
                bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ next_smpl.A[0, bone2smplx[i], :3, :3].T + next_smpl.A[0, bone2smplx[i], :3, 3]
                bone_pose_now_gt[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
                bone_pose_next_gt[bone_index[i] : bone_index[i+1]] = bone_pose_i_2
            positions_now_gt  = torch.cat([bone_pose_now_gt, positions1])
            positions_next_gt = torch.cat([bone_pose_next_gt, positions2])
            
            velocity_gt = (positions_next_gt - positions_now_gt) / smplx_dt # [373056, 3]
            velocity_gt = torch.mm(velocity_gt, rot_mats.T) * scale
            relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next, torch.inverse(rot_mats_now))
            
            MPM_sim.human_modify_model[list_idx].check_A_now.append(A_now_55)
            MPM_sim.human_modify_model[list_idx].check_A_next.append(A_next_55)
            MPM_sim.human_modify_model[list_idx].check_A_next_woRoot.append(A_next_woRoot_55)
            MPM_sim.human_modify_model[list_idx].check_velocity.append(velocity[bone_index[-1]:])
            # MPM_sim.human_modify_model[list_idx].check_bone_now.append(particle_x[ps+bone_index[0]:ps+bone_index[-1]])
            
            now_smpl_A = now_smpl.A; next_smpl_A = next_smpl.A; next_smpl_woRoot_A = next_smpl_woRoot.A
            MPM_sim.human_modify_model[list_idx].gt_A_now.append(now_smpl_A)
            MPM_sim.human_modify_model[list_idx].gt_A_next.append(next_smpl_A)
            MPM_sim.human_modify_model[list_idx].gt_A_next_woRoot.append(next_smpl_woRoot_A)        
            MPM_sim.human_modify_model[list_idx].gt_velocity.append(velocity_gt)
            
            err_a = (A_now_55 - now_smpl_A).abs().max()                        # 1.65e-06
            err_b = (A_next_55 - next_smpl_A).abs().max()                      # 1.65e-06
            err_c = (A_next_woRoot_55 - next_smpl_woRoot_A).abs().max()        # 1.69e-06
            err_d = (positions1 - positions_now[bone_index[-1]:]).abs().max()  # 6.43e-06, avatar
            err_e = (positions2 - positions_next[bone_index[-1]:]).abs().max() # 1.04e-05
            err_f = (velocity - velocity_gt)[:bone_index[-1]].abs().max()      # bone err
            err_g = (velocity - velocity_gt)[bone_index[-1]:].abs().max()      # avatar err
        except:
            pass
        
    # A_next_55_local = (torch.mm(A_next_55[0], rot_mats.T) - ori_mean) * scale + center
    # joint_mat_next = torch.matmul(A_next_55[0], inv_cano_jnt_mats)
    
    rot_mats44 = torch.eye(4, device=MPM_sim.device); rot_mats44[:3, :3] = rot_mats
    
    A_next_55_local = torch.matmul(A_next_55[0], rot_mats44.T)
    A_next_55_local[:, :3, 3] = (A_next_55_local[:, :3, 3] - ori_mean) * scale + center # [55, 4, 4]
    
    joint_mat_next_local = torch.matmul(joint_mat_next, rot_mats44.T)
    joint_mat_next_local[:, :3, 3] = (joint_mat_next_local[:, :3, 3] - ori_mean) * scale + center # [55, 4, 4]
         
    # print("end : ", end) # 전체
    # print("end2 : ", end2) # kabsch
    # print("end3 : ", end3) # ag
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM, rot_mats_next_total #, joint_mat_next_local, A_next_55 # , torch.flip(colors_next, dims=[1])

@torch.no_grad()
def compute_animatable_gaussians_velocity_tgt(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None):
    # list_idx -> human_idx
    
    # 1. 뼈대 움직임에서 keypoint R, t 계산 -> 현재 global A-pose matrix 계산
    # 1.1 이때 vector 2개로 R 구하는 방법이 필요할 수도 있다. (contribution 2)
    # torch
    avatar_net = MPM_sim.human_modify_model[list_idx].avatar_net
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    knn_indices = MPM_sim.human_modify_model[list_idx].knn_indices
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center
    
    A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
            
    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
    
    # temp test
    kabsch_A = torch.zeros_like(A_now)
    kabsch_A[:, 3, 3] = 1.0
    
    for i in range(len(bone_index)-1):
        # time1 = time.time()
        R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
        
        kabsch_A[bone2smplx[i], :3, :3] = R_est # temp test
        kabsch_A[bone2smplx[i], :3,  3] = t_est # temp test
        
        joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
        # 이거 맞는지 어떻게 확인?
        # joint_gt  = pose_J[bone2smplx[i]]
        A_now[bone2smplx[i], :3, :3] = R_est
        A_now[bone2smplx[i], :3, 3] = joint_cal
        if bone2smplx[i] == 7 or bone2smplx[i] == 8: # smpl foot 7 = 10, 8 = 11
            A_now[bone2smplx[i]+3, :3, :3] = R_est
            A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
            kabsch_A[bone2smplx[i]+3, :3, :3] = R_est # temp test
            kabsch_A[bone2smplx[i]+3, :3,  3] = t_est # temp test
        
    A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
    A_now = A_now.unsqueeze(0)
    
    # 2. 현재 global A-pose matrix 에서 다음 global A-pose matrix 계산
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        # now_frame = pose_dataset.pose_list[1]
        # next_frame = pose_dataset.pose_list[1]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        now_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        
        T_pred = A_now[0]
        T_gt = now_smpl.A[0, :22]
        parents = pose_dataset.smpl_model.parents[:22]
        
        if f is not None and frame % 100 == 0:
            rot_err_deg, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
            rot_err_deg_mean = rot_err_deg.mean()
            trans_err_mean = trans_err.mean()
            
            f.write(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f} \n")
        
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]           
        now_transl = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Get transl from root A & J             
        
        # Get transl from root A & J
        # temp = torch.eye(4, device='cuda')
        # temp[:3, :3] = A_now[0,0,:3,:3]
        # temp[:3, 3] = joints[0, 0, :, 0]
        # temp[:3, 3] -= (temp[:3, :3] @ joints[0, 0])[:, 0]        
        # transl = A_now[0,0] - temp
        # pose_dataset.transl[now_frame][None]
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        
        rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(A_now[0, :22, :3, :3]) # 여기서 각도 clamping        
        rel_local_rot = get_local_rot(rel_rot, A_now[0, :22, :3, :3], parents)        
        rel_local_rot_clamped = clamp_rel_rot_threshold(rel_local_rot, max_angle_deg=10.0)        
        rel_rot_new = recompose_global_rel_rot_if_needed(rel_rot, rel_local_rot, rel_local_rot_clamped, A_now[0, :22, :3, :3], parents)        
        
        # 2.2 다음 frame의 global rotation 계산
        # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        # transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        transforms[:, :3, :3] = rel_rot_new @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        
        # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        for i in range(1, parents.shape[0]):
            transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        
        max_dist = 0.05
        delta = next_transl - now_transl
        dist = torch.norm(delta, dim=-1, keepdim=True) + 1e-9
        if dist > max_dist: 
            next_transl = now_transl + delta * torch.clamp(max_dist / dist, max=1.0)  # clamp 적용된 next
            # print("clamp transl")
        rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        
        A_next = rel_transforms 
        # A_next = next_smpl.A[]
        ##############################################################################################################        
                
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]                      
        cano_xyz_next, cano_rot_next = avatar_net.get_outputs(pose_dataset, A_next_55, A_next_woRoot_55)
                
        # 4. velocity        
        # 4.1 Avatar Velocity
        # now frame
        inv_cano_jnt_mats = torch.linalg.inv(pose_dataset.cano_smpl['A'])
        cano_xyz_now = MPM_sim.human_modify_model[list_idx].cano_xyz
        cano_rot_now = MPM_sim.human_modify_model[list_idx].cano_rot
        joint_mat_now = torch.matmul(A_now_55[0], inv_cano_jnt_mats)
        pt_mats_now = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now) 
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_xyz_now) + pt_mats_now[..., :3, 3]        
        rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
                
        # next_frame
        joint_mat_next = torch.matmul(A_next_55[0], inv_cano_jnt_mats)
        pt_mats_next = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_next) 
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_next) + pt_mats_next[..., :3, 3]
        rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_next)) # [human_N, 3, 3]
        
        MPM_sim.human_modify_model[list_idx].pt_mats_next = pt_mats_next
        MPM_sim.human_modify_model[list_idx].cano_xyz = cano_xyz_next
        MPM_sim.human_modify_model[list_idx].cano_rot = cano_rot_next
        
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
    
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]            
            
        # 4.2.1 Export Bone Model        
        # save_path = "./Qualitative/h1o1_sandback1_skel/skel/"
        # os.makedirs(save_path, exist_ok=True)
        # merged_mesh = []
        # j = 0
        # for i, (key, val) in enumerate(bone.geometry.items()):
        #     if i == 7:
        #         continue
        #     bone_pose_i_1 = bone_pose1[bone_index[j] : bone_index[j+1]]
        #     bone_vertices = bone_pose_i_1.cpu().numpy()
        #     bone_centroid = bone_vertices.mean(axis=0)
        #     bone_vertices = (bone_vertices-bone_centroid) / 0.82 + bone_centroid                
        #     bone_mesh = trimesh.Trimesh(vertices=bone_vertices, faces=val.faces)
        #     bone_mesh.visual.vertex_colors = val.visual.vertex_colors[:, :3]
        #     # bone_mesh.export(os.path.join(save_path, f'{i:04d}.ply'))
        #     merged_mesh.append(bone_mesh)
        #     j += 1
        # final_mesh = trimesh.util.concatenate(merged_mesh)            
        # final_mesh.export(os.path.join(save_path, f'part_split_mesh_{human_step:04d}.ply'))    
        
        # 4.3 Total Velocity
        # maintain_avatar_shape
        alpha = 0.0
        alpha = MPM_sim.human_modify_model[list_idx].velocity_alpha
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

        velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]        
        relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        lbs = None
                
        velocity = torch.mm(velocity, rot_mats.T) * scale        
        
        if is_3d_measure and frame % 10 == 0 :
            positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
            
            now_smpl_woRoot = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
                                                    body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                    # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                    # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                    left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                    right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
            )            
            cano_xyz_now_gt, _ = avatar_net.get_outputs(pose_dataset, now_smpl.A, now_smpl_woRoot.A)
            joint_mat_now_gt = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats)
            pt_mats_now_gt = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now_gt) 
            positions_now_gt = torch.einsum('nxy,ny->nx', pt_mats_now_gt[..., :3, :3], cano_xyz_now_gt) + pt_mats_now_gt[..., :3, 3]
            
            # positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
            # positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
            # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
            # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
            
            diff = positions_now_sim - positions_now_gt
            dists = torch.linalg.norm(diff, axis=1)
            
            tau1 = 0.01
            tau2 = 0.02
            tau3 = 0.03
            mae = torch.mean(dists)
            rmse = torch.sqrt(torch.mean(dists ** 2))
            acc1 = torch.mean((dists < tau1).float())
            acc2 = torch.mean((dists < tau2).float())
            acc3 = torch.mean((dists < tau3).float())
            
            print("frame", frame, 
                "mean", mae,
                "rmse", rmse,
                "acc1", acc1,
                "acc2", acc2,
                "acc3", acc3,
                )
        
        if 0:
            # particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
            
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_now_mesh.ply')
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_next_total, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_next_mesh.ply')
            print('export mesh')
        
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM, rot_mats_next_total #, joint_mat_next_local, A_next_55 # , torch.flip(colors_next, dims=[1])


@torch.no_grad()
def compute_animatable_gaussians_velocity_gt(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None):
    # list_idx -> human_idx
    
    # 1. 뼈대 움직임에서 keypoint R, t 계산 -> 현재 global A-pose matrix 계산
    # 1.1 이때 vector 2개로 R 구하는 방법이 필요할 수도 있다. (contribution 2)
    # torch
    avatar_net = MPM_sim.human_modify_model[list_idx].avatar_net
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    knn_indices = MPM_sim.human_modify_model[list_idx].knn_indices
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center
    
    A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
            
    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
    
    # temp test
    kabsch_A = torch.zeros_like(A_now)
    kabsch_A[:, 3, 3] = 1.0
    
    for i in range(len(bone_index)-1):
        # time1 = time.time()
        R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
        
        kabsch_A[bone2smplx[i], :3, :3] = R_est # temp test
        kabsch_A[bone2smplx[i], :3,  3] = t_est # temp test
        
        joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
        # 이거 맞는지 어떻게 확인?
        # joint_gt  = pose_J[bone2smplx[i]]
        A_now[bone2smplx[i], :3, :3] = R_est
        A_now[bone2smplx[i], :3, 3] = joint_cal
        if bone2smplx[i] == 7 or bone2smplx[i] == 8: # smpl foot 7 = 10, 8 = 11
            A_now[bone2smplx[i]+3, :3, :3] = R_est
            A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
            kabsch_A[bone2smplx[i]+3, :3, :3] = R_est # temp test
            kabsch_A[bone2smplx[i]+3, :3,  3] = t_est # temp test
        
    A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
    A_now = A_now.unsqueeze(0)
    
    # 2. 현재 global A-pose matrix 에서 다음 global A-pose matrix 계산
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        # now_frame = pose_dataset.pose_list[1]
        # next_frame = pose_dataset.pose_list[1]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        now_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        
        T_pred = A_now[0]
        T_gt = now_smpl.A[0, :22]
        parents = pose_dataset.smpl_model.parents[:22]
        
        # rot_err, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
        # rot_err_deg_mean = rot_err_deg.mean()
        # trans_err_mean = trans_err.mean()
        
        if f is not None and frame % 100 == 0:
            f.write(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f} \n")
        
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]           
        now_transl = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Get transl from root A & J             
        
        # Get transl from root A & J
        # temp = torch.eye(4, device='cuda')
        # temp[:3, :3] = A_now[0,0,:3,:3]
        # temp[:3, 3] = joints[0, 0, :, 0]
        # temp[:3, 3] -= (temp[:3, :3] @ joints[0, 0])[:, 0]        
        # transl = A_now[0,0] - temp
        # pose_dataset.transl[now_frame][None]
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        
        A_next = next_smpl.A[:, :22]
        
        # rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(A_now[0, :22, :3, :3]) # 여기서 각도 clamping        
        # rel_local_rot = get_local_rot(rel_rot, A_now[0, :22, :3, :3], parents)        
        # rel_local_rot_clamped = clamp_rel_rot_threshold(rel_local_rot, max_angle_deg=10.0)        
        # rel_rot_new = recompose_global_rel_rot_if_needed(rel_rot, rel_local_rot, rel_local_rot_clamped, A_now[0, :22, :3, :3], parents)        
        
        # # 2.2 다음 frame의 global rotation 계산
        # # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        # transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        # # transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        # transforms[:, :3, :3] = rel_rot_new @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        
        # # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        # transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        # for i in range(1, parents.shape[0]):
        #     transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        # rel_transforms = transforms - F.pad(
        #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        # # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        
        # max_dist = 0.05
        # delta = next_transl - now_transl
        # dist = torch.norm(delta, dim=-1, keepdim=True) + 1e-9
        # if dist > max_dist: 
        #     next_transl = now_transl + delta * torch.clamp(max_dist / dist, max=1.0)  # clamp 적용된 next
        #     # print("clamp transl")
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        
        # A_next = rel_transforms 
        # A_next = next_smpl.A[]
        ##############################################################################################################        
                
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]                      
        cano_xyz_next, cano_rot_next = avatar_net.get_outputs(pose_dataset, A_next_55, A_next_woRoot_55)
                
        # 4. velocity        
        # 4.1 Avatar Velocity
        # now frame
        inv_cano_jnt_mats = torch.linalg.inv(pose_dataset.cano_smpl['A'])
        cano_xyz_now = MPM_sim.human_modify_model[list_idx].cano_xyz
        cano_rot_now = MPM_sim.human_modify_model[list_idx].cano_rot
        joint_mat_now = torch.matmul(A_now_55[0], inv_cano_jnt_mats)
        pt_mats_now = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now) 
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_xyz_now) + pt_mats_now[..., :3, 3]        
        rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
                
        # next_frame
        joint_mat_next = torch.matmul(A_next_55[0], inv_cano_jnt_mats)
        pt_mats_next = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_next) 
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_next) + pt_mats_next[..., :3, 3]
        rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_next)) # [human_N, 3, 3]
        
        MPM_sim.human_modify_model[list_idx].pt_mats_next = pt_mats_next
        MPM_sim.human_modify_model[list_idx].cano_xyz = cano_xyz_next
        MPM_sim.human_modify_model[list_idx].cano_rot = cano_rot_next
        
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
    
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]            
            
        # 4.2.1 Export Bone Model        
        # save_path = "./Qualitative/h1o1_sandback1_skel/skel/"
        # os.makedirs(save_path, exist_ok=True)
        # merged_mesh = []
        # j = 0
        # for i, (key, val) in enumerate(bone.geometry.items()):
        #     if i == 7:
        #         continue
        #     bone_pose_i_1 = bone_pose1[bone_index[j] : bone_index[j+1]]
        #     bone_vertices = bone_pose_i_1.cpu().numpy()
        #     bone_centroid = bone_vertices.mean(axis=0)
        #     bone_vertices = (bone_vertices-bone_centroid) / 0.82 + bone_centroid                
        #     bone_mesh = trimesh.Trimesh(vertices=bone_vertices, faces=val.faces)
        #     bone_mesh.visual.vertex_colors = val.visual.vertex_colors[:, :3]
        #     # bone_mesh.export(os.path.join(save_path, f'{i:04d}.ply'))
        #     merged_mesh.append(bone_mesh)
        #     j += 1
        # final_mesh = trimesh.util.concatenate(merged_mesh)            
        # final_mesh.export(os.path.join(save_path, f'part_split_mesh_{human_step:04d}.ply'))    
        
        # 4.3 Total Velocity
        # maintain_avatar_shape
        alpha = 0.0
        alpha = MPM_sim.human_modify_model[list_idx].velocity_alpha
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

        velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]        
        relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        lbs = None
                
        velocity = torch.mm(velocity, rot_mats.T) * scale        
        
        if is_3d_measure and frame % 10 == 0 :
            positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
            
            now_smpl_woRoot = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
                                                    body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                    # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                    # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                    left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                    right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
            )            
            cano_xyz_now_gt, _ = avatar_net.get_outputs(pose_dataset, now_smpl.A, now_smpl_woRoot.A)
            joint_mat_now_gt = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats)
            pt_mats_now_gt = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now_gt) 
            positions_now_gt = torch.einsum('nxy,ny->nx', pt_mats_now_gt[..., :3, :3], cano_xyz_now_gt) + pt_mats_now_gt[..., :3, 3]
            
            # positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
            # positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
            # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
            # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
            
            diff = positions_now_sim - positions_now_gt
            dists = torch.linalg.norm(diff, axis=1)
            
            tau1 = 0.01
            tau2 = 0.02
            tau3 = 0.03
            mae = torch.mean(dists)
            rmse = torch.sqrt(torch.mean(dists ** 2))
            acc1 = torch.mean((dists < tau1).float())
            acc2 = torch.mean((dists < tau2).float())
            acc3 = torch.mean((dists < tau3).float())
            
            print("frame", frame, 
                "mean", mae,
                "rmse", rmse,
                "acc1", acc1,
                "acc2", acc2,
                "acc3", acc3,
                )
        
        if 0:
            # particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
            
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_now_mesh.ply')
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_next_total, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_next_mesh.ply')
            print('export mesh')
        
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM #, joint_mat_next_local, A_next_55 # , torch.flip(colors_next, dims=[1])


@torch.no_grad()
def compute_animatable_gaussians_velocity_hp(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False):
    # list_idx -> human_idx
    
    # 1. 뼈대 움직임에서 keypoint R, t 계산 -> 현재 global A-pose matrix 계산
    # 1.1 이때 vector 2개로 R 구하는 방법이 필요할 수도 있다. (contribution 2)
    # torch
    avatar_net = MPM_sim.human_modify_model[list_idx].avatar_net
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    knn_indices = MPM_sim.human_modify_model[list_idx].knn_indices
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center
    
    A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
            
    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
    
    # temp test
    kabsch_A = torch.zeros_like(A_now)
    kabsch_A[:, 3, 3] = 1.0
    
    for i in range(len(bone_index)-1):
        # time1 = time.time()
        R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
        
        kabsch_A[bone2smplx[i], :3, :3] = R_est # temp test
        kabsch_A[bone2smplx[i], :3,  3] = t_est # temp test
        
        joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
        # 이거 맞는지 어떻게 확인?
        # joint_gt  = pose_J[bone2smplx[i]]
        A_now[bone2smplx[i], :3, :3] = R_est
        A_now[bone2smplx[i], :3, 3] = joint_cal
        if bone2smplx[i] == 7 or bone2smplx[i] == 8: # smpl foot 7 = 10, 8 = 11
            A_now[bone2smplx[i]+3, :3, :3] = R_est
            A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
            kabsch_A[bone2smplx[i]+3, :3, :3] = R_est # temp test
            kabsch_A[bone2smplx[i]+3, :3,  3] = t_est # temp test
        
    A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
    A_now = A_now.unsqueeze(0)
    
    # 2. 현재 global A-pose matrix 에서 다음 global A-pose matrix 계산
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        # now_frame = pose_dataset.pose_list[1]
        # next_frame = pose_dataset.pose_list[1]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        now_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                                transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                                body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None],
                                                # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                                # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                                # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]           
        now_transl = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Get transl from root A & J             
        
        # Get transl from root A & J
        # temp = torch.eye(4, device='cuda')
        # temp[:3, :3] = A_now[0,0,:3,:3]
        # temp[:3, 3] = joints[0, 0, :, 0]
        # temp[:3, 3] -= (temp[:3, :3] @ joints[0, 0])[:, 0]        
        # transl = A_now[0,0] - temp
        # pose_dataset.transl[now_frame][None]
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        
        rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(A_now[0, :22, :3, :3]) # 여기서 각도 clamping        
        # rel_local_rot = get_local_rot(rel_rot, A_now[0, :22, :3, :3], parents)        
        # rel_local_rot_clamped = clamp_rel_rot_threshold(rel_local_rot, max_angle_deg=10.0)        
        # rel_rot_new = recompose_global_rel_rot_if_needed(rel_rot, rel_local_rot, rel_local_rot_clamped, A_now[0, :22, :3, :3], parents)        
        
        # 2.2 다음 frame의 global rotation 계산
        # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        # transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        
        # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        for i in range(1, parents.shape[0]):
            transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        
        max_dist = 0.05
        delta = next_transl - now_transl
        dist = torch.norm(delta, dim=-1, keepdim=True) + 1e-9
        if dist > max_dist: 
            next_transl = now_transl + delta * torch.clamp(max_dist / dist, max=1.0)  # clamp 적용된 next
            print("clamp transl")
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        rel_transforms[0, :, :3, 3] += pose_dataset.transl[next_frame] - pose_dataset.transl[now_frame] + now_transl # [1, 22, 4, 4]
        
        A_next = rel_transforms 
        ##############################################################################################################        
                
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]                      
        cano_xyz_next, cano_rot_next = avatar_net.get_outputs(pose_dataset, A_next_55, A_next_woRoot_55)
                
        # 4. velocity        
        # 4.1 Avatar Velocity
        # now frame
        inv_cano_jnt_mats = torch.linalg.inv(pose_dataset.cano_smpl['A'])
        cano_xyz_now = MPM_sim.human_modify_model[list_idx].cano_xyz
        cano_rot_now = MPM_sim.human_modify_model[list_idx].cano_rot
        joint_mat_now = torch.matmul(A_now_55[0], inv_cano_jnt_mats)
        pt_mats_now = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now) 
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_xyz_now) + pt_mats_now[..., :3, 3]        
        rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
                
        # next_frame
        joint_mat_next = torch.matmul(A_next_55[0], inv_cano_jnt_mats)
        pt_mats_next = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_next) 
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_next) + pt_mats_next[..., :3, 3]
        rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_next)) # [human_N, 3, 3]
        
        MPM_sim.human_modify_model[list_idx].pt_mats_next = pt_mats_next
        MPM_sim.human_modify_model[list_idx].cano_xyz = cano_xyz_next
        MPM_sim.human_modify_model[list_idx].cano_rot = cano_rot_next
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
    
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]            
            
        # 4.3 Total Velocity
        # maintain_avatar_shape
        alpha = 0.2
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

        velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]        
        relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        lbs = None
                
        velocity = torch.mm(velocity, rot_mats.T) * scale        
        
        if is_3d_measure and frame % 10 == 0 :
            positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
            
            now_smpl_woRoot = pose_dataset.smpl_model.forward(betas = pose_dataset.smpl_shape[None], 
                                                    body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                                    # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                                    # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                                    left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                    right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
            )            
            cano_xyz_now_gt, _ = avatar_net.get_outputs(pose_dataset, now_smpl.A, now_smpl_woRoot.A)
            joint_mat_now_gt = torch.matmul(now_smpl.A[0], inv_cano_jnt_mats)
            pt_mats_now_gt = torch.einsum('nj,jxy->nxy', avatar_net.lbs, joint_mat_now_gt) 
            positions_now_gt = torch.einsum('nxy,ny->nx', pt_mats_now_gt[..., :3, :3], cano_xyz_now_gt) + pt_mats_now_gt[..., :3, 3]
            
            # positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
            # positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
            # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
            # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
            
            diff = positions_now_sim - positions_now_gt
            dists = torch.linalg.norm(diff, axis=1)
            tau1 = 0.01
            tau2 = 0.02
            tau3 = 0.03
            mae = torch.mean(dists)
            rmse = torch.sqrt(torch.mean(dists ** 2))
            acc1 = torch.mean((dists < tau1).float())
            acc2 = torch.mean((dists < tau2).float())
            acc3 = torch.mean((dists < tau3).float())
            
            print("frame", frame, 
                "mean", mae,
                "rmse", rmse,
                "acc1", acc1,
                "acc2", acc2,
                "acc3", acc3,
                )
            
            if f is not None:
                f.write(f"frame: {frame}, mean: {mae.item():.6f}, rmse: {rmse.item():.6f}, acc1: {acc1.item():.6f}, acc2: {acc2.item():.6f}, acc3: {acc3.item():.6f} \n")
        
        if 0:
            # particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
            
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_now_mesh.ply')
            temp_mesh = trimesh.Trimesh(
                vertices=((torch.mm(positions_next_total, rot_mats.T) - ori_mean) * scale + center).detach().cpu().numpy()
            )
            temp_mesh.export('./test_results/newvel_next_mesh.ply')
            print('export mesh')
        
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM #, joint_mat_next_local, A_next_55 # , torch.flip(colors_next, dims=[1])


###################################################################################################################################################

def modify_smplx(MPM_sim, model_type, velocity_type, velocity_alpha, pose_dataset, betas, smplx_model, cano_pts, cano_J,
    bone_cano, bone_index, particle_start, index, rot_mats, ori_mean, scale, center, g_time, device="cuda"
):
    # 1.
    human_model = HumanTorchModel()
    human_model.model_type = model_type
    human_n_particles = bone_index[-1] + cano_pts.shape[0]
    
    human_model.index = index
    human_model.avatar_index = slice(bone_index[-1], human_n_particles) # particle_start 반영 필요
    human_model.pose_dataset = pose_dataset
    human_model.particle_start = particle_start
    human_model.cano_J = cano_J
    
    human_model.bone_cano = bone_cano # [74496, 3]
    human_model.bone_index = bone_index
    human_model.bone2smplx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1, 4, 2, 5, 7, 8]
    
    human_model.rot_mats = rot_mats # bc params
    human_model.ori_mean = ori_mean
    human_model.scale = scale
    human_model.center = center
    human_model.human_n_particles = human_n_particles              # [Human Particles]    
    
    human_model.betas = betas
    human_model.smplx_model = smplx_model
    human_model.velocity_type = velocity_type
    human_model.velocity_alpha = velocity_alpha
    # human_model.g_frame = g_time[0]
    # human_model.g = torch.tensor(g_time[1:], device=device)
    
    MPM_sim.human_modify_model.append(human_model) # torch velocity
    
    # 2.
    # class HumanModifier:
    #     particle_id: wp.array(dtype=int)
    human_params = HumanModifier()
    human_params.g = wp.vec3(g_time[1], g_time[2], g_time[3])    
    human_params.g_frame = g_time[0]
    _human_idx_np = np.where(MPM_sim.mpm_state.particle_id.numpy() == index)[0]
    human_params.particle_id = wp.array(_human_idx_np, dtype=int) # MPM particle indices for this human (NOT contiguous when interior is appended after another subject)
    # store torch version on human_model so compute_*_velocity can index particle_x correctly
    MPM_sim.human_modify_model[-1].particle_id_torch = torch.from_numpy(_human_idx_np).long().to(device)
    MPM_sim.human_modify_params.append(human_params) # warp velocity
    
    # 3. 
    @wp.kernel
    def kinematic_velocity(
        state: MPMStateStruct,
        human_params: HumanModifier,
        kinematic_v: wp.array(dtype=wp.vec3), 
        relR: wp.array(dtype=wp.mat33),
        apply_rot: int,
        vSM: wp.array(dtype=wp.vec3),
        frame: int
    ): # dim = human_n_particles (avatar + bone)
        p = wp.tid()
        id = human_params.particle_id[p]
        state.particle_v[id]   = state.particle_v[id] + kinematic_v[p] - state.particle_vk[id]
        state.particle_vk[id]  = kinematic_v[p]
        state.particle_vko[id] = kinematic_v[p]
        # state.particle_F_trial[id] = relR[p] * state.particle_F_trial[id]
        # state.particle_F_trial[id] = overwrite_R_to_F(state.particle_F_trial[id], rot_mats_next[p])        
        
        if frame == human_params.g_frame:
            state.particle_gravity[id] = human_params.g
        # state.particle_F[id] = relR[p] * state.particle_F[id]
        
        # bid = state.bone_idx[p]
        # if bid < 0: # if not bone(if avatar)
        # state.particle_vSM[p] = vSM[p]
        
    MPM_sim.human_modify_changer.append(kinematic_velocity)
    
    # 4. apply particle bone index
    ps = particle_start
    particle_bone_val = np.zeros(bone_index[-1], dtype=np.int16) # [74496]
    for i in range(len(bone_index)-1):
        particle_bone_val[ bone_index[i] : bone_index[i+1] ] = i
    
    state_bone_idx = MPM_sim.mpm_state.bone_idx.numpy()
    state_bone_idx[ps:ps+bone_index[-1]] = particle_bone_val
    MPM_sim.mpm_state.bone_idx = wp.array(state_bone_idx, dtype=wp.int16, device=device)
    
    # 5. save bone cano
    bone_cano_torch = (torch.mm(bone_cano, rot_mats.T) - ori_mean) * scale + center # GT2Sim coordinate system
    bone_cano_wp    = torch2warp_vec3(bone_cano_torch, dvc=device)
    
    bone_mass_torch = wp.to_torch(MPM_sim.mpm_state.particle_mass)[ps:ps+bone_index[-1]]
    x_splits = torch.split(bone_cano_torch, MPM_sim.bone_p_num.tolist(), dim=0)
    m_splits = torch.split(bone_mass_torch, MPM_sim.bone_p_num.tolist(), dim=0)
    
    bone_cano_c_torch = torch.stack([chunk.mean(dim=0) for chunk in x_splits], dim=0) # [20, 3]
    bone_cano_c_wp    = torch2warp_vec3(bone_cano_c_torch, dvc=device)
    
    bone_cano_q_torch = bone_cano_torch - bone_cano_c_torch[particle_bone_val]
    bone_cano_q_wp    = torch2warp_vec3(bone_cano_q_torch, dvc=device)

    @wp.kernel
    def particle_bone_cano(arr: wp.array(dtype=wp.vec3, ndim=2), idx: int, val: wp.array(dtype=wp.vec3)):
        p = wp.tid()
        arr[idx, p] = val[p]
    
    wp.launch(
        kernel=particle_bone_cano,
        dim=bone_index[-1], # 74496
        inputs=[MPM_sim.mpm_state.bone_x0, index, bone_cano_wp],
        device=device,
    )   
    wp.launch(
        kernel=particle_bone_cano,
        dim=20, # 20
        inputs=[MPM_sim.mpm_state.bone_x0cm, index, bone_cano_c_wp],
        device=device,
    )
    wp.launch(
        kernel=particle_bone_cano,
        dim=bone_index[-1], # 20
        inputs=[MPM_sim.mpm_state.bone_q, index, bone_cano_q_wp],
        device=device,
    )
            
    @wp.kernel
    def set_bone_E_nu(
        state: MPMStateStruct, 
        model: MPMModelStruct
    ):
        p = wp.tid()
        bone_idx = state.bone_idx[p]
        if bone_idx >= 0:
            E = 1e5
            nu = 0.3
            model.E[p] = E
            model.nu[p] = nu
            model.mu[p] = E / (2.0 * (1.0 + nu))
            model.lam[p] = (
                E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            )
    
    wp.launch(
        kernel=set_bone_E_nu,
        dim=MPM_sim.n_particles,
        inputs=[MPM_sim.mpm_state, MPM_sim.mpm_model],
        device=device,
    )
    
    # For shape matching
    offset_np = MPM_sim.mpm_state.avatar_offset.numpy()
    last_offset = offset_np[-1] + human_n_particles
    new_offset_np = np.concatenate([offset_np, [last_offset]]).astype(np.int32)
    MPM_sim.mpm_state.avatar_offset = wp.array(new_offset_np, dtype=wp.int32, device=device)

@torch.no_grad()
def compute_smplx_velocity_rel(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None, sim_params=None):
    
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center    
    
    smplx_model = MPM_sim.human_modify_model[list_idx].smplx_model
    betas = MPM_sim.human_modify_model[list_idx].betas
    lbs = smplx_model.lbs_weights
    
    A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
    
    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
    
    kabsch_A = torch.zeros_like(A_now)
    kabsch_A[:, 3, 3] = 1.0
    for i in range(len(bone_index)-1):
        # time1 = time.time()
        R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
        
        kabsch_A[bone2smplx[i], :3, :3] = R_est # temp test
        kabsch_A[bone2smplx[i], :3,  3] = t_est # temp test
        
        joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
        # 이거 맞는지 어떻게 확인?
        # joint_gt  = pose_J[bone2smplx[i]]
        A_now[bone2smplx[i], :3, :3] = R_est
        A_now[bone2smplx[i], :3, 3] = joint_cal
        if bone2smplx[i] == 7 or bone2smplx[i] == 8: # smpl foot 7 = 10, 8 = 11
            A_now[bone2smplx[i]+3, :3, :3] = R_est
            A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
            kabsch_A[bone2smplx[i]+3, :3, :3] = R_est # temp test
            kabsch_A[bone2smplx[i]+3, :3,  3] = t_est # temp test
        # time1 = time.time() - time1; print(time1*1000, "ms")
        
        # bone_cano[bone_index[i]:bone_index[i+1]].detach().cpu().numpy()
        # particle_x[ps+bone_index[i]:ps+bone_index[i+1]].detach().cpu().numpy()
        
    A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
    A_now = A_now.unsqueeze(0)
    
    # from AnimatableGaussians.smplx.lbs import batch_rodrigues
    # print("\n")
    # print(pose_dataset.body_poses[:5, 12])
    # print(pose_dataset.body_poses[:5, 15])
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        cano_smpl = smplx_model.forward(betas = betas,
                                        global_orient = torch.zeros([1, 3], device=config.device), # [1, 3]
                                        transl = torch.zeros([1, 3], device=config.device), # [1, 3]   
                                        body_pose = torch.zeros([1, 63], device=config.device), # [1, 63]                                            
                                        left_hand_pose = torch.zeros([1, 45], device=config.device), # [1, 45]
                                        right_hand_pose = torch.zeros([1, 45], device=config.device) # [1, 45]
        )
        now_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = smplx_model.forward(betas = betas,
                                        # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        
        if 0:
            T_pred = A_now[0]
            T_gt = now_smpl.A[0, :22]
            parents = pose_dataset.smpl_model.parents[:22]
            
            rot_err_deg, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
            
            rot_err_deg_mean = rot_err_deg.mean()
            trans_err_mean = trans_err.mean()
            rel_rot_err_deg_mean = rel_rot_err_deg.mean()
            rel_trans_err_mean = rel_trans_err.mean()
            
            if f is not None and frame % 100 == 0:
                print(f"frame: {frame}, \
                        rot_mean: {rot_err_deg_mean.item():.6f}, \
                        trl_mean: {trans_err_mean.item():.6f}, \
                        rot_0: {rot_err_deg[0].item():.6f}, \
                        trl_0: {trans_err[0].item():.6f},\
                        rel_rot_err_deg_mean: {rel_rot_err_deg_mean.item():.6f},\
                        rel_trans_err_mean: {rel_trans_err_mean.item():.6f} \n")
                f.write(f"frame: {frame}, \
                        rot_mean: {rot_err_deg_mean.item():.6f}, \
                        trl_mean: {trans_err_mean.item():.6f}, \
                        rot_0: {rot_err_deg[0].item():.6f}, \
                        trl_0: {trans_err[0].item():.6f},\
                        rel_rot_err_deg_mean: {rel_rot_err_deg_mean.item():.6f},\
                        rel_trans_err_mean: {rel_trans_err_mean.item():.6f} \n")
        
        # import trimesh
        # testmesh = trimesh.Trimesh(vertices=cano_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/cano_smpl.ply')
        # testmesh = trimesh.Trimesh(vertices=now_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/now_smpl.ply')
        
        # pt_mats_now = torch.einsum('nj,jxy->nxy', pose_dataset.smpl_model.lbs_weights, now_smpl.A[0])
        # positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_smpl.vertices[0]) + pt_mats_now[..., :3, 3]
        # testmesh = trimesh.Trimesh(vertices=positions_now.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_now.ply')
        
        # particle_x_ori_smplx = particle_x_ori[ps+bone_index[-1]: ps+human_n_particles]
        # testmesh = trimesh.Trimesh(vertices=particle_x_ori_smplx.detach().cpu().numpy())
        # testmesh.export('./test_results/particle_x_ori_smplx.ply')        
    
        N = cano_smpl.vertices.shape[1]
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]
        now_transl = pose_dataset.transl[now_frame]
        
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(now_smpl.A[0, :22, :3, :3]) # rel rotation, 위의 batch_rodrigues로 대체 가능
        
        # 2.2 다음 frame의 global rotation 계산
        # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        transforms[:, :3, :3] = rel_rot @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        # tgt_A[:, :3, :3] = rel_rot @ now_smpl.A[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        # now_smpl.A[0, :22], A_mat, cal_A # [22, 4, 4], the same
        
        # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        for i in range(1, parents.shape[0]):
            transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        now_transl_sim = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0])        
        rel_transforms[0, :, :3, 3] += next_transl - now_transl + now_transl_sim # [1, 22, 4, 4] #%#%#%#
        A_next = rel_transforms
        # (A_next[0] - next_smpl.A[0, :22]).abs().max() # check !!, yes !!
        
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]
        
        # 4. velocity 적용
        # 4.1 Avatar Velocity
        # separable-contact: splice in filled interior particles' LBS + canonical positions
        # (set by particle_filling.lbs_extend.extend_lbs_for_filled_particles).
        # We do NOT modify smplx_model.lbs_weights in place because SMPL-X's own
        # forward() uses it with v_template (10475 verts) and would shape-mismatch.
        lbs_ext = lbs
        cano_verts_ext = cano_smpl.vertices[0]
        if hasattr(smplx_model, '_interior_lbs') and smplx_model._interior_lbs is not None:
            lbs_ext = torch.cat([lbs, smplx_model._interior_lbs.to(lbs.device, lbs.dtype)], dim=0)
            cano_verts_ext = torch.cat([cano_verts_ext, smplx_model._interior_canonical.to(cano_verts_ext.device, cano_verts_ext.dtype)], dim=0)
        pt_mats_now = torch.einsum('nj,jxy->nxy', lbs_ext, A_now_55[0])
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_verts_ext) + pt_mats_now[..., :3, 3]
        # cano_rot_now = torch.tensor([[1, 0, 0, 0]], device=config.device).repeat(N, 1)
        # rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]

        pt_mats_next = torch.einsum('nj,jxy->nxy', lbs_ext, A_next_55[0])
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_verts_ext) + pt_mats_next[..., :3, 3]
        # rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]
            
        # 4.3 Total Velocity
        alpha = 0.0 # must 0.0, only for tgt
        # alpha2 = 0.3
        alpha2 = MPM_sim.human_modify_model[list_idx].velocity_alpha
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        # rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        # rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])
        
        velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]
        velocity += alpha2 * (positions_next_total - positions_now_total_sim) / smplx_dt
        
        # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        velocity = torch.mm(velocity, rot_mats.T) * scale        
        
        # testmesh = trimesh.Trimesh(vertices=positions_next_total.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_next_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=positions_now_total_pos.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_now_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=now_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/now_smpl.ply')
        # testmesh = trimesh.Trimesh(vertices=next_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/next_smpl.ply')
        
        # velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        
    # if is_3d_measure and frame % 10 == 0 :
    if is_3d_measure and (frame == 18 or frame == 16) :
        positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
        positions_now_gt  = now_smpl.vertices[0]
        
        positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
        positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
        
        sim_params["output_path"]
        positions_now_gt_ply.export(sim_params["output_path"] + "positions_now_gt_{:04d}.ply".format(frame))
        positions_now_sim_ply.export(sim_params["output_path"] + "positions_now_sim_{:04d}.ply".format(frame))
        # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
        # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
        
        diff = positions_now_sim - positions_now_gt
        dists = torch.linalg.norm(diff, axis=1)
        
        print("frame : ", frame, \
            # "max : ", (dists).max(), \
            # "min : ", (dists).min(), \
            "mean : ", torch.mean(dists))
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM

# 원본 최신 코드
@torch.no_grad()
def compute_smplx_velocity_tgt(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None, sim_params=None):
    
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center    
    
    smplx_model = MPM_sim.human_modify_model[list_idx].smplx_model
    betas = MPM_sim.human_modify_model[list_idx].betas
    lbs = smplx_model.lbs_weights

    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats

    # A_now derivation: smplx_direct_A (default True) replaces the kabsch fit of
    # canonical bones to current MPM bone particle positions with a direct read of
    # SMPL-X joint matrices at the current pose. Breaks the kabsch-drift feedback
    # loop where bone particles, pulled by interior MPM coupling, cause A_now to
    # drift from the true SMPL-X pose, propagating wrong LBS velocity to all human
    # particles. With smplx_direct_A, bones can still deviate under interaction
    # (paper's contribution preserved) but the LBS prescription stays anchored to
    # the clean SMPL-X pose.
    use_smplx_direct_A = True
    if sim_params is not None and "smplx_direct_A" in sim_params:
        use_smplx_direct_A = bool(sim_params["smplx_direct_A"])

    if use_smplx_direct_A and len(pose_dataset.pose_list) > human_step + 1:
        _first_idx_for_A = pose_dataset.pose_list[0]
        _now_frame_for_A = pose_dataset.pose_list[human_step]
        _now_smpl_for_A = smplx_model.forward(
            betas=betas,
            global_orient=pose_dataset.body_poses[_now_frame_for_A, :3][None],
            transl=pose_dataset.transl[_now_frame_for_A][None],
            body_pose=pose_dataset.body_poses[_now_frame_for_A, 3:66][None],
            left_hand_pose=pose_dataset.left_hand_pose[_first_idx_for_A][None].to(config.device),
            right_hand_pose=pose_dataset.right_hand_pose[_first_idx_for_A][None].to(config.device),
        )
        A_now = _now_smpl_for_A.A[:, :22].clone()
    else:
        # legacy: kabsch fit from MPM bone particle positions
        A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
        kabsch_A = torch.zeros_like(A_now)
        kabsch_A[:, 3, 3] = 1.0
        for i in range(len(bone_index)-1):
            R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]])
            kabsch_A[bone2smplx[i], :3, :3] = R_est
            kabsch_A[bone2smplx[i], :3,  3] = t_est
            joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
            A_now[bone2smplx[i], :3, :3] = R_est
            A_now[bone2smplx[i], :3, 3] = joint_cal
            if bone2smplx[i] == 7 or bone2smplx[i] == 8:
                A_now[bone2smplx[i]+3, :3, :3] = R_est
                A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
                kabsch_A[bone2smplx[i]+3, :3, :3] = R_est
                kabsch_A[bone2smplx[i]+3, :3,  3] = t_est
        A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
        A_now = A_now.unsqueeze(0)
    
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        cano_smpl = smplx_model.forward(betas = betas,
                                        global_orient = torch.zeros([1, 3], device=config.device), # [1, 3]
                                        transl = torch.zeros([1, 3], device=config.device), # [1, 3]   
                                        body_pose = torch.zeros([1, 63], device=config.device), # [1, 63]                                            
                                        left_hand_pose = torch.zeros([1, 45], device=config.device), # [1, 45]
                                        right_hand_pose = torch.zeros([1, 45], device=config.device) # [1, 45]
        )
        now_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = smplx_model.forward(betas = betas,
                                        # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
    
        if 0:
            T_pred = A_now[0]
            T_gt = now_smpl.A[0, :22]
            
            parents = pose_dataset.smpl_model.parents[:22]
            rot_err_deg, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
            rot_err_deg_mean = rot_err_deg.mean()
            trans_err_mean = trans_err.mean()
            
            if f is not None and frame % 100 == 0:
                f.write(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f} \n")
    
        N = cano_smpl.vertices.shape[1]
        parents = pose_dataset.smpl_model.parents[:22]
        joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        joints_homogen = F.pad(joints, [0, 0, 0, 1])
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]        
        next_transl = pose_dataset.transl[next_frame]
        now_transl = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Get transl from root A & J
        
        # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(now_smpl.A[0, :22, :3, :3]) # rel rotation, 위의 batch_rodrigues로 대체 가능
        rel_local_rot = get_local_rot(rel_rot, A_now[0, :22, :3, :3], parents)        
        rel_local_rot_clamped = clamp_rel_rot_threshold(rel_local_rot, max_angle_deg=10.0)        
        rel_rot_new = recompose_global_rel_rot_if_needed(rel_rot, rel_local_rot, rel_local_rot_clamped, A_now[0, :22, :3, :3], parents)        
        
        # 2.2 다음 frame의 global rotation 계산
        # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        transforms[:, :3, :3] = rel_rot_new @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        
        # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        for i in range(1, parents.shape[0]):
            transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        A_next = rel_transforms
        # (A_next[0] - next_smpl.A[0, :22]).abs().max() # check !!, yes !!
        
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]
        
        # 4. velocity 적용
        # 4.1 Avatar Velocity
        # separable-contact: splice in filled interior particles' LBS + canonical positions
        # (set by particle_filling.lbs_extend.extend_lbs_for_filled_particles).
        # We do NOT modify smplx_model.lbs_weights in place because SMPL-X's own
        # forward() uses it with v_template (10475 verts) and would shape-mismatch.
        lbs_ext = lbs
        cano_verts_ext = cano_smpl.vertices[0]
        if hasattr(smplx_model, '_interior_lbs') and smplx_model._interior_lbs is not None:
            lbs_ext = torch.cat([lbs, smplx_model._interior_lbs.to(lbs.device, lbs.dtype)], dim=0)
            cano_verts_ext = torch.cat([cano_verts_ext, smplx_model._interior_canonical.to(cano_verts_ext.device, cano_verts_ext.dtype)], dim=0)
        pt_mats_now = torch.einsum('nj,jxy->nxy', lbs_ext, A_now_55[0])
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_verts_ext) + pt_mats_now[..., :3, 3]
        # cano_rot_now = torch.tensor([[1, 0, 0, 0]], device=config.device).repeat(N, 1)
        # rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]

        pt_mats_next = torch.einsum('nj,jxy->nxy', lbs_ext, A_next_55[0])
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_verts_ext) + pt_mats_next[..., :3, 3]
        # rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]
            
        # 4.3 Total Velocity
        # alpha = 0.3
        alpha = MPM_sim.human_modify_model[list_idx].velocity_alpha
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        # rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        # rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

        lbs_belly = bool(sim_params.get("belly_attenuation", False)) if sim_params is not None else False  # soft-tissue: attenuate kinematic velocity of belly (spine joints 3,6)
        if lbs_belly:
            lbs_copy = lbs.clone() # [N, 55]
            # lbs_copy[:, 3] = lbs_copy[:, 3] / 2.0 # 뱃살쪽은 velocity를 덜 주는거 / index = 3, 6
            lbs_copy[:, 3] = 0.0 # 뱃살쪽은 velocity를 덜 주는거 / index = 3, 6
            # lbs_copy[:, 6] = 0.0
            lbs_copy[:, 6] = lbs_copy[:, 6] / 5.0
            velo_ratio = lbs_copy.sum(axis=1).unsqueeze(-1)
            velo_ratio = torch.concat([torch.ones(bone_verts_num, 1, device=velo_ratio.device), velo_ratio])

            # velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]        
            velocity = velo_ratio * (1-alpha) * (positions_next_total - positions_now_total_pos) / smplx_dt # [373056, 3]
            velocity += alpha * (positions_next_total - positions_now_total_sim) / smplx_dt # [373056, 3]        
            # velocity = velo_ratio * velocity
        else:
            velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]

        # spring-anchored bones: when running with kabsch A_now (smplx_direct_A=false),
        # add an extra spring velocity for bones that pulls them toward the SMPL-X-prescribed
        # bone positions (computed from next_smpl.A, NOT from kabsch). This anchors bones to
        # the dataset pose to prevent indefinite drift, while the main kabsch motion term
        # preserves interaction-driven pose changes (paper's contribution). β=bone_anchor_beta:
        #   0 → no anchor (legacy kabsch-only, drifts over long sequences)
        #   small (~0.05–0.2) → soft anchor: interaction effects persist for ~5–20 frames before relaxing toward dataset pose
        #   large (>0.5) → strong anchor: interaction effects dampened quickly
        bone_anchor_beta = 0.0
        if sim_params is not None and "bone_anchor_beta" in sim_params:
            bone_anchor_beta = float(sim_params["bone_anchor_beta"])
        if bone_anchor_beta > 0.0:
            A_next_smplx = next_smpl.A[:, :22]
            bone_pose2_smplx = torch.zeros_like(bone_pose2)
            for _bi in range(len(bone_index) - 1):
                bone_pose2_smplx[bone_index[_bi] : bone_index[_bi+1]] = (
                    bone_cano[bone_index[_bi] : bone_index[_bi+1]] @ A_next_smplx[0, bone2smplx[_bi], :3, :3].T
                    + A_next_smplx[0, bone2smplx[_bi], :3, 3]
                )
            velocity[:bone_verts_num] = velocity[:bone_verts_num] + bone_anchor_beta * (
                bone_pose2_smplx - positions_now_total_sim[:bone_verts_num]
            ) / smplx_dt

        # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)

        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt
        vSM[:bone_verts_num] = 0.0

        velocity = torch.mm(velocity, rot_mats.T) * scale
        
        # testmesh = trimesh.Trimesh(vertices=positions_next_total.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_next_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=positions_now_total_pos.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_now_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=now_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/now_smpl.ply')
        # testmesh = trimesh.Trimesh(vertices=next_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/next_smpl.ply')
        
        # velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)        
    
    if is_3d_measure and frame % 10 == 0 :
        positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
        positions_now_gt  = now_smpl.vertices[0]
        
        positions_now_gt_ply = trimesh.Trimesh(vertices=positions_now_gt.detach().cpu().numpy())
        positions_now_sim_ply = trimesh.Trimesh(vertices=positions_now_sim.detach().cpu().numpy())
        positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
        positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
        
        diff = positions_now_sim - positions_now_gt
        dists = torch.linalg.norm(diff, axis=1)
        
        print("frame : ", frame, \
            # "max : ", (dists).max(), \
            # "min : ", (dists).min(), \
            "mean : ", torch.mean(dists))
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM

@torch.no_grad()
def compute_smplx_velocity_gt(MPM_sim, particle_x, human_step, list_idx, smplx_dt, frame, is_3d_measure=False, f=None, sim_params=None):
    
    pose_dataset = MPM_sim.human_modify_model[list_idx].pose_dataset
    human_n_particles = MPM_sim.human_modify_model[list_idx].human_n_particles
    
    bone_cano = MPM_sim.human_modify_model[list_idx].bone_cano # [74496, 3]
    bone_index = MPM_sim.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
    bone2smplx = MPM_sim.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
    cano_J = MPM_sim.human_modify_model[list_idx].cano_J # [55, 3]
    ps = MPM_sim.human_modify_model[list_idx].particle_start
    
    ori_mean = MPM_sim.human_modify_model[list_idx].ori_mean
    rot_mats = MPM_sim.human_modify_model[list_idx].rot_mats
    scale    = MPM_sim.human_modify_model[list_idx].scale
    center   = MPM_sim.human_modify_model[list_idx].center    
    
    smplx_model = MPM_sim.human_modify_model[list_idx].smplx_model
    betas = MPM_sim.human_modify_model[list_idx].betas
    lbs = smplx_model.lbs_weights

    particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats

    # A_now derivation: smplx_direct_A (default True) replaces the kabsch fit of
    # canonical bones to current MPM bone particle positions with a direct read of
    # SMPL-X joint matrices at the current pose. Breaks the kabsch-drift feedback
    # loop where bone particles, pulled by interior MPM coupling, cause A_now to
    # drift from the true SMPL-X pose, propagating wrong LBS velocity to all human
    # particles. With smplx_direct_A, bones can still deviate under interaction
    # (paper's contribution preserved) but the LBS prescription stays anchored to
    # the clean SMPL-X pose.
    use_smplx_direct_A = True
    if sim_params is not None and "smplx_direct_A" in sim_params:
        use_smplx_direct_A = bool(sim_params["smplx_direct_A"])

    if use_smplx_direct_A and len(pose_dataset.pose_list) > human_step + 1:
        _first_idx_for_A = pose_dataset.pose_list[0]
        _now_frame_for_A = pose_dataset.pose_list[human_step]
        _now_smpl_for_A = smplx_model.forward(
            betas=betas,
            global_orient=pose_dataset.body_poses[_now_frame_for_A, :3][None],
            transl=pose_dataset.transl[_now_frame_for_A][None],
            body_pose=pose_dataset.body_poses[_now_frame_for_A, 3:66][None],
            left_hand_pose=pose_dataset.left_hand_pose[_first_idx_for_A][None].to(config.device),
            right_hand_pose=pose_dataset.right_hand_pose[_first_idx_for_A][None].to(config.device),
        )
        A_now = _now_smpl_for_A.A[:, :22].clone()
    else:
        # legacy: kabsch fit from MPM bone particle positions
        A_now = torch.eye(4, device=MPM_sim.device).unsqueeze(0).repeat(22, 1, 1)
        kabsch_A = torch.zeros_like(A_now)
        kabsch_A[:, 3, 3] = 1.0
        for i in range(len(bone_index)-1):
            R_est, t_est = MPM_sim.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]])
            kabsch_A[bone2smplx[i], :3, :3] = R_est
            kabsch_A[bone2smplx[i], :3,  3] = t_est
            joint_cal = R_est @ cano_J[bone2smplx[i], :3] + t_est
            A_now[bone2smplx[i], :3, :3] = R_est
            A_now[bone2smplx[i], :3, 3] = joint_cal
            if bone2smplx[i] == 7 or bone2smplx[i] == 8:
                A_now[bone2smplx[i]+3, :3, :3] = R_est
                A_now[bone2smplx[i]+3, :3, 3] = R_est @ cano_J[bone2smplx[i]+3, :3] + t_est
                kabsch_A[bone2smplx[i]+3, :3, :3] = R_est
                kabsch_A[bone2smplx[i]+3, :3,  3] = t_est
        A_now[:, :, 3] = A_now[:, :, 3] - torch.einsum('bij,bj->bi', A_now, cano_J)
        A_now = A_now.unsqueeze(0)
    
    if len(pose_dataset.pose_list) > human_step + 1:
        first_idx = pose_dataset.pose_list[0]
        now_frame = pose_dataset.pose_list[human_step]
        next_frame = pose_dataset.pose_list[human_step + 1]
        cano_smpl = smplx_model.forward(betas = betas,
                                        global_orient = torch.zeros([1, 3], device=config.device), # [1, 3]
                                        transl = torch.zeros([1, 3], device=config.device), # [1, 3]   
                                        body_pose = torch.zeros([1, 63], device=config.device), # [1, 63]                                            
                                        left_hand_pose = torch.zeros([1, 45], device=config.device), # [1, 45]
                                        right_hand_pose = torch.zeros([1, 45], device=config.device) # [1, 45]
        )
        now_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        next_smpl = smplx_model.forward(betas = betas,
                                        global_orient = pose_dataset.body_poses[next_frame, :3][None], # [1, 3]
                                        transl = pose_dataset.transl[next_frame][None], # [1, 3]   
                                        body_pose = pose_dataset.body_poses[next_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[next_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[next_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
        only_finger = smplx_model.forward(betas = betas,
                                        # global_orient = pose_dataset.body_poses[now_frame, :3][None], # [1, 3]
                                        # transl = pose_dataset.transl[now_frame][None], # [1, 3]   
                                        # body_pose = pose_dataset.body_poses[now_frame, 3: 66][None], # [1, 63]
                                        # left_hand_pose = pose_dataset.left_hand_pose[now_frame][None].to(config.device), # [1, 45]
                                        # right_hand_pose = pose_dataset.right_hand_pose[now_frame][None].to(config.device), # [1, 45]                                                    
                                        left_hand_pose = pose_dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = pose_dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
        )
    
        if 0:
            T_pred = A_now[0]
            T_gt = now_smpl.A[0, :22]
            
            parents = pose_dataset.smpl_model.parents[:22]
            rot_err_deg, trans_err, rel_rot_err_deg, rel_trans_err = compute_rt_errors(T_pred, T_gt, parents)
            rot_err_deg_mean = rot_err_deg.mean()
            trans_err_mean = trans_err.mean()
            
            if f is not None and frame % 100 == 0:
                
                print(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f} \n")
                
                f.write(f"frame: {frame}, \
                    rot_mean: {rot_err_deg_mean.item():.6f}, \
                    trl_mean: {trans_err_mean.item():.6f}, \
                    rot_0: {rot_err_deg[0].item():.6f}, \
                    trl_0: {trans_err[0].item():.6f} \n")
    
        A_next = next_smpl.A[:, :22]
    
        # N = cano_smpl.vertices.shape[1]
        # parents = pose_dataset.smpl_model.parents[:22]
        # joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
        # joints_homogen = F.pad(joints, [0, 0, 0, 1])
        # rel_joints = joints.clone()
        # rel_joints[:, 1:] -= joints[:, parents[1:]]        
        # next_transl = pose_dataset.transl[next_frame]
        # now_transl = A_now[0, 0, :3, 3] - joints[0, 0, :, 0] + (A_now[0, 0, :3, :3] @ joints[0, 0, :, 0]) # Get transl from root A & J
        
        # # 2.1 현재 frame에서 다음 frame으로 각 관절 global relative rotation 계산 (relative rotation, from 데이터셋에서)
        # rel_rot = next_smpl.A[0, :22, :3, :3] @ torch.linalg.inv(now_smpl.A[0, :22, :3, :3]) # rel rotation, 위의 batch_rodrigues로 대체 가능
        # rel_local_rot = get_local_rot(rel_rot, A_now[0, :22, :3, :3], parents)        
        # rel_local_rot_clamped = clamp_rel_rot_threshold(rel_local_rot, max_angle_deg=10.0)        
        # rel_rot_new = recompose_global_rel_rot_if_needed(rel_rot, rel_local_rot, rel_local_rot_clamped, A_now[0, :22, :3, :3], parents)        
        
        # # 2.2 다음 frame의 global rotation 계산
        # # 현재 global rotation에 relative rotation 곱하기, 현재 transmat @ rel_rot
        # transforms = torch.eye(4, device=config.device).unsqueeze(0).repeat(22, 1, 1)
        # transforms[:, :3, :3] = rel_rot_new @ A_now[0, :22, :3, :3] # 나중에 현재 A-pose matrix로 대체
        
        # # 2.3 smplx 방식 그대로 다음 frame global A-pose matrix 계산, only Global Rotation만 으로 계산 !!
        # transforms[0, :3, 3] = rel_joints[0, 0, :3, 0]
        # for i in range(1, parents.shape[0]):
        #     transforms[i, :3, 3] = transforms[parents[i], :3, 3] + torch.matmul(transforms[parents[i], :3, :3], rel_joints[0, i, :3, 0])
        # rel_transforms = transforms - F.pad(
        #     torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]) # [1, 55, 4, 4]
        # rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
        # A_next = rel_transforms
        # (A_next[0] - next_smpl.A[0, :22]).abs().max() # check !!, yes !!
        
        # 3.1 live_smpl.A에서 live_smpl_woRoot.A 구하기
        global_rotation = torch.eye(4, device = config.device)
        global_rotation[:3, :3] = A_next[0, 0, :3, :3]
        A_next_woRoot = A_next.clone()
        A_next_woRoot[:, :, :3, 3] -= A_next[0, 0, :3, 3]
        A_next_woRoot = torch.linalg.inv(global_rotation) @ A_next_woRoot
        
        # 3.2 delta_position
        A_now_55  = torch.concat([A_now, torch.zeros((1, 33, 4, 4), device=A_now.device)], dim=1)        
        A_next_55 = torch.concat([A_next, torch.zeros((1, 33, 4, 4), device=A_next.device)], dim=1)
        A_next_woRoot_55 = torch.concat([A_next_woRoot, torch.zeros((1, 33, 4, 4), device=A_next_woRoot.device)], dim=1)
        
        A_now_55[0, 22:25] = A_now_55[0, 15]
        A_now_55[0, 25:40] = A_now_55[0, 20] @ only_finger.A[0, 25:40]
        A_now_55[0, 40:55] = A_now_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_55[0, 22:25] = A_next_55[0, 15]
        A_next_55[0, 25:40] = A_next_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_55[0, 40:55] = A_next_55[0, 21] @ only_finger.A[0, 40:55]
        
        A_next_woRoot_55[0, 22:25] = A_next_woRoot_55[0, 15]
        A_next_woRoot_55[0, 25:40] = A_next_woRoot_55[0, 20] @ only_finger.A[0, 25:40]
        A_next_woRoot_55[0, 40:55] = A_next_woRoot_55[0, 21] @ only_finger.A[0, 40:55]
        
        # 4. velocity 적용
        # 4.1 Avatar Velocity
        # separable-contact: splice in filled interior particles' LBS + canonical positions
        # (set by particle_filling.lbs_extend.extend_lbs_for_filled_particles).
        # We do NOT modify smplx_model.lbs_weights in place because SMPL-X's own
        # forward() uses it with v_template (10475 verts) and would shape-mismatch.
        lbs_ext = lbs
        cano_verts_ext = cano_smpl.vertices[0]
        if hasattr(smplx_model, '_interior_lbs') and smplx_model._interior_lbs is not None:
            lbs_ext = torch.cat([lbs, smplx_model._interior_lbs.to(lbs.device, lbs.dtype)], dim=0)
            cano_verts_ext = torch.cat([cano_verts_ext, smplx_model._interior_canonical.to(cano_verts_ext.device, cano_verts_ext.dtype)], dim=0)
        pt_mats_now = torch.einsum('nj,jxy->nxy', lbs_ext, A_now_55[0])
        positions_now = torch.einsum('nxy,ny->nx', pt_mats_now[..., :3, :3], cano_verts_ext) + pt_mats_now[..., :3, 3]
        # cano_rot_now = torch.tensor([[1, 0, 0, 0]], device=config.device).repeat(N, 1)
        # rot_mats_now = torch.einsum('nxy,nyz->nxz', pt_mats_now[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]

        pt_mats_next = torch.einsum('nj,jxy->nxy', lbs_ext, A_next_55[0])
        positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_verts_ext) + pt_mats_next[..., :3, 3]
        # rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_now)) # [human_N, 3, 3]
        
        # 4.2 Bone Velocity
        bone_verts_num = bone_cano.shape[0]
        bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
        bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        
        for i in range(len(bone_index)-1):
            bone_pose_i_1 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_now[0, bone2smplx[i], :3, :3].T + A_now[0, bone2smplx[i], :3, 3]
            bone_pose_i_2 = bone_cano[bone_index[i] : bone_index[i+1]] @ A_next[0, bone2smplx[i], :3, :3].T + A_next[0, bone2smplx[i], :3, 3]
            
            bone_pose1[bone_index[i] : bone_index[i+1]] = bone_pose_i_1
            bone_pose2[bone_index[i] : bone_index[i+1]] = bone_pose_i_2                
            bone_rot1[bone_index[i] : bone_index[i+1]] = A_now[0, bone2smplx[i], :3, :3]
            bone_rot2[bone_index[i] : bone_index[i+1]] = A_next[0, bone2smplx[i], :3, :3]
            
        # 4.3 Total Velocity
        # alpha = 0.3
        alpha = MPM_sim.human_modify_model[list_idx].velocity_alpha
        # human particles are NOT contiguous in MPM order when filled-interior is appended after another subject
        # (merge_subjects layout: [surf_0, surf_1, interior_0, interior_1, ...]). Use particle_id_torch for correct indexing.
        _human_idx = MPM_sim.human_modify_model[list_idx].particle_id_torch
        positions_now_total_sim  = particle_x_ori[_human_idx] # xyz of real MPM simulation
        positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
        positions_next_total = torch.cat([bone_pose2, positions_next])
        
        # rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
        # rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])
        lbs_belly = bool(sim_params.get("belly_attenuation", False)) if sim_params is not None else False  # soft-tissue: attenuate kinematic velocity of belly (spine joints 3,6)
        if lbs_belly:
            lbs_copy = lbs.clone() # [N, 55]
            # lbs_copy[:, 3] = lbs_copy[:, 3] / 2.0 # 뱃살쪽은 velocity를 덜 주는거 / index = 3, 6
            lbs_copy[:, 3] = 0.0 # 뱃살쪽은 velocity를 덜 주는거 / index = 3, 6
            # lbs_copy[:, 6] = 0.0
            lbs_copy[:, 6] = lbs_copy[:, 6] / 5.0
            velo_ratio = lbs_copy.sum(axis=1).unsqueeze(-1)
            velo_ratio = torch.concat([torch.ones(bone_verts_num, 1, device=velo_ratio.device), velo_ratio])

            # velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]        
            velocity = velo_ratio * (1-alpha) * (positions_next_total - positions_now_total_pos) / smplx_dt # [373056, 3]
            velocity += alpha * (positions_next_total - positions_now_total_sim) / smplx_dt # [373056, 3]        
            # velocity = velo_ratio * velocity
        else:
            velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]
        # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        
        # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        
        # 4.4 Velocity Shape Matching
        positions_now_total_glb = particle_x[MPM_sim.human_modify_model[list_idx].particle_id_torch]
        positions_now_total_pos_glb = (torch.mm(positions_now_total_pos, rot_mats.T) - ori_mean) * scale + center
        vSM = (positions_now_total_pos_glb - positions_now_total_glb) / smplx_dt 
        vSM[:bone_verts_num] = 0.0
        
        velocity = torch.mm(velocity, rot_mats.T) * scale        
        
        # testmesh = trimesh.Trimesh(vertices=positions_next_total.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_next_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=positions_now_total_pos.detach().cpu().numpy())
        # testmesh.export('./test_results/positions_now_total_pos.ply')
        # testmesh = trimesh.Trimesh(vertices=now_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/now_smpl.ply')
        # testmesh = trimesh.Trimesh(vertices=next_smpl.vertices[0].detach().cpu().numpy())
        # testmesh.export('./test_results/next_smpl.ply')
        
        # velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
    
    else:
        velocity = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
        colors_next = torch.zeros([human_n_particles, 3], device=MPM_sim.device)
        vSM = torch.zeros([human_n_particles, 3], device=MPM_sim.device)        
    
    if is_3d_measure and (frame == 18 or frame == 16) :
        positions_now_sim = particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
        positions_now_gt  = now_smpl.vertices[0]
        
        positions_now_gt_ply = trimesh.points.PointCloud(vertices=positions_now_gt.detach().cpu().numpy())
        positions_now_sim_ply = trimesh.points.PointCloud(vertices=positions_now_sim.detach().cpu().numpy())
        
        sim_params["output_path"]
        positions_now_gt_ply.export(sim_params["output_path"]  + "gt_{:04d}.ply".format(frame))
        positions_now_sim_ply.export(sim_params["output_path"] + "sim_{:04d}.ply".format(frame))
        # positions_now_gt_ply.export("./test_results/positions_now_gt_{:04d}.ply".format(frame))
        # positions_now_sim_ply.export("./test_results/positions_now_sim_{:04d}.ply".format(frame))
        
        # sim_load = trimesh.load(sim_params["output_path"] + "positions_now_sim_{:04d}.ply".format(frame), process=False)
        # gt_load = trimesh.load(sim_params["output_path"] + "positions_now_gt_{:04d}.ply".format(frame), process=False)
        # sim_pc = np.asarray(sim_load.vertices)
        # gt_pc = np.asarray(gt_load.vertices)
        
        diff = positions_now_sim - positions_now_gt
        dists = torch.linalg.norm(diff, axis=1)
        
        print("frame : ", frame, \
            # "max : ", (dists).max(), \
            # "min : ", (dists).min(), \
            "mean : ", torch.mean(dists))
    
    # interior_no_velocity ablation: zero kinematic velocity for filled interior
    # particles so they drift only via grid coupling from surface particles
    if sim_params is not None and sim_params.get("interior_no_velocity", False):
        if hasattr(smplx_model, '_interior_canonical') and smplx_model._interior_canonical is not None:
            n_int = smplx_model._interior_canonical.shape[0]
            if n_int > 0:
                velocity = velocity.clone()
                velocity[-n_int:] = 0.0
                vSM = vSM.clone()
                vSM[-n_int:] = 0.0
    return velocity, relative_rot_mats, lbs, vSM


###################################################################################################################################################

import torch

@torch.no_grad()
def compute_rt_errors(T_pred, T_gt, parents):
    """
    Args:
        T_pred, T_gt: [N, 4, 4] (조인트별 글로벌 변환)
        parents: list/1D tensor 길이 N, 루트는 -1
    
    Returns:
        rot_err:         [N]         (글로벌 회전 오차, deg)
        trans_err:       [N]         (글로벌 평행이동 오차, same unit as t)
        rel_rot_err_deg: [N-#roots]  (부모-자식 로컬 회전 오차, deg; 루트 제외)
        rel_trans_err:   [N-#roots]  (부모-자식 로컬 평행이동 오차; 루트 제외)
    """
    # --- 분리 ---
    R_pred = T_pred[:, :3, :3]
    R_gt   = T_gt[:, :3, :3]
    t_pred = T_pred[:, :3, 3]
    t_gt   = T_gt[:, :3, 3]

    # --- Global Rotation Error (deg) ---
    R_rel = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
    trace = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_err = torch.acos(trace) * (180.0 / torch.pi)

    # --- Global Translation Error ---
    trans_err = torch.norm(t_pred - t_gt, dim=-1)

    # --- Local (parent-relative) Rotation Error (deg) ---
    rel_R_gt_list = []
    rel_R_pred_list = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        # R_loc = R_p^T * R_i
        rel_R_gt_list.append(R_gt[p].transpose(-1, -2) @ R_gt[i])
        rel_R_pred_list.append(R_pred[p].transpose(-1, -2) @ R_pred[i])

    rel_R_gt   = torch.stack(rel_R_gt_list, dim=0)  # [M,3,3]
    rel_R_pred = torch.stack(rel_R_pred_list, dim=0)  # [M,3,3]
    rel_R_rel  = rel_R_gt.transpose(-1, -2) @ rel_R_pred

    trace = torch.diagonal(rel_R_rel, dim1=-2, dim2=-1).sum(-1)
    trace = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rel_rot_err_deg = torch.acos(trace) * (180.0 / torch.pi)  # [M]

    # --- Local (parent-relative) Translation Error ---
    # t_loc = R_p^T * (t_i - t_p)
    rel_t_gt_list   = []
    rel_t_pred_list = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        rel_t_gt   = R_gt[p].transpose(-1, -2) @ (t_gt[i]   - t_gt[p])
        rel_t_pred = R_pred[p].transpose(-1, -2) @ (t_pred[i] - t_pred[p])
        rel_t_gt_list.append(rel_t_gt)
        rel_t_pred_list.append(rel_t_pred)

    rel_t_gt   = torch.stack(rel_t_gt_list, dim=0)     # [M,3]
    rel_t_pred = torch.stack(rel_t_pred_list, dim=0)   # [M,3]
    rel_trans_err = torch.norm(rel_t_pred - rel_t_gt, dim=-1)  # [M]

    return rot_err, trans_err, rel_rot_err_deg, rel_trans_err


@torch.no_grad()
def compute_rt_errors_hier(
    T_pred: torch.Tensor,   # [J,4,4]
    T_gt:   torch.Tensor,   # [J,4,4]
    parents,                # [J], root = -1
    anchor: str = 'gt_parent'  # 'gt_parent' or 'sym'
):
    """
    Returns:
      dict with keys:
        rot_err_global:  [J] deg
        trans_err_global:[J]
        rot_err_local:   [J] deg (root=NaN)
        trans_err_local: [J]     (root=NaN)
        local_valid_mask:[J] bool
    """
    assert T_pred.shape == T_gt.shape and T_pred.shape[-2:] == (4,4)
    J = T_pred.shape[0]
    device = T_pred.device
    parents = torch.as_tensor(parents, device=device, dtype=torch.long)
    root_mask = (parents < 0)
    has_parent = ~root_mask

    def so3_angle_deg(R):
        tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        x = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        return torch.acos(x) * (180.0 / torch.pi)

    def se3_inv(T):
        R = T[..., :3, :3]; t = T[..., :3, 3]
        Rt = R.transpose(-1, -2)
        Tout = T.clone()
        Tout[..., :3, :3] = Rt
        Tout[..., :3, 3]  = (-Rt @ t[..., None])[..., 0]
        Tout[..., 3, 3]   = 1.0
        return Tout

    # ---------- Global errors ----------
    R_pred = T_pred[:, :3, :3]; R_gt = T_gt[:, :3, :3]
    t_pred = T_pred[:, :3, 3];  t_gt = T_gt[:, :3, 3]

    R_rel_g = R_gt.transpose(-1, -2) @ R_pred
    rot_err_global = so3_angle_deg(R_rel_g)
    trans_err_global = torch.norm(t_pred - t_gt, dim=-1)

    # ---------- Local transforms ----------
    T_local_gt   = torch.empty_like(T_gt)
    T_local_pred = torch.empty_like(T_pred)

    # Root: 로컬 정의 불가. 자리만 채움
    T_local_gt[root_mask]   = T_gt[root_mask]
    T_local_pred[root_mask] = T_pred[root_mask]

    idxs = torch.nonzero(has_parent, as_tuple=False).flatten()
    if idxs.numel() > 0:
        p = parents[idxs]

        # 공통: GT 로컬(진리값) = inv(T_parent_gt) @ T_gt
        T_parent_gt = T_gt[p]
        T_local_gt[idxs] = se3_inv(T_parent_gt) @ T_gt[idxs]

        if anchor == 'gt_parent':
            # 부모 프레임을 GT로 고정해 예측을 투영
            T_local_pred[idxs] = se3_inv(T_parent_gt) @ T_pred[idxs]
        elif anchor == 'sym':
            # 예측도 자기 부모(예측)의 프레임에서 비교
            T_parent_pred = T_pred[p]
            T_local_pred[idxs] = se3_inv(T_parent_pred) @ T_pred[idxs]
        else:
            raise ValueError("anchor must be 'gt_parent' or 'sym'")

    # ---------- Local errors ----------
    R_local_pred = T_local_pred[:, :3, :3]
    R_local_gt   = T_local_gt[:, :3, :3]
    t_local_pred = T_local_pred[:, :3, 3]
    t_local_gt   = T_local_gt[:, :3, 3]

    R_rel_l = R_local_gt.transpose(-1, -2) @ R_local_pred
    rot_err_local   = so3_angle_deg(R_rel_l)
    trans_err_local = torch.norm(t_local_pred - t_local_gt, dim=-1)

    # Root의 로컬 오차는 NaN 처리
    rot_err_local   = rot_err_local.masked_fill(root_mask, float('nan'))
    trans_err_local = trans_err_local.masked_fill(root_mask, float('nan'))

    return {
        "rot_err_global": rot_err_global,
        "trans_err_global": trans_err_global,
        "rot_err_local": rot_err_local,
        "trans_err_local": trans_err_local,
        "local_valid_mask": has_parent,
    }


@torch.no_grad()
def clamp_rel_rot_threshold(R_rel: torch.Tensor, max_angle_deg: float, eps: float = 1e-9):
    """
    R_rel: (..., 3, 3) 상대 회전행렬들
    max_angle_deg: 각도 임계값(도 단위). 임계값 초과시에만 clamp, 아니면 원본 유지
    eps: 수치 안정용
    반환: R_out (..., 3, 3)
    """
    assert R_rel.shape[-2:] == (3, 3)
    device, dtype = R_rel.device, R_rel.dtype
    I = torch.eye(3, device=device, dtype=dtype).expand_as(R_rel)

    max_ang = torch.deg2rad(torch.tensor(max_angle_deg, device=device, dtype=dtype))

    # 회전각 추정
    # cosθ = (tr(R) - 1)/2
    cos_theta = (R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2] - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)                          # (...,)

    # 임계각 초과 마스크
    mask = theta > (max_ang + 1e-12)                       # (...,)
    # torch.rad2deg(theta)

    # 아무 것도 안 넘으면 그대로 반환
    if not mask.any():
        return R_rel

    # 넘는 항목만 클램프용 회전 재구성
    Rm = R_rel[mask]                                       # (M,3,3)
    thetam = theta[mask].unsqueeze(-1)                     # (M,1)

    # 축 추출 (반대칭 성분)
    axis = torch.stack([
        Rm[:, 2, 1] - Rm[:, 1, 2],
        Rm[:, 0, 2] - Rm[:, 2, 0],
        Rm[:, 1, 0] - Rm[:, 0, 1],
    ], dim=-1)                                            # (M,3)

    # sin(theta) ~= 0 근처 보호용 정규화
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    axis = axis / axis_norm                                # 단위축

    # 클램프된 각도
    theta_c = max_ang.expand_as(thetam)                    # (M,1)

    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2
    # K (skew-symmetric)
    K = torch.zeros_like(Rm)
    K[:, 0, 1] = -axis[:, 2];  K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1];  K[:, 2, 1] =  axis[:, 0]

    sin_tc = torch.sin(theta_c)[:, None]                   # (M,1,1)
    cos_tc = torch.cos(theta_c)[:, None]

    R_clamped_m = torch.eye(3, device=device, dtype=dtype).expand_as(Rm) \
                + sin_tc * K \
                + (1 - cos_tc) * (K @ K)                 # (M,3,3)

    # 결과 합치기: 넘는 곳만 교체, 나머지는 원본 유지
    R_out = R_rel.clone()
    R_out[mask] = R_clamped_m
    return R_out

@torch.no_grad()
def get_local_rot(rel_rot, R_now, parents):

    J = rel_rot.shape[0]
    dL = torch.empty_like(rel_rot)
    R_next = rel_rot @ R_now
    for i in range(J):
        p = parents[i].item()
        if p < 0 or p == i:          # root
            dL[i] = rel_rot[i]
        else:
            dL[i] = R_next[p].transpose(-1, -2) @ rel_rot[i] @ R_now[p]
        
    # R_rel_ = dL
    # cos_theta_ = (R_rel_[..., 0, 0] + R_rel_[..., 1, 1] + R_rel_[..., 2, 2] - 1.0) * 0.5
    # cos_theta_ = torch.clamp(cos_theta_, -1.0, 1.0)
    # deg_ = torch.rad2deg(torch.acos(cos_theta_))
    
    return dL

@torch.no_grad()
def recompose_global_rel_rot_if_needed(
    rel_rot: torch.Tensor,                  # (J,3,3) 기존 전역 상대회전 D_i
    rel_local_rot: torch.Tensor,            # (J,3,3) 원래 로컬 ΔL_i
    rel_local_rot_clamped: torch.Tensor,    # (J,3,3) 클램프 적용된 로컬 ΔL_i
    R_now: torch.Tensor,                    # (J,3,3) 현 프레임 전역 회전 G_i^t
    parents: torch.Tensor,                  # (J,)  루트는 -1 또는 자기 자신
    reortho: bool = True,
    atol: float = 1e-7, rtol: float = 1e-6
):
    """
    로컬 ΔR가 바뀌었는지 관절별로 검사.
    바뀌었으면 트리를 따라 전역 rel_rot 재계산, 아니면 원본 유지.
    """

    J = rel_rot.shape[0]
    device, dtype = rel_rot.device, rel_rot.dtype

    # ---- 튜플 dim 사용 안 하고 안정적으로 마스크 만들기 ----
    eq = torch.isclose(rel_local_rot, rel_local_rot_clamped, rtol=rtol, atol=atol)  # (J,3,3) bool
    # (J,3,3) → (J,9)으로 평탄화 후, 한 요소라도 다르면 changed=True
    changed_per_joint = (~eq).reshape(J, -1).any(dim=1)  # (J,) bool

    if not bool(changed_per_joint.any()):
        return rel_rot  # 완전 동일하면 원본 유지

    # 바뀐 조인트만 클램프값으로 교체
    # print("clamped rotation")
    dL_new = rel_local_rot.clone()
    dL_new[changed_per_joint] = rel_local_rot_clamped[changed_per_joint]

    # 재구성용 텐서
    D_new  = torch.empty_like(rel_rot)  # 전역 상대 회전 D_i^new
    R_next = torch.empty_like(R_now)    # 다음 프레임 전역 회전 R_i^{t+1}

    # 루트→자식 순서로 전파 (parents가 위상정렬되어 있다고 가정)
    for i in range(J):
        p = int(parents[i].item())
        if p < 0 or p == i:
            # 루트: D_0 = ΔL_0
            D_new[i]  = dL_new[i]
            R_next[i] = D_new[i] @ R_now[i]
        else:
            # D_i = R_{p}^{t+1} * ΔL_i * (R_{p}^{t})^T
            D_new[i]  = R_next[p] @ dL_new[i] @ R_now[p].transpose(-1, -2)
            R_next[i] = D_new[i] @ R_now[i]

        if reortho:
            # 수치 안정화(직교화)
            U, _, Vh = torch.linalg.svd(D_new[i])
            D_new[i] = U @ Vh
            U, _, Vh = torch.linalg.svd(R_next[i])
            R_next[i] = U @ Vh

    return D_new