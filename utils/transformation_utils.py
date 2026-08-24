import torch
import numpy as np
from utils.camera_view_utils import *

def transform2origin(position_tensor):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = 1.0 / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def get_rotation_degree(subject_params):
    degrees = []
    for subject in subject_params:
        degrees.append(torch.tensor(subject["rotation_degree"]))
    return degrees
    
def get_rotation_axis(subject_params):
    axes = []
    for subject in subject_params:
        axes.append(torch.tensor(subject["rotation_axis"]))
    return axes    

def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated

#
def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated

#
def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)

#
def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_uppder_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rot_rotations(rot, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        # rot = rot @ R
        rot = R.T @ rot
    return rot

def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor

#
def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_uppder_from_mat(cov_tensor)


# position -> [0, 0, 0] -> scaling -> center
def transform2center(position_tensor, scale, center):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    # max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    # scale = 1.0 / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    # scale = scale.to(device="cuda")
    center = torch.tensor(center, device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale + center

    return new_position_tensor, original_mean_pos

def undotransform2center(position_tensor, original_mean_pos, scale, center):
    center = torch.tensor(center, device="cuda")
    return (position_tensor - center) / scale + original_mean_pos    


# input must be (n,3) tensor on cuda
def undo_all_transforms_with_center(input, rotation_matrices, scale, original_mean_pos, center):
    return apply_inverse_rotations(
            undotransform2center(input, original_mean_pos, scale, center),
            rotation_matrices,
        )
    # return apply_inverse_rotations(
    #     undotransform2origin(
    #         undoshift2center111(input), scale_origin, original_mean_pos
    #     ),
    #     rotation_matrices,
    # )
    
def get_center_view_worldspace_and_observant_coordinate_arbitrary_center(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
    center,
):
    viewpoint_center_worldspace = undo_all_transforms_with_center( # array([-3.8497150e-04, -4.5872062e-01,  7.1185105e-02], dtype=float32)
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos, center
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center      # [0, 0, 1] + [1, 1, 1]
    worldspace_up = undo_all_transforms_with_center(                                # tensor([[-3.8497e-04,  1.2079e+00,  7.1185e-02]]
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos, center 
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace # torch.nn.functional.normalize(world_space_vertical_axis)
    # world_space_vertical_axis = torch.nn.functional.normalize(world_space_vertical_axis)
    
    viewpoint_center_worldspace = np.squeeze(viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0)
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates



def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates

def orthonormalize_rotation(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    수치오차로 살짝 어긋난 3x3을 정규직교화 (Polar 보정).
    입력: R (...,3,3)
    출력: R_ortho (...,3,3) with det≈+1
    """
    U, _, Vh = torch.linalg.svd(R)
    R_ortho = U @ Vh
    # det<0이면 마지막 축 반전으로 보정
    det = torch.det(R_ortho)
    mask = det < 0
    if mask.any():
        # 마지막 열에 -1을 곱해 반전
        U2 = U.clone()
        U2[mask, :, -1] *= -1
        R_ortho = U2 @ Vh
    return R_ortho

def rotmat_to_quat(R: torch.Tensor, ordering: str = "wxyz", normalize: bool = True,
                   orthonormalize: bool = False) -> torch.Tensor:
    """
    회전행렬(...,3,3) -> 쿼터니언(...,4)
    ordering: "wxyz" (기본) 또는 "xyzw"
    normalize: 출력 쿼터니언을 단위화
    orthonormalize: 입력 R을 SVD로 정규직교화(수치오차가 큰 경우 켜기)
    """
    if orthonormalize:
        R = orthonormalize_rotation(R)

    # 배치 평탄화
    orig_shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)

    r00 = Rf[:, 0, 0]; r01 = Rf[:, 0, 1]; r02 = Rf[:, 0, 2]
    r10 = Rf[:, 1, 0]; r11 = Rf[:, 1, 1]; r12 = Rf[:, 1, 2]
    r20 = Rf[:, 2, 0]; r21 = Rf[:, 2, 1]; r22 = Rf[:, 2, 2]

    trace = r00 + r11 + r22
    q = torch.empty((Rf.shape[0], 4), dtype=R.dtype, device=R.device)

    # 네 가지 분기 마스크 (trace>0, 그 외 최대 대각성분)
    t_pos = trace > 0
    m0 = (r00 >= r11) & (r00 >= r22) & (~t_pos)
    m1 = (r11 > r22) & (~t_pos) & (~m0)
    m2 = (~t_pos) & (~m0) & (~m1)

    # case 1: trace > 0
    if t_pos.any():
        t = trace[t_pos]
        S = torch.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (r21[t_pos] - r12[t_pos]) / S
        qy = (r02[t_pos] - r20[t_pos]) / S
        qz = (r10[t_pos] - r01[t_pos]) / S
        q[t_pos] = torch.stack([qw, qx, qy, qz], dim=-1)

    # case 2: r00 is largest
    if m0.any():
        S = torch.sqrt(1.0 + r00[m0] - r11[m0] - r22[m0]) * 2.0
        qw = (r21[m0] - r12[m0]) / S
        qx = 0.25 * S
        qy = (r01[m0] + r10[m0]) / S
        qz = (r02[m0] + r20[m0]) / S
        q[m0] = torch.stack([qw, qx, qy, qz], dim=-1)

    # case 3: r11 is largest
    if m1.any():
        S = torch.sqrt(1.0 + r11[m1] - r00[m1] - r22[m1]) * 2.0
        qw = (r02[m1] - r20[m1]) / S
        qx = (r01[m1] + r10[m1]) / S
        qy = 0.25 * S
        qz = (r12[m1] + r21[m1]) / S
        q[m1] = torch.stack([qw, qx, qy, qz], dim=-1)

    # case 4: r22 is largest
    if m2.any():
        S = torch.sqrt(1.0 + r22[m2] - r00[m2] - r11[m2]) * 2.0
        qw = (r10[m2] - r01[m2]) / S
        qx = (r02[m2] + r20[m2]) / S
        qy = (r12[m2] + r21[m2]) / S
        qz = 0.25 * S
        q[m2] = torch.stack([qw, qx, qy, qz], dim=-1)

    if normalize:
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

    # 부호 일관성: w>=0로 맞추면 시간 연속성에 유리
    flip = q[:, 0] < 0
    q[flip] = -q[flip]

    q = q.reshape(*orig_shape, 4)

    if ordering.lower() == "xyzw":
        # (w,x,y,z) -> (x,y,z,w)
        q = torch.stack([q[...,1], q[...,2], q[...,3], q[...,0]], dim=-1)

    return q
