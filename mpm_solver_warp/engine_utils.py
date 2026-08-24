import numpy as np
import h5py
import os
import sys
import warp as wp
import torch
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

def save_data_at_frame(mpm_solver, dir_name, frame, save_to_ply=True, save_to_h5=False):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)
    os.chmod(dir_name, 0o777)

    fullfilename = dir_name + "/sim_" + str(frame).zfill(10) + ".h5"

    if save_to_ply:
        particle_position_to_ply(mpm_solver, fullfilename[:-2] + "ply")

    if save_to_h5:

        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        newFile = h5py.File(fullfilename, "w")

        x_np = (
            mpm_solver.mpm_state.particle_x.numpy().transpose()
        )  # x_np has shape (3, n_particles)
        newFile.create_dataset("x", data=x_np)  # position

        currentTime = np.array([mpm_solver.time]).reshape(1, 1)
        newFile.create_dataset("time", data=currentTime)  # current time

        f_tensor_np = (
            mpm_solver.mpm_state.particle_F.numpy().reshape(-1, 9).transpose()
        )  # shape = (9, n_particles)
        newFile.create_dataset("f_tensor", data=f_tensor_np)  # deformation grad

        v_np = (
            mpm_solver.mpm_state.particle_v.numpy().transpose()
        )  # v_np has shape (3, n_particles)
        newFile.create_dataset("v", data=v_np)  # particle velocity

        C_np = (
            mpm_solver.mpm_state.particle_C.numpy().reshape(-1, 9).transpose()
        )  # shape = (9, n_particles)
        newFile.create_dataset("C", data=C_np)  # particle C
        
        # print("save siumlation data at frame ", frame, " to ", fullfilename)
    if frame == 0:
        print("save siumlation data to ", dir_name)

def particle_position_to_ply(mpm_solver, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.mpm_state.particle_x.numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        # print("write", filename)
        
#     with open(filename[:-4]+'_bone.ply', "wb") as f:  # write binary only bone
#         header = f"""ply
# format binary_little_endian 1.0
# element vertex {74496}
# property float x
# property float y
# property float z
# end_header
# """
#         f.write(str.encode(header))
#         f.write(position[:74496].tobytes())
#         print("write", filename[:-4]+'_bone.ply')


def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

######################################################3

def export_grid_vertor(state, save_path1, save_path2, arrow_head_length=0.005, wedge_angle=np.radians(30)):
    # vec: [200, 200, 200, 3]
    vec1 = state.grid_v_check.numpy()
    vec2 = state.grid_v_check2.numpy()
    print(vec1.max())
    print(vec2.max())
    
    # pos 계산: state.grid_v_mean_pos의 shape: [200, 200, 200, n_index, 3]
    a = state.grid_v_mean_pos.numpy() 
    b = state.grid_v_particle_num.numpy()  # shape: [200,200,200,n_index]
    b = np.expand_dims(b, axis=4)           # shape: [200,200,200,n_index,1]
    b[b == 0] = 1                          # 0이면 1로 대체 (나눗셈 방지)
    cc = a / b                             # shape: [200,200,200,n_index,3]
    mask = np.any(cc != 0, axis=-1)         # shape: [200,200,200,n_index]
    pos = cc[mask]                         # pos: [N, 3]
    
    # mask의 인덱스: (i, j, k, n_index) 순서의 tuple
    indices = np.nonzero(mask)
    
    # vec1
    for vec, save_path in zip([vec1, vec2], [save_path1, save_path2]):
        
        # vec: [200,200,200,3]에서 mask에 해당하는 값 추출 → shape: [N, 3]
        vec = vec[indices[0], indices[1], indices[2], :]
        # 단위 방향 벡터
        vec_norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-11)
        
        # 고정된 random seed를 통해 reproducible한 색상 선택 (grid의 n_index 별 색상)
        np.random.seed(888)
        indices_colors = np.random.randint(0, 256, size=(b.shape[3], 3), dtype=np.uint8)
        
        # 선분(shaft)의 고정 길이
        fixed_length = 0.003
        arrow_head_length = 0.0015
        
        N = pos.shape[0]  # 벡터 개수
        vertices = []
        edges = []
    
        for i in range(N):
            start = pos[i]
            # shaft의 끝점
            end = pos[i] + fixed_length * vec_norm[i]
            
            # 시작점 색상은 grid 내 n_index (indices[3])로 결정
            start_idx = indices[3][i]
            start_color = indices_colors[start_idx]  # 고정 색상
            
            # arrow head는 두 edge를 만듭니다.
            # arrow head의 방향은 shaft의 반대 방향으로, 그리고 좌우로 약간 회전한 방향입니다.
            # 우선, vec_norm[i]와 수직인 임의의 벡터 u를 구합니다.
            v = vec_norm[i]
            # v와 거의 평행하지 않은 벡터 선택 (예: (0,0,1) 또는 (1,0,0))
            arbitrary = np.array([0,0,1]) if abs(v[2]) < 0.99 else np.array([1,0,0])
            u = np.cross(v, arbitrary)
            u = u / (np.linalg.norm(u) + 1e-11)
            
            # arrow head의 두 edge는 shaft의 끝점에서, 
            # v의 반대 방향과 u 방향의 성분을 혼합하여 만듭니다.
            # 기본: head_dir = cos(wedge_angle)*v + sin(wedge_angle)*u, (뒤로 향하도록 -)
            head_dir_left  = - (np.cos(wedge_angle) * v + np.sin(wedge_angle) * u)
            head_dir_right = - (np.cos(wedge_angle) * v - np.sin(wedge_angle) * u)
            
            left = end + arrow_head_length * head_dir_left
            right = end + arrow_head_length * head_dir_right
            
            # 끝점(shaft의 끝) 색상은 방향 벡터 정보를 이용해 계산: 
            # 여기서는 vec_norm를 [0,1]로 변환 후 0~255 scale
            end_color = ((v + 1) / 2 * 255).astype(np.uint8)
            
            # 각 arrow에 대해 vertex 구성:
            # 0: start (shaft 시작)
            # 1: end   (shaft 끝, arrow head 시작)
            # 2: left  (arrow head 왼쪽 점)
            # 3: right (arrow head 오른쪽 점)
            vertices.append((start[0], start[1], start[2],
                            int(start_color[0]), int(start_color[1]), int(start_color[2])))
            vertices.append((end[0], end[1], end[2],
                            int(start_color[0]), int(start_color[1]), int(start_color[2])))
                            #  int(end_color[0]), int(end_color[1]), int(end_color[2])))
            vertices.append((left[0], left[1], left[2],
                            int(start_color[0]), int(start_color[1]), int(start_color[2])))
                            #  int(end_color[0]), int(end_color[1]), int(end_color[2])))
            vertices.append((right[0], right[1], right[2],
                            int(start_color[0]), int(start_color[1]), int(start_color[2])))
                            #  int(end_color[0]), int(end_color[1]), int(end_color[2])))
            
            base_idx = 4 * i
            # edge 구성:
            # shaft: start -> end
            # arrow head: end -> left, end -> right
            edges.append((base_idx, base_idx + 1))
            edges.append((base_idx + 1, base_idx + 2))
            edges.append((base_idx + 1, base_idx + 3))
        
        vertex_count = len(vertices)
        edge_count = len(edges)
        
        with open(save_path, 'w') as f:
            # PLY 헤더 작성 (ASCII 포맷)
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("comment author: Your Name\n")
            f.write("comment object: grid vector field with arrow head\n")
            f.write("element vertex {}\n".format(vertex_count))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element edge {}\n".format(edge_count))
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            # vertex 데이터 작성
            for v in vertices:
                f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                    v[0], v[1], v[2], v[3], v[4], v[5]))
            
            # edge 데이터 작성
            for e in edges:
                f.write("{} {}\n".format(e[0], e[1]))