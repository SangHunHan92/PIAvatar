import sys
import os
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils import *
from warp_utils_separable_contact import *
from mpm_utils_separable_contact import *
from mpm_human_utils_separable_contact import *
# from mpm_human_utils_hair import *
# from mpm_human_utils_haircloth import *

import pytorch3d
import torch.nn.functional as F
import trimesh

os.environ["CUDA_LAUNCH_BLOCKING"]= "1"

class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, n_subjects=1, n_humans=1, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, n_subjects, n_humans, device=device)
        self.time_profile = {}
        
        check_time = dict()
        check_time = {'time1': 0.0, 'time2': 0.0, 'time3': 0.0, 'time4': 0.0, 'time5': 0.0,
                      'time6': 0.0, 'time7': 0.0, 'time8': 0.0, 'time9': 0.0, 'time_total': 0.0,
                      'time111': 0.0, 'time10': 0.0, 'step':0 }
        self.check_time = check_time

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, n_subjects=1, n_humans=1, device="cuda:0"):
        self.n_particles = n_particles
        
        self.hun_time = 0
        self.device = device

        self.mpm_model = MPMModelStruct()
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.n_subjects = n_subjects
        self.mpm_model.n_humans = n_humans
        
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        self.mpm_model.E = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)

        self.mpm_model.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = 0
        self.mpm_model.material_list = wp.zeros(shape=1, dtype=int, device=device)

        self.mpm_model.plastic_viscosity = 0.0
        self.mpm_model.softening = 0.1
        self.mpm_model.yield_stress = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied
        
        self.mpm_model.material_list
        
        # self.mpm_model.n_slot = 8

        self.mpm_state = MPMStateStruct()

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity
        
        # self.mpm_state.particle_v_given = wp.zeros(
        #     shape=n_particles, dtype=wp.vec3, device=device
        # )  # human particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic
        self.mpm_state.particle_F_before = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix
        
        self.mpm_state.particle_quat = wp.zeros(
            shape=n_particles, dtype=wp.vec4f, device=device
        )
        self.mpm_state.particle_scale = wp.zeros(
            shape=n_particles, dtype=wp.vec3f, device=device
        )

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield
        
        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        
        self.mpm_state.particle_selection = wp.zeros(    # maybe not use
            shape=n_particles, dtype=int, device=device
        )
        
        ####### for avatar stress
        self.mpm_state.particle_gravity = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )
        self.mpm_state.particle_Fe = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        # self.mpm_state.particle_Fe = wp.full(
        #     shape=n_particles, value=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), dtype=wp.mat33, device=device
        # )
        
        self.mpm_state.particle_F_add = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        # self.mpm_state.particle_Fe_add = wp.full(
        #     shape=n_particles, value=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), dtype=wp.mat33, device=device
        # )
        self.mpm_state.particle_Fe_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  
        self.mpm_state.particle_Fk = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  
        self.mpm_state.particle_id = wp.full(
            shape=n_particles,
            value=-1,
            dtype=int,
            device=device,
        )
        
        self.mpm_state.particle_vk = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity
        self.mpm_state.particle_vko = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity
        self.mpm_state.particle_vSM = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )
        self.mpm_state.particle_LBS = wp.zeros(
            shape=(n_particles, 55), dtype=wp.float32, device=device
        )
        self.mpm_state.particle_SM_test = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )
        
        
        self.mpm_state.particle_material = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        ####### for avatar stress
        # self.mpm_model.n_humans = n_humans
        # self.mpm_model.n_subjects = n_subjects
        self.mpm_state.grid_f = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_humans),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_vk = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_humans),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_mk = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_humans),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_id = wp.full(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_humans),
            value=-1,
            dtype=int,
            device=device,
        )
        self.mpm_state.grid_count = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=int,
            device=device,
        )

        ####### separable-contact (Bardenhagen 2000): per-subject grid arrays
        # disabled by default; turn on via set_separable_contact(True)
        self.mpm_model.use_separable_contact = 0
        self.mpm_model.contact_eps = 1e-15
        ng = self.mpm_model.n_grid
        ns = self.mpm_model.n_subjects
        self.mpm_state.grid_p_in_s = wp.zeros(
            shape=(ng, ng, ng, ns), dtype=wp.vec3, device=device,
        )
        self.mpm_state.grid_ms = wp.zeros(
            shape=(ng, ng, ng, ns), dtype=float, device=device,
        )
        self.mpm_state.grid_v_subj = wp.zeros(
            shape=(ng, ng, ng, ns), dtype=wp.vec3, device=device,
        )
        self.mpm_state.grid_v_resolved = wp.zeros(
            shape=(ng, ng, ng, ns), dtype=wp.vec3, device=device,
        )
        self.mpm_state.grid_normal = wp.zeros(
            shape=(ng, ng, ng, ns), dtype=wp.vec3, device=device,
        )
        
        # num of bones = 20
        self.mpm_state.bone_mx = wp.zeros(
            shape=(n_humans, 20), dtype=wp.vec3, device=device
            # shape=(n_humans, 20), dtype=wp.vec3d, device=device
        )
        self.mpm_state.bone_mv = wp.zeros(
            shape=(n_humans, 20), dtype=wp.vec3, device=device
        )
        self.mpm_state.bone_m = wp.zeros(
            shape=(n_humans, 20), dtype=wp.float32, device=device
        )
        self.mpm_state.bone_L = wp.zeros(
            shape=(n_humans, 20), dtype=wp.vec3, device=device
        )
        self.mpm_state.bone_I = wp.zeros(
            shape=(n_humans, 20), dtype=wp.mat33, device=device
        )
        self.mpm_state.bone_w = wp.zeros(
            shape=(n_humans, 20), dtype=wp.vec3, device=device
        )
        self.mpm_state.bone_A = wp.zeros(
            shape=(n_humans, 20), dtype=wp.mat33, device=device
        )
        self.mpm_state.bone_Apose = wp.zeros(
            shape=(n_humans, 20), dtype=wp.mat44, device=device
        )
        self.mpm_state.bone_R = wp.zeros(
            shape=(n_humans, 20), dtype=wp.mat33, device=device
        )              
        self.mpm_state.bone_x0 = wp.zeros(
            shape=(n_humans, 74496), dtype=wp.vec3, device=device,
        )
        self.mpm_state.bone_x0cm = wp.zeros(
            shape=(n_humans, 20), dtype=wp.vec3, device=device,
        )
        self.mpm_state.bone_q = wp.zeros(
            shape=(n_humans, 74496), dtype=wp.vec3, device=device,
        )
        self.mpm_state.bone_idx = wp.full(
            shape=n_particles, value=-1, dtype=wp.int16, device=device
        )      
        bone_p_num = np.array([            
            4495, 4454, 4159, 2949, 3886, 
            1351, 1351, 28309, 1079, 1079, 
            516, 516, 3487, 3487, 941, 
            845, 941, 845, 4903, 4903
        ])
        self.mpm_state.bone_pnum = wp.array(bone_p_num, dtype=wp.int16, device=device)
        self.bone_p_num = torch.tensor(bone_p_num, device=device)
        self.mpm_state.avatar_offset = wp.zeros(
            shape=1, dtype=int, device=device,
        )

        # bone index mapping
        bone_starts = [
                 0, 
              4495,  8949, 13108, 16057, 19943, 
             21294, 22645, 50954, 52033, 53112, 
             53628, 54144, 57631, 61118, 62059, 
             62904, 63845, 64690, 69593, 74496]    
            
        # bone_index = np.zeros(bone_starts[-1], dtype=np.int32)        
        # for i in range(len(bone_starts) - 1):
        #     start = bone_starts[i]
        #     end = bone_starts[i + 1]
        #     bone_index[start:end] = i
        
        # self.mpm_model.bone_index = wp.array(bone_index, dtype=int, device=device)
        self.mpm_model.bone_index = wp.array(bone_starts, dtype=int, device=device)
        
        if 0:
            self.mpm_state.grid_v_mean_pos = wp.zeros(
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_particle_num = wp.zeros(
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                dtype=int,
                device=device,
            )
            self.mpm_state.grid_v_in_prescribed = wp.zeros(
                # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_out_prescribed = wp.zeros(
                # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_check = wp.zeros(
                # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_check2 = wp.zeros(
                # shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans + 1),
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_in_human = wp.zeros(
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans),
                dtype=wp.vec3,
                device=device,
            )
            self.mpm_state.grid_v_out_human = wp.zeros(
                shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, n_humans),
                dtype=wp.vec3,
                device=device,
            )

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []
        
        self.mpm_model.method = 0
        
        ####### for avatar stress
        self.human_modify_params = [] # wp, human velocity
        self.human_modify_model = []  # torch, cano, lbs        
        self.human_modify_changer = [] # wp kernel
        self.human_modify_applier = [] # wp kernel        
        self.human_step = -1
        
    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of pos is (n, 3); shape of vol is (n,)
    def load_initial_data_from_torch(
        self,
        pos,
        vol,        
        index,
        cov=None,
        human_particle=None,
        n_grid=100,
        grid_lim=1.0,
        n_subjects=1,
        n_humans=1,
        device="cuda:0",
    ):
        self.dim, self.n_particles = pos.shape[1], pos.shape[0]
        assert pos.shape[0] == vol.shape[0]
        # assert pos.shape[0] == cov.reshape(-1, 6).shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, n_subjects, n_humans, device=device) # why? from original code

        self.import_particle_x_from_torch(pos, device)
        
        self.mpm_state.particle_vol = wp.from_numpy(
            vol.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        self.mpm_state.particle_id = wp.from_numpy(
            index.detach().clone().cpu().numpy(), dtype=int, device=device
        )
        # self.mpm_state.particle_human = wp.from_numpy(
        #     w.detach().clone().cpu().numpy(), dtype=bool, device=device
        # )
        
        if cov is not None: # True
            self.mpm_state.particle_init_cov = wp.from_numpy(
                cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )
            if self.mpm_model.update_cov_with_F: # False
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov
        
        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )

        # initial deformation gradient is set to identity
        # initial trial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_Fe],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_Fe_trial],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_Fk],
            device=device,
        )
        
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid       
        
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_subjects),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_subjects),
            dtype=wp.vec3,
            device=device,
        )

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)
        print("Num of subjects: ", self.mpm_model.n_subjects)
        print("Num of humans: ", self.mpm_model.n_humans)
        # print("Cloth particles: ", np.sum(self.mpm_state.human_particle.numpy() == 0))      # 0
        # print("Human particles: ", np.count_nonzero(self.mpm_state.human_particle.numpy())) # 1

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def set_simulater_parameters(self, sim_params, method="ours", device="cuda:0"):
        if "g" in sim_params.keys():
            self.mpm_model.gravitational_accelaration = wp.vec3(
            sim_params["g"][0], sim_params["g"][1], sim_params["g"][2]
            )
        if "penalty_d" in sim_params.keys():
            self.mpm_model.penalty_d = sim_params["penalty_d"]
        if "penalty_v" in sim_params.keys():
            self.mpm_model.penalty_v = sim_params["penalty_v"]
        if "penalty_th" in sim_params.keys():
            self.mpm_model.penalty_th = sim_params["penalty_th"]
            
        if method == "ours":
            self.mpm_model.method = 0
        elif method == "vanila":
            self.mpm_model.method = 1
        else:
            self.mpm_model.method = 0

        # separable-contact (Bardenhagen 2000 multi-field MPM)
        if "use_separable_contact" in sim_params.keys():
            self.mpm_model.use_separable_contact = int(bool(sim_params["use_separable_contact"]))
        if "contact_eps" in sim_params.keys():
            self.mpm_model.contact_eps = float(sim_params["contact_eps"])

    # hun
    def set_subjects_parameters(self, subject_params, device="cuda:0"):
        
        # read material list
        material_list = []
        for params in subject_params:
            if params["material"] == "jelly":
                material = 0
            elif params["material"] == "metal":
                material = 1
            elif params["material"] == "sand":
                material = 2
            elif params["material"] == "foam":
                material = 3
            elif params["material"] == "snow":
                material = 4
            elif params["material"] == "plasticine":
                material = 5
            else:
                raise TypeError("Undefined material type")
            material_list.append(material)
            
        material_list = list(set(material_list))
        
        # Material
        for index, material in enumerate(material_list):
            # with wp.ScopedTimer(
            #     "set_material_paramater",
            #     synchronize=True,
            #     print=True,
            #     dict=self.time_profile,
            # ):
            # self.mpm_state.particle_id.numpy()
            wp.launch(
                kernel=set_material_paramater,
                dim=self.n_particles,
                inputs=[self.mpm_state, index, material],
                device=device,
            )
        
        material_list = list(set(material_list)) # remain only unique material   
        self.mpm_model.material_list = wp.array(material_list, dtype=int, device=device) 
            
        # velocity, density, E, nu, g, 기타 등등?
        for index, subject in enumerate(subject_params):
            # with wp.ScopedTimer(
            #     "set_subject_E",
            #     synchronize=True,
            #     print=True,
            #     dict=self.time_profile,
            # ):
            wp.launch(
                kernel=set_subject_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_id, self.mpm_model.E, index, subject["E"]],
                device=device,
            )
            wp.launch(
                kernel=set_subject_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_id, self.mpm_model.nu, index, subject["nu"]],
                device=device,
            )
            if "yield_stress" in subject.keys():
                wp.launch(
                    kernel=set_subject_value_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_model.yield_stress, index, subject["yield_stress"]],
                    device=device,
                )
            # if "hardening" in subject.keys():
            #     self.mpm_model.hardening = total_param["hardening"]
            # if "xi" in subject.keys():
            #     self.mpm_model.xi = total_param["xi"]
            # if "friction_angle" in subject.keys():
            #     self.mpm_model.friction_angle = total_param["friction_angle"]
            #     sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            #     self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            # if "rpic_damping" in subject.keys():
            #     self.mpm_model.rpic_damping = total_param["rpic_damping"]
            # if "plastic_viscosity" in subject.keys():
            #     self.mpm_model.plastic_viscosity = total_param["plastic_viscosity"]
            # if "softening" in subject.keys():
            #     self.mpm_model.softening = total_param["softening"]
            # if "grid_v_damping_scale" in subject.keys():
            #     self.mpm_model.grid_v_damping_scale = total_param["grid_v_damping_scale"]
            
            # 나중에 고치자...
            # if "g" in subject.keys():
            #     self.mpm_model.gravitational_accelaration = wp.vec3(
            #     subject["g"][0], subject["g"][1], subject["g"][2]
            #     )
            
            # g init 선언, zeros로 초기화, #4에 g 더해지게끔 설정
            
            if "g" in subject.keys():
                wp.launch(
                    kernel=set_subject_vec_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_state.particle_gravity, index, subject["g"]],
                    device=device,
                )                
               
            if "density" in subject.keys():
                wp.launch(
                    kernel=set_subject_value_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_state.particle_density, index, subject["density"]],
                    device=device,
                )
                
            if "initial_velocity" in subject.keys():
                wp.launch(
                    kernel=set_subject_vec_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_state.particle_v, index, subject["initial_velocity"]],
                    device=device,
                )
             
        # temp code for hair dress #############################################################################################################################
        if 0:
            mask_in_avatar = torch.load("./test_data/mask_hairdress.pt").int()        
            mask_in_avatar = torch.cat([torch.zeros(74496, dtype=torch.int32, device=mask_in_avatar.device), mask_in_avatar])
            mask_in_avatar *= 99
            mask_in_avatar = mask_in_avatar.cpu().numpy()
            particle_id = self.mpm_state.particle_id.numpy()
            particle_id[:411996] = mask_in_avatar # set hair, dress == 99
            self.mpm_state.particle_id = wp.array(particle_id, dtype=int, device=device)         
            
            wp.launch(
                    kernel=set_subject_value_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_model.E, 99, 20.0],
                    device=device,
                )
            wp.launch(
                kernel=set_subject_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_id, self.mpm_model.nu, 99, 0.15],
                device=device,
            )
            if "g" in subject.keys():
                wp.launch(
                    kernel=set_subject_vec_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_state.particle_gravity, 99, [0.0, 0.0, -0.1]],
                    device=device,
                )                
                
            if "density" in subject.keys():
                wp.launch(
                    kernel=set_subject_value_to_float_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_id, self.mpm_state.particle_density, 99, 5],
                    device=device,
                )            
            wp.launch(
                kernel=set_material_paramater,
                dim=self.n_particles,
                inputs=[self.mpm_state, 99, 2],
                device=device,
            )
                
            particle_id[:411996] = np.zeros_like(mask_in_avatar) # set hair, dress == 99
            self.mpm_state.particle_id = wp.array(particle_id, dtype=int, device=device)      
        
        ##############################################################################################################################
           
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )        
        
        # yield_stress, friction_angle, alpha, gravitational_accelaration, hardening, xi, plastic_viscosity, softening
        # rpic_damping, grid_v_damping_scale
        
        return 0

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                self.mpm_model.material = 0
            elif kwargs["material"] == "metal":
                self.mpm_model.material = 1
            elif kwargs["material"] == "sand":
                self.mpm_model.material = 2
            elif kwargs["material"] == "foam":
                self.mpm_model.material = 3
            elif kwargs["material"] == "snow":
                self.mpm_model.material = 4
            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5
            else:
                raise TypeError("Undefined material type")

        # if "grid_lim" in kwargs:
        #     self.mpm_model.grid_lim = kwargs["grid_lim"]
        # if "n_grid" in kwargs:
        #     self.mpm_model.n_grid = kwargs["n_grid"]
            
        # self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        # self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        # self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        # (
        #     self.mpm_model.dx,
        #     self.mpm_model.inv_dx,
        # ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
        #     self.mpm_model.n_grid / self.mpm_model.grid_lim
        # )
        # self.mpm_state.grid_m = wp.zeros(
        #     shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
        #     dtype=float,
        #     device=device,
        # )
        # self.mpm_state.grid_v_in = wp.zeros(
        #     shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
        #     dtype=wp.vec3,
        #     device=device,
        # )
        # self.mpm_state.grid_v_out = wp.zeros(
        #     shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
        #     dtype=wp.vec3,
        #     device=device,
        # )

        if "E" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.E, kwargs["E"]],
                device=device,
            )
        if "nu" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.nu, kwargs["nu"]],
                device=device,
            )
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(
                kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]
            )

        if "human_density" in kwargs or "cloth_density" in kwargs:
            if "human_density" in kwargs:
                density_value = kwargs["human_density"] # human_particle = 1
                wp.launch(
                    kernel=set_value_to_float_array_condition,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_density, self.mpm_state.human_particle, density_value, 1],
                    device=device,
                )
            if "cloth_density" in kwargs:
                density_value = kwargs["cloth_density"] # cloth_particle = 0
                wp.launch(
                    kernel=set_value_to_float_array_condition,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_density, self.mpm_state.human_particle, density_value, 0],
                    device=device,
                )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
            
        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                param_modifier.E = params["E"]
                param_modifier.nu = params["nu"]
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

    @wp.kernel
    def set_human_velocity_zero(state: MPMStateStruct, model: MPMModelStruct):
        p = wp.tid()
        if state.particle_id[p] < model.n_humans:
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
    
    def kabsch(self, P, Q):
    
        # 1) 중심화
        P_mean = torch.mean(P, axis=0) # bone_cano
        Q_mean = torch.mean(Q, axis=0) # particle_x_ori
        P_centered = P - P_mean
        Q_centered = Q - Q_mean

        # 2) 상관행렬(Correlation matrix)
        C = Q_centered.T @ P_centered
        # C = np.dot(Q_centered.T, P_centered)

        # 3) SVD
        # U, S, Vt = np.linalg.svd(C)
        U, S, Vt = torch.linalg.svd(C)
        
        # 4) R 계산
        # R_ = np.dot(U, Vt)
        R_ = U @ Vt
        # 반사(reflection) 제거 보정: det(R_)이 음수라면 마지막 축 부호 반전
        # if np.linalg.det(R_) < 0:
        if torch.linalg.det(R_) < 0:
            U[:, -1] *= -1
            R_ = U @ Vt
            # R_ = np.dot(U, Vt)
            
        # 5) t 계산
        t_ = Q_mean - R_ @ P_mean

        return R_, t_
        
    def compute_manual_gradients_parallel(self, points, knn_indices, L_target):
        """
        벡터화된 방식으로 points의 gradient를 계산하는 함수.
        
        Args:
            points (torch.Tensor): (N, 3) 형태의 포인트 클라우드
            knn_indices (torch.Tensor): (N, k) 형태의 KNN 인덱스
            L_target (torch.Tensor): (N, k) 형태의 목표 거리
        
        Returns:
            gradients (torch.Tensor): (N, 3) 형태의 manually computed gradients
        """
        N, k = knn_indices.shape  # 점 개수, 이웃 개수
        gradients = torch.zeros_like(points)  # (N, 3) 크기의 gradient 저장할 텐서
        
        # 현재 점의 좌표
        v1 = points.unsqueeze(1)  # (N, 1, 3)
        v2 = points[knn_indices]  # (N, k, 3)
        
        # 유클리드 거리 계산 (N, k)
        knn_distances = torch.norm(v1 - v2, dim=2, keepdim=True)  # (N, k, 1)
        
        # 거리 미분 (∂d/∂v) 계산
        diff = v1 - v2  # (N, k, 3)
        grad_d = diff / (knn_distances + 1e-8)  # (N, k, 3), 0으로 나누는 오류 방지
        
        # Loss 미분 적용: 2(d - L_target) * (∂d/∂v)
        L_diff = (knn_distances.squeeze(-1) - L_target).unsqueeze(-1)  # (N, k, 1)
        grad_L = 2 * L_diff * grad_d  # (N, k, 3)
        
        # 이웃 포인트들에 대한 gradient 업데이트
        for i in range(k):
            gradients.index_add_(0, knn_indices[:, i], -grad_L[:, i])  # 이웃 점들에 대한 반대 방향 적용
        
        gradients += grad_L.sum(dim=1)  # (N, 3)
        
        return gradients / k  # 평균화하여 크기 정규화
    
    # hun : this !!
    def p2g2p(self, frame, step, dt, mpm_params, device="cuda:0", smplx_dt=4e-2, is_3d_measure=False, f=None, sim_params=None):
        
        # wp.config.mode = "debug"
        # wp.config.verify_cuda = True
        # wp.config.verify_fp = True
        time_total = time.time()
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            # self.mpm_model.n_humans + 1, # 0 for global, others for humans
        )
        grid_size_nsub = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            self.mpm_model.n_humans, # 0 for global, others for humans
        )
        # separable-contact: per-subject launch shape (last dim = n_subjects)
        grid_size_subj = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            self.mpm_model.n_subjects,
        )
        # 3D launch for contact resolution
        grid_size_3d = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )

        time1 = time.time()
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size_nsub),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        ) # 1
        if self.mpm_model.use_separable_contact == 1:
            wp.launch(
                kernel=zero_grid_separable,
                dim=grid_size_subj,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )
        time1 = time.time() - time1
        
        # Human Velocity
        time111 = time.time()
        human_step = int(round(self.time / smplx_dt))
        if step == 0:
        # if self.human_step != human_step : # step이랑 smpl step이랑 다름
        # if 0 : # step이랑 smpl step이랑 다름
            # self.mpm_state.particle_v.numpy()[:5]
            # self.mpm_state.particle_vk.numpy()[:5]
            # self.mpm_state.particle_vko.numpy()[:5]
            self.human_step = human_step
            maintain_avatar_shape = True 
            
            if human_step == 0:
                lbs_temp = torch.zeros([self.mpm_state.particle_x.shape[0], 55], device=device)
            for k in range(len(self.human_modify_changer)): 
                with wp.ScopedTimer("to_torch_particle_x", synchronize=True, print=False, dict=self.time_profile):
                    particle_x = warp.to_torch(self.mpm_state.particle_x) # 0.07~0.12ms
                if self.human_modify_model[k].model_type == 'animatable_gaussians':
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM, rot_mats_next_total = compute_animatable_gaussians_velocity_rel(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f)
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM, rot_mats_next_total = compute_animatable_gaussians_velocity_tgt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f) # 0.11s                    
                    elif self.human_modify_model[k].velocity_type == 'gt':
                        velocity, rel_rot_mats, lbs, vSM = compute_animatable_gaussians_velocity_gt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f) # 0.11s                    
                elif self.human_modify_model[k].model_type == 'smplx':
                    rot_mats_next_total = None
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_rel(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params)
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_tgt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params)
                    elif self.human_modify_model[k].velocity_type == 'gt':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_gt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure, f=f, sim_params=sim_params) # 0.11s                    
                else:
                    assert False, "No velocity function"
                torch.cuda.synchronize()  # CUDA 연산을 동기화하여 이전 오류 감지
                torch.cuda.empty_cache()                
                if 0:
                    avt_off1 = self.mpm_state.avatar_offset.numpy()[k]
                    avt_off2 = self.mpm_state.avatar_offset.numpy()[k+1]
                    lbs_temp[avt_off1:avt_off2] = lbs                                                                    
                with wp.ScopedTimer("from_torch", synchronize=True, print=False, dict=self.time_profile):
                    new_human_velocity = wp.array(velocity.detach().cpu().numpy(), dtype=wp.vec3) # 4ms, 오래 걸리긴한데 일단 돌리자...
                    rel_rot_mats = wp.array(rel_rot_mats.detach().cpu().numpy(), dtype=wp.mat33)
                    vSM = wp.array(vSM.detach().cpu().numpy(), dtype=wp.vec3)
                    if rot_mats_next_total is not None:
                        rot_mats_next_total = wp.array(rot_mats_next_total.detach().cpu().numpy(), dtype=wp.mat33)
                wp.synchronize()
                wp.launch( # add velocity
                    kernel=self.human_modify_changer[k],
                    dim=self.human_modify_model[k].human_n_particles,
                    inputs=[self.mpm_state, self.human_modify_params[k], new_human_velocity, rel_rot_mats, 0, vSM, frame], # vSM은 뼈에는 적용하면 안된다
                    # inputs=[self.mpm_state, self.human_modify_params[k], new_human_velocity, rel_rot_mats, 0, vSM, frame, rot_mats_next_total], # vSM은 뼈에는 적용하면 안된다
                    device=device,
                )
                wp.synchronize()
            if human_step == 0:
                with wp.ScopedTimer("LBS", synchronize=True, print=False, dict=self.time_profile):
                    self.mpm_state.particle_LBS = wp.array(lbs_temp.detach().cpu().numpy(), dtype=wp.float32, ndim=2)
        time111 = time.time() - time111
              
        # apply pre-p2g operations on particles
        # None for pillow2sofa example
        time2 = time.time()
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            ) # 2
        time2 = time.time() - time2
        
        # 3
        # apply dirichlet particle v modifier
        # 특정 particles에 init으로 mask 적용을 하고, 
        # 지정 시간동안 state.paricle_v=particle_velocity_modifier_params.velocity 설정
        # pillow2sofa example은 1개
        time3 = time.time()
        for k in range(len(self.particle_velocity_modifiers)):
            # start_time2 = time.time()
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        time3 = time.time() - time3

        if 0: # step == 0:
            print("\n")
            print(self.mpm_state.particle_F_trial.numpy().max(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().min(axis=0))
            print(self.mpm_state.particle_F.numpy().max(axis=0))
            print(self.mpm_state.particle_F.numpy().min(axis=0))
            print(self.mpm_state.particle_stress.numpy().max(axis=0))
            print(self.mpm_state.particle_stress.numpy().min(axis=0))
            print("model material : ", self.mpm_model.material)

        # 4, # compute stress = stress(returnMap(F_trial))        
        time4 = time.time()
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt], # state, model in function
                device=device,
            )  # F and stress are updated
        time4 = time.time() - time4
        
        if 0:
            print("\n")
            print(self.mpm_state.particle_x.numpy().max(axis=0), self.mpm_state.particle_x.numpy().min(axis=0))
            print(self.mpm_state.particle_v.numpy().max(axis=0), self.mpm_state.particle_v.numpy().min(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().max(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().min(axis=0))
            print(self.mpm_state.particle_F.numpy().max(axis=0))
            print(self.mpm_state.particle_F.numpy().min(axis=0))
            print(self.mpm_state.particle_C.numpy().max(axis=0))
            print(self.mpm_state.particle_C.numpy().min(axis=0))
            print(self.mpm_state.particle_stress.numpy().max(axis=0))
            print(self.mpm_state.particle_stress.numpy().min(axis=0))
            print("\n")

        # 5, p2g
        time5 = time.time()
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # 5, # apply p2g
            wp.synchronize()
        time5 = time.time() - time5
        # self.mpm_state.grid_v_out.numpy()
        # self.mpm_state.grid_v_in.numpy().max()
        # self.mpm_state.grid_vk.numpy().max()

        # 6, grid update
        time6 = time.time()
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                # dim=(grid_size),
                dim=(grid_size_nsub),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
            # separable-contact: per-subject grid v_subj = p_in_s/ms + dt·g
            if self.mpm_model.use_separable_contact == 1:
                wp.launch(
                    kernel=grid_normalization_and_gravity_separable,
                    dim=grid_size_subj,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
        time6 = time.time() - time6
        # self.mpm_state.grid_v_out.numpy()
        
        # threshold = self.mpm_model.dx, 0.01        
        # self.mpm_state.particle_x.shape = 957730
        # self.mpm_state.grid_v_particle_num.numpy().sum() # 957730
        # self.mpm_state.grid_v_particle_num.numpy().max() # 5771
        
        # if step == 0 and frame > 0:
        if 0:
            a = self.mpm_state.grid_v_mean_pos.numpy() # [:, :, :, n_index, 3]
            b = self.mpm_state.grid_v_particle_num.numpy()
            b = np.expand_dims(self.mpm_state.grid_v_particle_num.numpy(), axis=4) # [:, :, :, n_index, 1]
            b[b == 0] = 1        
            cc = a/b
            for i in range(a.shape[3]):
                c = cc[:, :, :, i, :].reshape(-1, 3)
                c = c[~np.all(c == 0, axis=1)]
                c_mesh = trimesh.Trimesh(vertices=c)
                c_mesh.export(f"./test_data/grid/parts/grid_points_{frame-1:03d}_{i}.ply")          
            cc = cc.reshape(-1, 3)
            cc = cc[~np.all(cc == 0, axis=1)]
            cc_mesh = trimesh.Trimesh(vertices=cc)
            cc_mesh.export(f"./test_data/grid/grid_points_{frame-1:03d}.ply")   
        
        # 67, grid penalty (Multi-Material Contact), grid update front? back?
        
        # time67 = time.time()
        # with wp.ScopedTimer(
        #     "grid_penalty", synchronize=True, print=False, dict=self.time_profile
        # ):
        #     wp.launch(
        #         kernel=grid_penalty,
        #         dim=(grid_size),
        #         inputs=[self.mpm_state, self.mpm_model, dt],
        #         device=device,
        #     )
        # time67 = time.time() - time67
        # wp.synchronize()
        
        save_path1 = f"./test_data/grid/force_vector1_{frame-1:03d}.ply"
        save_path2 = f"./test_data/grid/force_vector2_{frame-1:03d}.ply"
        if 0:
        # if step == 0 and frame > 0:
            # self.mpm_state.grid_v_mean_pos.numpy()
            # self.mpm_state.grid_v_check.numpy()
            export_grid_vertor(self.mpm_state, save_path1, save_path2)
                
        # 7
        time7 = time.time()
        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )
        time7 = time.time() - time7
        # self.mpm_state.grid_v_out.numpy()
        
        # 8, apply BC on grid
        time8 = time.time()
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                # add bounding box
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])
        time8 = time.time() - time8

        # separable-contact resolve (Bardenhagen 2000 free-slip multi-field):
        # only the COMPRESSIVE normal component of (v_k - v_cm) is projected out;
        # tangential motion is preserved, separation is automatically free.
        # then re-apply the same bounding-box BC to grid_v_resolved 4D so particles
        # on the per-subject path also obey the domain walls (the existing 3D BC
        # only touched grid_v_out).
        if self.mpm_model.use_separable_contact == 1:
            wp.launch(
                kernel=compute_vcm_and_resolve,
                dim=grid_size_3d,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )
            wp.launch(
                kernel=apply_bounding_box_separable,
                dim=grid_size_subj,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        # damping이 잘 되는지 테스트
        # step 0에서는 기존 속도와, 적용 속도가 같아야 한다

        # 9, g2p
        # self.mpm_state.particle_id.numpy()
        
        time9 = time.time()
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated
            wp.synchronize()
        time9 = time.time() - time9
        
        numpy_test = False
        if numpy_test:            
            def SM_numpy(self, avatar, bone, device="cuda:0"):                
                avt = self.mpm_state.avatar_offset.numpy()
                bone_pnum = self.mpm_state.bone_pnum.numpy()
                bone_index = self.mpm_model.bone_index.numpy()                
                x_np = self.mpm_state.particle_x.numpy()
                v_np = self.mpm_state.particle_v.numpy()
                F_np = self.mpm_state.particle_F_trial.numpy()
                Fk_np = self.mpm_state.particle_Fk.numpy()
                C_np = self.mpm_state.particle_C.numpy()        
                bone_cano = self.mpm_state.bone_x0.numpy() # [2, 74496, 3]
                bone_cano_c = self.mpm_state.bone_x0cm.numpy()

                srt = avt[avatar] + bone_index[bone]
                end = avt[avatar] + bone_index[bone+1]
                
                # R_, t_가 original world와 MPM world가 같을까
                # ori world  
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                particle_x = warp.to_torch(self.mpm_state.particle_x)
                particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
                bone_cano_ori = ((torch.tensor(bone_cano, device=self.device) - center)/scale + ori_mean) @ rot_mats
                particle_x_ori_np = particle_x_ori.detach().cpu().numpy()
                
                # bone_cano_ori_mesh0 = trimesh.Trimesh(vertices=bone_cano_ori[0].detach().cpu().numpy())
                # bone_cano_ori_mesh0.export(f"./log/ag_FeFk_SM1/bone_cano_ori_mesh0.ply")
                        
                # x_np = x_np.astype(np.float64)
                # bone_cano = bone_cano.astype(np.float64)
                
                # MPM world        
                R_glob, t_glob = self.kabsch(torch.tensor(bone_cano[avatar, srt: end], device=self.device), 
                                    torch.tensor(x_np[srt: end], device=self.device))
                R_loc, t_loc = self.kabsch(torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device), 
                                    torch.tensor(particle_x_ori_np[srt: end], device=self.device))
                
                torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_glob.T + t_glob - torch.tensor(x_np[srt: end], device=self.device)
                torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device) @ R_loc.T + t_loc - torch.tensor(particle_x_ori_np[srt: end], device=self.device)
                
                # numpy나 torch로 shape matching 결과를 확인해보자
                # 1
                x_i = x_np[srt: end]
                x_cm = x_np[srt: end].mean(axis=0)
                x_i0 = bone_cano[avatar, srt: end]
                x_cm0 = bone_cano[avatar, srt: end].mean(axis=0)

                v_i = v_np[srt: end]
                v_cm = v_np[srt: end].mean(axis=0)

                # 2.1
                r_i = x_i - x_cm                
                q_i = x_i0 - x_cm0
                Apq_parts = (r_i[:, :, None] * q_i[:, None, :])
                Apq = Apq_parts.sum(axis=0)
                
                # 2.2
                L = np.cross(r_i, v_i).sum(axis=0) # [3, 3]
                
                r_sq = np.einsum('ij,ij->i', r_i, r_i)   # (N,)  |r_i|²
                I3   = np.eye(3)
                term1 = r_sq[:, None, None] * I3          # (N,3,3)  |r_i|² I₃
                term2 = r_i[:, :, None] * r_i[:, None, :]     # (N,3,3)  r_i r_iᵀ
                I     = np.sum(term1 - term2, axis=0)     # (3,3)
                w     = np.linalg.solve(I, L)  # (3,)  I w = L
                
                (v_cm + np.cross(w, r_i)) # new v_i
                v_i
                np.abs((v_cm + np.cross(w, r_i)) - v_i).max() # CHECK
                
                # 3
                U, S, Vh = np.linalg.svd(Apq)
                UVt  = U @ Vh
                detUV = np.linalg.det(UVt)
                sign = -1.0 if detUV < 0.0 else 1.0
                Dfix = np.eye(3)
                Dfix[-1, -1] = sign
                R = U @ Dfix @ Vh
                
                (q_i @ R.T + x_cm) # new x_i
                x_i
                np.abs((q_i @ R.T + x_cm) - x_i).max() # CHECK
                        
                # Check !!
                # R_.detach().cpu().numpy() - R # 0
                
                # kabsch, local cano to local posed
                # torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_.T + t_ 
                # torch.tensor(x_np[srt: end], device=self.device)
                
                # SM
                # (q_i @ R.T + x_cm)
                # x_i = x_np[srt: end]
                # x_i0 = bone_cano[avatar, srt: end]
                # x_cm0 = bone_cano[avatar, srt: end].mean(axis=0)
                # q_i = x_i0 - x_cm0
                                             
                # return R, x_cm, R_, t_
                return R_glob, t_glob, R_loc, t_loc
            
            R_glob, t_glob, R_loc, t_loc = SM_numpy(self, 0, 0, device=device)
            # R0, x_cm0, R0_, t0_ = SM_numpy(self, 0, 0, device=device)
            # R1, x_cm1, R1_, t1_ = SM_numpy(self, 0, 1, device=device)

            bone_index = self.mpm_model.bone_index.numpy()        
            self.mpm_state.bone_q.numpy()[0, bone_index[1]:bone_index[2]]
            self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]] - self.mpm_state.bone_x0cm.numpy()[0, 1]
            
            x_i = self.mpm_state.particle_x.numpy()[4495:8949] # x_i
            x_cm = self.mpm_state.particle_x.numpy()[4495:8949].mean(axis=0) # x_cm
            x_i0 = self.mpm_state.bone_x0.numpy()[0, 4495:8949] # x_i0
            x_cm0 = self.mpm_state.bone_x0.numpy()[0, 4495:8949].mean(axis=0) # x_cm0
            r_i = x_i - x_cm # r_i
            q_i = x_i0 - x_cm0 # q_i
        
        shape_matching = False
        save_test = False
        
        if 0 and shape_matching and frame==0 and step==0 :
            @wp.kernel
            def add_noise(x: wp.array(dtype=wp.vec3),
                        noise_scale: float):

                i = wp.tid()
                seed = 12777
                rand_state = wp.rand_init(seed + i)
                rx = 2.0 * wp.randf(rand_state) - 1.0
                ry = 2.0 * wp.randf(rand_state) - 1.0
                rz = 2.0 * wp.randf(rand_state) - 1.0

                noise = wp.vec3(rx, ry, rz) * noise_scale
                x[i] += noise

            # 3) 실행
            wp.launch(
                kernel=add_noise,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_v, 0.01],
                device="cuda"
            )
        
        if shape_matching:
            # bone 변수들 zeros
            time10 = time.time()
            with wp.ScopedTimer(
                "zero_bone", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=zero_bone,
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
            
            # time10 = time.time()
            with wp.ScopedTimer(
                "shape_matching_center", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching1_reduce,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            # time10 = time.time() - time10
            
            # time11 = time.time()
            with wp.ScopedTimer(
                "shape_matching_solve", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching2_solve,
                    dim=self.n_particles,
                    # dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            # time11 = time.time() - time11
            
            # time12 = time.time()
            with wp.ScopedTimer(
                "shape_matching_particle", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching3_Rw,
                    # dim=(self.n_particles),
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            # time12 = time.time() - time12
            
            if save_test:                
                bone_trimesh1 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[0:74496],
                )
                bone_trimesh1.export(f"./test_data/shape_matching_before1.ply")
                bone_trimesh2 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[420414:420414+74496],
                )
                bone_trimesh2.export(f"./test_data/shape_matching_before2.ply")
            
            local_global_A_test = False
            if local_global_A_test:     
                bone_R = self.mpm_state.bone_R.numpy()[0,0]
                bone_x0 = self.mpm_state.bone_x0.numpy()[0,0]
                bone_x0cm = self.mpm_state.bone_x0cm.numpy()[0,0]
                                
                # R_glob, t_glob, R_loc, t_loc      
                
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                                
                ## local에서의 A 만드는법 구현하기, 완료 !!!
                rot = rot_mats
                A = rot / scale
                b = (-center/scale + ori_mean) @ rot
                R_loc_test = rot.T @ R_glob @ rot
                t_loc_test = t_glob @ (rot/scale) + b - b @ R_loc_test.T
                
                
                ## 과연 local로 돌리는게 필요할까, global에서 A를 구해서 바로 LBS를 써도 되지 않을까?
                
                avt = self.mpm_state.avatar_offset.numpy()
                bone_index = self.mpm_model.bone_index.numpy()                
                x_np = self.mpm_state.particle_x.numpy()
                bone_cano = self.mpm_state.bone_x0.numpy() # [2, 74496, 3]

                avatar = 1
                bone = 1
                srt = avt[avatar] + bone_index[bone]
                end = avt[avatar] + bone_index[bone+1]
                                
                # ori world  
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                particle_x = warp.to_torch(self.mpm_state.particle_x)
                particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
                bone_cano_ori = ((torch.tensor(bone_cano, device=self.device) - center)/scale + ori_mean) @ rot_mats
                particle_x_ori_np = particle_x_ori.detach().cpu().numpy()
                
                torch.set_printoptions(precision=8)
                R_glob, t_glob = self.kabsch(torch.tensor(bone_cano[avatar, bone_index[bone]: bone_index[bone+1]], device=self.device), 
                                    torch.tensor(x_np[srt: end], device=self.device))
                R_loc, t_loc = self.kabsch(torch.tensor(bone_cano_ori[avatar, bone_index[bone]: bone_index[bone+1]], device=self.device), 
                                    torch.tensor(particle_x_ori_np[srt: end], device=self.device))
                
                # torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_glob.T + t_glob - torch.tensor(x_np[srt: end], device=self.device)
                # torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device) @ R_loc.T + t_loc - torch.tensor(particle_x_ori_np[srt: end], device=self.device)
                                
                M  = self.mpm_state.bone_m.numpy()[avatar, bone] # bone mass
                x_cm = self.mpm_state.bone_mx.numpy()[avatar, bone] * (1.0 / M)  # now bone center
                bone_x0cm = self.mpm_state.bone_x0cm.numpy()[avatar, bone]
                R_glob_SM = self.mpm_state.bone_R.numpy()[avatar, bone] # R_glob
                t_ = x_cm - R_glob_SM @ bone_x0cm # t_glob
                
                self.mpm_state.particle_LBS # 90% 정도가 0.001 미만
                
                lbs = wp.to_torch(self.mpm_state.particle_LBS)
                thresh = 1e-3
                N, K = lbs.shape
                per_row_counts = (lbs >= thresh).sum(dim=1)  # (N,)
                hist = torch.bincount(per_row_counts, minlength=K+1)  # (K+1,)
                
                ####
                # lbs는 avatar load할때 불러오고
                # cano avatar (transformation 적용하고)는 매 frame 마다
                # 1. SM bone to avatar A-matrix
                avt_off = self.mpm_state.avatar_offset.numpy()
                lbs[avt_off[0]:avt_off[1]]
                
            
            # time13 = time.time()
            with wp.ScopedTimer(
                "shape_matching_bone_particle", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching4_bone_particle, # bone & avatar
                    dim=(self.n_particles),
                    # dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            # time13 = time.time() - time13
            time10 = time.time() - time10
            
            # SM test, well done !!
            # if step == 0 :
            #     bone_s = self.human_modify_model[k].particle_start
            #     bone_e = self.human_modify_model[k].particle_start + self.human_modify_model[k].bone_index[-1]
            #     bone_e_smpl = self.human_modify_model[k].particle_start + self.human_modify_model[k].human_n_particles
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_x.numpy()[bone_s:bone_e])
            #     temp_mesh.export('./test_results/bone_noise_{:04d}.ply'.format(frame))
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_SM_test.numpy()[bone_s:bone_e])
            #     temp_mesh.export('./test_results/bone_SM_{:04d}.ply'.format(frame))                
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_x.numpy()[bone_s:bone_e_smpl])
            #     temp_mesh.export('./test_results/bone_noise_full_{:04d}.ply'.format(frame))
            
            # time14 = time.time()
            # with wp.ScopedTimer(
            #     "shape_matching_avatar_particle", synchronize=True, print=False, dict=self.time_profile
            # ):
            #     wp.launch(
            #         kernel=shape_matching5_avatar_particle, # bone & avatar
            #         dim=(self.n_particles),
            #         # dim=(self.mpm_model.n_humans, 20),
            #         inputs=[self.mpm_state, self.mpm_model, dt],
            #         device=device,
            #     )
            #     wp.synchronize()
            # time14 = time.time() - time14            
                        
            if save_test:                
                bone_trimesh1 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[0:74496],
                )
                bone_trimesh1.export(f"./test_data/shape_matching_after1.ply")
                bone_trimesh2 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[420414:420414+74496],
                )
                bone_trimesh2.export(f"./test_data/shape_matching_after2.ply")
                
                bone_index = self.mpm_model.bone_index.numpy() 
                bone_cano = self.mpm_state.bone_x0.numpy()
                self.mpm_state.bone_q.numpy()[0, bone_index[0]:bone_index[1]]
                self.mpm_state.bone_x0.numpy()[0, bone_index[0]:bone_index[1]] - self.mpm_state.bone_x0cm.numpy()[0, 0]
                
                self.mpm_state.bone_q.numpy()[0, bone_index[1]:bone_index[2]]
                self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]] - self.mpm_state.bone_x0cm.numpy()[0, 1]
                
                self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]].mean(axis=0)
                
                self.mpm_state.bone_R.numpy()[0, 1]
        
        # print("x max : ", self.mpm_state.particle_x.numpy().max())
        # print("x min : ", self.mpm_state.particle_x.numpy().min())
        # x, v에 적용하고 결과도 확인하기
        # g 가우시안 우는거 해결하기
         
        ######################################################################################################
        
        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        # print("total", (time.time() - start_time1)*1000, "ms") 
        time_total = time.time() - time_total
        
        self.time = self.time + dt # 10
        if step == self.hun_time :
            self.hun_time += 1

        # if human_step == 3:
        # if step == 100 and frame > 0:
        print_time = False
        if print_time:
            if step == 0:
                self.check_time = {'time1': 0.0, 'time2': 0.0, 'time3': 0.0, 'time4': 0.0, 'time5': 0.0,
                        'time6': 0.0, 'time7': 0.0, 'time8': 0.0, 'time9': 0.0, 'time_total': 0.0,
                        'time111': 0.0, 'time10': 0.0, 'step':0 }
                
            self.check_time['time1']      += time1*1000
            self.check_time['time2']      += time2*1000
            self.check_time['time3']      += time3*1000
            self.check_time['time4']      += time4*1000
            self.check_time['time5']      += time5*1000
            self.check_time['time6']      += time6*1000
            self.check_time['time7']      += time7*1000
            self.check_time['time8']      += time8*1000
            self.check_time['time9']      += time9*1000
            self.check_time['time10']     += time10*1000
            self.check_time['time111']    += time111*1000
            self.check_time['time_total'] += time_total*1000
            self.check_time['step']       += 1
            
            if step == 99 and frame > 0:
            # if 1:            
                print("")
                print("time1 : ", self.check_time['time1'] / self.check_time['step'], "ms") # 3.5ms
                print("time2 : ", self.check_time['time2'] / self.check_time['step'], "ms")
                print("time3 : ", self.check_time['time3'] / self.check_time['step'], "ms")
                print("time4 : ", self.check_time['time4'] / self.check_time['step'], "ms")
                print("time5 : ", self.check_time['time5'] / self.check_time['step'], "ms") # 1.8ms
                print("time6 : ", self.check_time['time6'] / self.check_time['step'], "ms") # 0.7~1ms            
                print("time7 : ", self.check_time['time7'] / self.check_time['step'], "ms")
                print("time8 : ", self.check_time['time8'] / self.check_time['step'], "ms") # 1.8ms
                print("time9 : ", self.check_time['time9'] / self.check_time['step'], "ms") # 0.5ms
                print("time10 : ", self.check_time['time10'] / self.check_time['step'], "ms") # 0.5ms
                print("time111 : ", self.check_time['time111'] / self.check_time['step'], "ms") # avatar velocity
                # print("time10 : ", time10*1000, "ms") # 0.7ms
                # print("time11 : ", time11*1000, "ms") # 1.17ms
                # print("time12 : ", time12*1000, "ms") # 0.17ms
                print("time total : ", self.check_time['time_total'] / self.check_time['step'], "ms") # 8~9ms
                print()

    
    # hun : this !!
    def p2g2p_base(self, frame, step, dt, mpm_params, device="cuda:0", smplx_dt=4e-2, is_3d_measure=False):
        
        # wp.config.mode = "debug"
        # wp.config.verify_cuda = True
        # wp.config.verify_fp = True
        time_total = time.time()
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            # self.mpm_model.n_humans + 1, # 0 for global, others for humans
        )
        grid_size_nsub = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
            self.mpm_model.n_humans, # 0 for global, others for humans
        )
        
        time1 = time.time()
        wp.launch(
            kernel=zero_grid_base,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        ) # 1
        time1 = time.time() - time1
        
        # Human Velocity
        time111 = time.time()
        human_step = int(self.time / smplx_dt)
        if step == 0:
        # if self.human_step != human_step : # step이랑 smpl step이랑 다름
        # if 0 : # step이랑 smpl step이랑 다름
            # self.mpm_state.particle_v.numpy()[:5]
            # self.mpm_state.particle_vk.numpy()[:5]
            # self.mpm_state.particle_vko.numpy()[:5]
            self.human_step = human_step
            maintain_avatar_shape = True
            
            if human_step == 0:
                lbs_temp = torch.zeros([self.mpm_state.particle_x.shape[0], 55], device=device)
            for k in range(len(self.human_modify_changer)): 
                with wp.ScopedTimer("to_torch_particle_x", synchronize=True, print=False, dict=self.time_profile):
                    particle_x = warp.to_torch(self.mpm_state.particle_x) # 0.07~0.12ms
                if self.human_modify_model[k].model_type == 'animatable_gaussians':
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM = compute_animatable_gaussians_velocity_rel(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure)
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM = compute_animatable_gaussians_velocity_tgt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure) # 0.11s                    
                elif self.human_modify_model[k].model_type == 'smplx':
                    if self.human_modify_model[k].velocity_type == 'rel':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_rel(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure)
                    elif self.human_modify_model[k].velocity_type == 'tgt':
                        velocity, rel_rot_mats, lbs, vSM = compute_smplx_velocity_tgt(self, particle_x, human_step, k, smplx_dt, frame, is_3d_measure=is_3d_measure)
                else:
                    assert False, "No velocity function"
                torch.cuda.synchronize()  # CUDA 연산을 동기화하여 이전 오류 감지
                torch.cuda.empty_cache()
                
                if 0:
                    avt_off1 = self.mpm_state.avatar_offset.numpy()[k]
                    avt_off2 = self.mpm_state.avatar_offset.numpy()[k+1]
                    lbs_temp[avt_off1:avt_off2] = lbs
                                                                    
                with wp.ScopedTimer("from_torch", synchronize=True, print=False, dict=self.time_profile):
                    new_human_velocity = wp.array(velocity.detach().cpu().numpy(), dtype=wp.vec3) # 4ms, 오래 걸리긴한데 일단 돌리자...
                    rel_rot_mats = wp.array(rel_rot_mats.detach().cpu().numpy(), dtype=wp.mat33)
                    vSM  = wp.array(vSM.detach().cpu().numpy(), dtype=wp.vec3)
                    
                wp.synchronize()
                wp.launch( # add velocity
                    kernel=self.human_modify_changer[k],
                    dim=self.human_modify_model[k].human_n_particles,
                    inputs=[self.mpm_state, self.human_modify_params[k], new_human_velocity, rel_rot_mats, 0, vSM], # vSM은 뼈에는 적용하면 안된다
                    device=device,
                )
                wp.synchronize()
                # print("compute_human_particle_velocity", (time.time() - start_time2)*1000, "ms") 
            if human_step == 0:
                with wp.ScopedTimer("LBS", synchronize=True, print=False, dict=self.time_profile):
                    self.mpm_state.particle_LBS = wp.array(lbs_temp.detach().cpu().numpy(), dtype=wp.float32, ndim=2)
        time111 = time.time() - time111
               
        # apply pre-p2g operations on particles
        # None for pillow2sofa example
        time2 = time.time()
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            ) # 2
        time2 = time.time() - time2
        
        # 3
        # apply dirichlet particle v modifier
        # 특정 particles에 init으로 mask 적용을 하고, 
        # 지정 시간동안 state.paricle_v=particle_velocity_modifier_params.velocity 설정
        # pillow2sofa example은 1개
        time3 = time.time()
        for k in range(len(self.particle_velocity_modifiers)):
            # start_time2 = time.time()
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        time3 = time.time() - time3

        if 0: # step == 0:
            print("\n")
            print(self.mpm_state.particle_F_trial.numpy().max(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().min(axis=0))
            print(self.mpm_state.particle_F.numpy().max(axis=0))
            print(self.mpm_state.particle_F.numpy().min(axis=0))
            print(self.mpm_state.particle_stress.numpy().max(axis=0))
            print(self.mpm_state.particle_stress.numpy().min(axis=0))
            print("model material : ", self.mpm_model.material)

        # 4, # compute stress = stress(returnMap(F_trial))        
        time4 = time.time()
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt], # state, model in function
                device=device,
            )  # F and stress are updated
        time4 = time.time() - time4
        
        if 0:
            print("\n")
            print(self.mpm_state.particle_x.numpy().max(axis=0), self.mpm_state.particle_x.numpy().min(axis=0))
            print(self.mpm_state.particle_v.numpy().max(axis=0), self.mpm_state.particle_v.numpy().min(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().max(axis=0))
            print(self.mpm_state.particle_F_trial.numpy().min(axis=0))
            print(self.mpm_state.particle_F.numpy().max(axis=0))
            print(self.mpm_state.particle_F.numpy().min(axis=0))
            print(self.mpm_state.particle_C.numpy().max(axis=0))
            print(self.mpm_state.particle_C.numpy().min(axis=0))
            print(self.mpm_state.particle_stress.numpy().max(axis=0))
            print(self.mpm_state.particle_stress.numpy().min(axis=0))
            print("\n")

        # 5, p2g
        time5 = time.time()
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # 5, # apply p2g
            wp.synchronize()
        time5 = time.time() - time5
        # self.mpm_state.grid_v_out.numpy()        
        # self.mpm_state.grid_v_in.numpy().max()
        # self.mpm_state.grid_vk.numpy().max()
        
        # 6, grid update
        time6 = time.time()
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity_base,
                # dim=(grid_size),
                dim=(grid_size_nsub),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
        time6 = time.time() - time6
        # self.mpm_state.grid_v_out.numpy()
        
        # threshold = self.mpm_model.dx, 0.01        
        # self.mpm_state.particle_x.shape = 957730
        # self.mpm_state.grid_v_particle_num.numpy().sum() # 957730
        # self.mpm_state.grid_v_particle_num.numpy().max() # 5771
        
        # if step == 0 and frame > 0:
        if 0:
            a = self.mpm_state.grid_v_mean_pos.numpy() # [:, :, :, n_index, 3]
            b = self.mpm_state.grid_v_particle_num.numpy()
            b = np.expand_dims(self.mpm_state.grid_v_particle_num.numpy(), axis=4) # [:, :, :, n_index, 1]
            b[b == 0] = 1        
            cc = a/b
            for i in range(a.shape[3]):
                c = cc[:, :, :, i, :].reshape(-1, 3)
                c = c[~np.all(c == 0, axis=1)]
                c_mesh = trimesh.Trimesh(vertices=c)
                c_mesh.export(f"./test_data/grid/parts/grid_points_{frame-1:03d}_{i}.ply")          
            cc = cc.reshape(-1, 3)
            cc = cc[~np.all(cc == 0, axis=1)]
            cc_mesh = trimesh.Trimesh(vertices=cc)
            cc_mesh.export(f"./test_data/grid/grid_points_{frame-1:03d}.ply")   
        
        # 67, grid penalty (Multi-Material Contact), grid update front? back?
        
        # time67 = time.time()
        # with wp.ScopedTimer(
        #     "grid_penalty", synchronize=True, print=False, dict=self.time_profile
        # ):
        #     wp.launch(
        #         kernel=grid_penalty,
        #         dim=(grid_size),
        #         inputs=[self.mpm_state, self.mpm_model, dt],
        #         device=device,
        #     )
        # time67 = time.time() - time67
        # wp.synchronize()
        
        save_path1 = f"./test_data/grid/force_vector1_{frame-1:03d}.ply"
        save_path2 = f"./test_data/grid/force_vector2_{frame-1:03d}.ply"
        if 0:
        # if step == 0 and frame > 0:
            # self.mpm_state.grid_v_mean_pos.numpy()
            # self.mpm_state.grid_v_check.numpy()
            export_grid_vertor(self.mpm_state, save_path1, save_path2)
                
        # 7
        time7 = time.time()
        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )
        time7 = time.time() - time7
        # self.mpm_state.grid_v_out.numpy()
        
        # 8, apply BC on grid
        time8 = time.time()
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                # add bounding box
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])
        time8 = time.time() - time8
        
        # damping이 잘 되는지 테스트
        # step 0에서는 기존 속도와, 적용 속도가 같아야 한다
                      
        # 9, g2p
        time9 = time.time()
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p_base,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated
            wp.synchronize()
        time9 = time.time() - time9
        
        
        numpy_test = False
        if numpy_test:            
            def SM_numpy(self, avatar, bone, device="cuda:0"):                
                avt = self.mpm_state.avatar_offset.numpy()
                bone_pnum = self.mpm_state.bone_pnum.numpy()
                bone_index = self.mpm_model.bone_index.numpy()                
                x_np = self.mpm_state.particle_x.numpy()
                v_np = self.mpm_state.particle_v.numpy()
                F_np = self.mpm_state.particle_F_trial.numpy()
                Fk_np = self.mpm_state.particle_Fk.numpy()
                C_np = self.mpm_state.particle_C.numpy()        
                bone_cano = self.mpm_state.bone_x0.numpy() # [2, 74496, 3]
                bone_cano_c = self.mpm_state.bone_x0cm.numpy()

                srt = avt[avatar] + bone_index[bone]
                end = avt[avatar] + bone_index[bone+1]
                
                # R_, t_가 original world와 MPM world가 같을까
                # ori world  
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                particle_x = warp.to_torch(self.mpm_state.particle_x)
                particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
                bone_cano_ori = ((torch.tensor(bone_cano, device=self.device) - center)/scale + ori_mean) @ rot_mats
                particle_x_ori_np = particle_x_ori.detach().cpu().numpy()
                
                # bone_cano_ori_mesh0 = trimesh.Trimesh(vertices=bone_cano_ori[0].detach().cpu().numpy())
                # bone_cano_ori_mesh0.export(f"./log/ag_FeFk_SM1/bone_cano_ori_mesh0.ply")
                        
                # x_np = x_np.astype(np.float64)
                # bone_cano = bone_cano.astype(np.float64)
                
                # MPM world        
                R_glob, t_glob = self.kabsch(torch.tensor(bone_cano[avatar, srt: end], device=self.device), 
                                    torch.tensor(x_np[srt: end], device=self.device))
                R_loc, t_loc = self.kabsch(torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device), 
                                    torch.tensor(particle_x_ori_np[srt: end], device=self.device))
                
                torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_glob.T + t_glob - torch.tensor(x_np[srt: end], device=self.device)
                torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device) @ R_loc.T + t_loc - torch.tensor(particle_x_ori_np[srt: end], device=self.device)
                
                # numpy나 torch로 shape matching 결과를 확인해보자
                # 1
                x_i = x_np[srt: end]
                x_cm = x_np[srt: end].mean(axis=0)
                x_i0 = bone_cano[avatar, srt: end]
                x_cm0 = bone_cano[avatar, srt: end].mean(axis=0)

                v_i = v_np[srt: end]
                v_cm = v_np[srt: end].mean(axis=0)

                # 2.1
                r_i = x_i - x_cm                
                q_i = x_i0 - x_cm0
                Apq_parts = (r_i[:, :, None] * q_i[:, None, :])
                Apq = Apq_parts.sum(axis=0)
                
                # 2.2
                L = np.cross(r_i, v_i).sum(axis=0) # [3, 3]
                
                r_sq = np.einsum('ij,ij->i', r_i, r_i)   # (N,)  |r_i|²
                I3   = np.eye(3)
                term1 = r_sq[:, None, None] * I3          # (N,3,3)  |r_i|² I₃
                term2 = r_i[:, :, None] * r_i[:, None, :]     # (N,3,3)  r_i r_iᵀ
                I     = np.sum(term1 - term2, axis=0)     # (3,3)
                w     = np.linalg.solve(I, L)  # (3,)  I w = L
                
                (v_cm + np.cross(w, r_i)) # new v_i
                v_i
                np.abs((v_cm + np.cross(w, r_i)) - v_i).max() # CHECK
                
                # 3
                U, S, Vh = np.linalg.svd(Apq)
                UVt  = U @ Vh
                detUV = np.linalg.det(UVt)
                sign = -1.0 if detUV < 0.0 else 1.0
                Dfix = np.eye(3)
                Dfix[-1, -1] = sign
                R = U @ Dfix @ Vh
                
                (q_i @ R.T + x_cm) # new x_i
                x_i
                np.abs((q_i @ R.T + x_cm) - x_i).max() # CHECK
                        
                # Check !!
                # R_.detach().cpu().numpy() - R # 0
                
                # kabsch, local cano to local posed
                # torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_.T + t_ 
                # torch.tensor(x_np[srt: end], device=self.device)
                
                # SM
                # (q_i @ R.T + x_cm)
                # x_i = x_np[srt: end]
                # x_i0 = bone_cano[avatar, srt: end]
                # x_cm0 = bone_cano[avatar, srt: end].mean(axis=0)
                # q_i = x_i0 - x_cm0
                                             
                # return R, x_cm, R_, t_
                return R_glob, t_glob, R_loc, t_loc
            
            R_glob, t_glob, R_loc, t_loc = SM_numpy(self, 0, 0, device=device)
            # R0, x_cm0, R0_, t0_ = SM_numpy(self, 0, 0, device=device)
            # R1, x_cm1, R1_, t1_ = SM_numpy(self, 0, 1, device=device)

            bone_index = self.mpm_model.bone_index.numpy()        
            self.mpm_state.bone_q.numpy()[0, bone_index[1]:bone_index[2]]
            self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]] - self.mpm_state.bone_x0cm.numpy()[0, 1]
            
            x_i = self.mpm_state.particle_x.numpy()[4495:8949] # x_i
            x_cm = self.mpm_state.particle_x.numpy()[4495:8949].mean(axis=0) # x_cm
            x_i0 = self.mpm_state.bone_x0.numpy()[0, 4495:8949] # x_i0
            x_cm0 = self.mpm_state.bone_x0.numpy()[0, 4495:8949].mean(axis=0) # x_cm0
            r_i = x_i - x_cm # r_i
            q_i = x_i0 - x_cm0 # q_i
        
        shape_matching = False
        save_test = False
        
        if 1 and shape_matching and frame==0 and step==0 :
            @wp.kernel
            def add_noise(x: wp.array(dtype=wp.vec3),
                        noise_scale: float):

                i = wp.tid()
                seed = 12777
                rand_state = wp.rand_init(seed + i)
                rx = 2.0 * wp.randf(rand_state) - 1.0
                ry = 2.0 * wp.randf(rand_state) - 1.0
                rz = 2.0 * wp.randf(rand_state) - 1.0

                noise = wp.vec3(rx, ry, rz) * noise_scale
                x[i] += noise

            # 3) 실행
            wp.launch(
                kernel=add_noise,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_v, 0.01],
                device="cuda"
            )
        
        if shape_matching:
            # bone 변수들 zeros
            with wp.ScopedTimer(
                "zero_bone", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=zero_bone,
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
            
            time10 = time.time()
            with wp.ScopedTimer(
                "shape_matching_center", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching1_reduce,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            time10 = time.time() - time10
            
            time11 = time.time()
            with wp.ScopedTimer(
                "shape_matching_solve", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching2_solve,
                    dim=self.n_particles,
                    # dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            time11 = time.time() - time11
            
            time12 = time.time()
            with wp.ScopedTimer(
                "shape_matching_particle", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching3_Rw,
                    # dim=(self.n_particles),
                    dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            time12 = time.time() - time12
            
            if save_test:                
                bone_trimesh1 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[0:74496],
                )
                bone_trimesh1.export(f"./test_data/shape_matching_before1.ply")
                bone_trimesh2 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[420414:420414+74496],
                )
                bone_trimesh2.export(f"./test_data/shape_matching_before2.ply")
            
            local_global_A_test = False
            if local_global_A_test:     
                bone_R = self.mpm_state.bone_R.numpy()[0,0]
                bone_x0 = self.mpm_state.bone_x0.numpy()[0,0]
                bone_x0cm = self.mpm_state.bone_x0cm.numpy()[0,0]
                                
                # R_glob, t_glob, R_loc, t_loc      
                
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                                
                ## local에서의 A 만드는법 구현하기, 완료 !!!
                rot = rot_mats
                A = rot / scale
                b = (-center/scale + ori_mean) @ rot
                R_loc_test = rot.T @ R_glob @ rot
                t_loc_test = t_glob @ (rot/scale) + b - b @ R_loc_test.T
                
                
                ## 과연 local로 돌리는게 필요할까, global에서 A를 구해서 바로 LBS를 써도 되지 않을까?
                
                avt = self.mpm_state.avatar_offset.numpy()
                bone_index = self.mpm_model.bone_index.numpy()                
                x_np = self.mpm_state.particle_x.numpy()
                bone_cano = self.mpm_state.bone_x0.numpy() # [2, 74496, 3]

                avatar = 1
                bone = 1
                srt = avt[avatar] + bone_index[bone]
                end = avt[avatar] + bone_index[bone+1]
                                
                # ori world  
                ori_mean = self.human_modify_model[0].ori_mean
                rot_mats = self.human_modify_model[0].rot_mats
                scale    = self.human_modify_model[0].scale
                center   = self.human_modify_model[0].center
                particle_x = warp.to_torch(self.mpm_state.particle_x)
                particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats
                bone_cano_ori = ((torch.tensor(bone_cano, device=self.device) - center)/scale + ori_mean) @ rot_mats
                particle_x_ori_np = particle_x_ori.detach().cpu().numpy()
                
                torch.set_printoptions(precision=8)
                R_glob, t_glob = self.kabsch(torch.tensor(bone_cano[avatar, bone_index[bone]: bone_index[bone+1]], device=self.device), 
                                    torch.tensor(x_np[srt: end], device=self.device))
                R_loc, t_loc = self.kabsch(torch.tensor(bone_cano_ori[avatar, bone_index[bone]: bone_index[bone+1]], device=self.device), 
                                    torch.tensor(particle_x_ori_np[srt: end], device=self.device))
                
                # torch.tensor(bone_cano[avatar, srt: end], device=self.device) @ R_glob.T + t_glob - torch.tensor(x_np[srt: end], device=self.device)
                # torch.tensor(bone_cano_ori[avatar, srt: end], device=self.device) @ R_loc.T + t_loc - torch.tensor(particle_x_ori_np[srt: end], device=self.device)
                                
                M  = self.mpm_state.bone_m.numpy()[avatar, bone] # bone mass
                x_cm = self.mpm_state.bone_mx.numpy()[avatar, bone] * (1.0 / M)  # now bone center
                bone_x0cm = self.mpm_state.bone_x0cm.numpy()[avatar, bone]
                R_glob_SM = self.mpm_state.bone_R.numpy()[avatar, bone] # R_glob
                t_ = x_cm - R_glob_SM @ bone_x0cm # t_glob
                
                self.mpm_state.particle_LBS # 90% 정도가 0.001 미만
                
                lbs = wp.to_torch(self.mpm_state.particle_LBS)
                thresh = 1e-3
                N, K = lbs.shape
                per_row_counts = (lbs >= thresh).sum(dim=1)  # (N,)
                hist = torch.bincount(per_row_counts, minlength=K+1)  # (K+1,)
                
                ####
                # lbs는 avatar load할때 불러오고
                # cano avatar (transformation 적용하고)는 매 frame 마다
                # 1. SM bone to avatar A-matrix
                avt_off = self.mpm_state.avatar_offset.numpy()
                lbs[avt_off[0]:avt_off[1]]
                
            
            time13 = time.time()
            with wp.ScopedTimer(
                "shape_matching_bone_particle", synchronize=True, print=False, dict=self.time_profile
            ):
                wp.launch(
                    kernel=shape_matching4_bone_particle, # bone & avatar
                    dim=(self.n_particles),
                    # dim=(self.mpm_model.n_humans, 20),
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
                wp.synchronize()
            time13 = time.time() - time13            
            
            # SM test, well done !!
            # if step == 0 :
            #     bone_s = self.human_modify_model[k].particle_start
            #     bone_e = self.human_modify_model[k].particle_start + self.human_modify_model[k].bone_index[-1]
            #     bone_e_smpl = self.human_modify_model[k].particle_start + self.human_modify_model[k].human_n_particles
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_x.numpy()[bone_s:bone_e])
            #     temp_mesh.export('./test_results/bone_noise_{:04d}.ply'.format(frame))
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_SM_test.numpy()[bone_s:bone_e])
            #     temp_mesh.export('./test_results/bone_SM_{:04d}.ply'.format(frame))                
            #     temp_mesh = trimesh.Trimesh(vertices = self.mpm_state.particle_x.numpy()[bone_s:bone_e_smpl])
            #     temp_mesh.export('./test_results/bone_noise_full_{:04d}.ply'.format(frame))
            
            # time14 = time.time()
            # with wp.ScopedTimer(
            #     "shape_matching_avatar_particle", synchronize=True, print=False, dict=self.time_profile
            # ):
            #     wp.launch(
            #         kernel=shape_matching5_avatar_particle, # bone & avatar
            #         dim=(self.n_particles),
            #         # dim=(self.mpm_model.n_humans, 20),
            #         inputs=[self.mpm_state, self.mpm_model, dt],
            #         device=device,
            #     )
            #     wp.synchronize()
            # time14 = time.time() - time14            
                        
            if save_test:                
                bone_trimesh1 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[0:74496],
                )
                bone_trimesh1.export(f"./test_data/shape_matching_after1.ply")
                bone_trimesh2 = trimesh.Trimesh(
                    vertices=self.mpm_state.particle_x.numpy()[420414:420414+74496],
                )
                bone_trimesh2.export(f"./test_data/shape_matching_after2.ply")
                
                bone_index = self.mpm_model.bone_index.numpy() 
                bone_cano = self.mpm_state.bone_x0.numpy()
                self.mpm_state.bone_q.numpy()[0, bone_index[0]:bone_index[1]]
                self.mpm_state.bone_x0.numpy()[0, bone_index[0]:bone_index[1]] - self.mpm_state.bone_x0cm.numpy()[0, 0]
                
                self.mpm_state.bone_q.numpy()[0, bone_index[1]:bone_index[2]]
                self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]] - self.mpm_state.bone_x0cm.numpy()[0, 1]
                
                self.mpm_state.bone_x0.numpy()[0, bone_index[1]:bone_index[2]].mean(axis=0)
                
                self.mpm_state.bone_R.numpy()[0, 1]
        
        # print("x max : ", self.mpm_state.particle_x.numpy().max())
        # print("x min : ", self.mpm_state.particle_x.numpy().min())
        # x, v에 적용하고 결과도 확인하기
        # g 가우시안 우는거 해결하기
         
        ######################################################################################################
        
        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        # print("total", (time.time() - start_time1)*1000, "ms") 
        time_total = time.time() - time_total
        
        self.time = self.time + dt # 10
        if step == self.hun_time :
            self.hun_time += 1

        # if human_step == 3:
        if step == 0:
            self.check_time = {'time1': 0.0, 'time2': 0.0, 'time3': 0.0, 'time4': 0.0, 'time5': 0.0,
                      'time6': 0.0, 'time7': 0.0, 'time8': 0.0, 'time9': 0.0, 'time_total': 0.0,
                      'time111': 0.0, 'time10': 0.0, 'step':0 }
            
        self.check_time['time1']      += time1*1000
        self.check_time['time2']      += time2*1000
        self.check_time['time3']      += time3*1000
        self.check_time['time4']      += time4*1000
        self.check_time['time5']      += time5*1000
        self.check_time['time6']      += time6*1000
        self.check_time['time7']      += time7*1000
        self.check_time['time8']      += time8*1000
        self.check_time['time9']      += time9*1000
        # self.check_time['time10']     += time10*1000
        self.check_time['time111']    += time111*1000
        self.check_time['time_total'] += time_total*1000
        self.check_time['step']       += 1
        
        if step == 99 and frame > 0:
        # if 1:            
            print("")
            print("time1 : ", self.check_time['time1'] / self.check_time['step'], "ms") # 3.5ms
            print("time2 : ", self.check_time['time2'] / self.check_time['step'], "ms")
            print("time3 : ", self.check_time['time3'] / self.check_time['step'], "ms")
            print("time4 : ", self.check_time['time4'] / self.check_time['step'], "ms")
            print("time5 : ", self.check_time['time5'] / self.check_time['step'], "ms") # 1.8ms
            print("time6 : ", self.check_time['time6'] / self.check_time['step'], "ms") # 0.7~1ms            
            print("time7 : ", self.check_time['time7'] / self.check_time['step'], "ms")
            print("time8 : ", self.check_time['time8'] / self.check_time['step'], "ms") # 1.8ms
            print("time9 : ", self.check_time['time9'] / self.check_time['step'], "ms") # 0.5ms            
            print("time111 : ", self.check_time['time111'] / self.check_time['step'], "ms") # avatar velocity
            # print("time10 : ", time10*1000, "ms") # 0.7ms
            # print("time11 : ", time11*1000, "ms") # 1.17ms
            # print("time12 : ", time12*1000, "ms") # 0.17ms
            print("time total : ", self.check_time['time_total'] / self.check_time['step'], "ms") # 8~9ms
            print()


    # set particle densities to all_particle_densities,
    def reset_densities_and_update_masses(
        self, all_particle_densities, device="cuda:0"
    ):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(
            all_particle_densities, dvc=device
        )
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)
            
    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)

    def export_particle_F_trial_to_torch(self):
        Ftr_tensor = wp.to_torch(self.mpm_state.particle_F_trial)
        Ftr_tensor = Ftr_tensor.reshape(-1, 9)
        return Ftr_tensor
    
    def export_particle_F_before_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F_before)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor
    
    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor
    
    def export_particle_Fe_to_torch(self):
        Fe_tensor = wp.to_torch(self.mpm_state.particle_Fe)
        Fe_tensor = Fe_tensor.reshape(-1, 9)
        return Fe_tensor
    
    def export_particle_F_add_to_torch(self):
        Fadd_tensor = wp.to_torch(self.mpm_state.particle_F_add)
        Fadd_tensor = Fadd_tensor.reshape(-1, 9)
        return Fadd_tensor

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )
        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov
    
    def export_particle_scale_rot_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_scale_rot_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_scale_rot_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        scale = wp.to_torch(self.mpm_state.particle_scale)
        rot = wp.to_torch(self.mpm_state.particle_R)
        return scale, rot
    
    def export_particle_quat_scale_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_quat_scale_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_quat_scale_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        quat = wp.to_torch(self.mpm_state.particle_quat)
        scale = wp.to_torch(self.mpm_state.particle_scale)
        rot = wp.to_torch(self.mpm_state.particle_R)
        return quat, scale, rot


    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # hun : remove temporary, 병렬화
    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        return 0
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            # grid_x, grid_y, grid_z, n_humans = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)
    
    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(
        self,
        force,
        dt,
        point=[1, 1, 1],
        size=[1, 1, 1],
        num_dt=1,
        start_time=0.0,
        device="cuda:0",
    ):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[self.mpm_state, impulse_param],
            device=device,
        )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)

    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(
        self,
        point,
        normal,
        half_height_and_radius,
        rotation_scale,
        translation_scale,
        start_time,
        end_time,
        device="cuda:0",
    ):

        normal_scale = 1.0 / wp.sqrt(
            float(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        )
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(
            half_height_and_radius[0], half_height_and_radius[1]
        )
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0, 1.0, 1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = (
            horizontal_1
            - wp.dot(horizontal_1, velocity_modifier_params.normal)
            * velocity_modifier_params.normal
        )
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_cylinder,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        # class ParticleVelocityModifier:
        #     point: wp.vec3
        #     normal: wp.vec3
        #     half_height_and_radius: wp.vec2
        #     rotation_scale: float
        #     translation_scale: float
        #     size: wp.vec3
        #     horizontal_axis_1: wp.vec3
        #     horizontal_axis_2: wp.vec3
        #     start_time: float
        #     end_time: float
        #     velocity: wp.vec3
        #     mask: wp.array(dtype=int)
        
        # particle_v: wp.array(dtype=wp.vec3)
    
        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(
                        offset
                        - wp.dot(offset, velocity_modifier_params.normal)
                        * velocity_modifier_params.normal
                    )
                    cosine = (
                        wp.dot(offset, velocity_modifier_params.horizontal_axis_1)
                        / horizontal_distance
                    )
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = (
                        -horizontal_distance
                        * wp.sin(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis2_scale = (
                        horizontal_distance
                        * wp.cos(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = (
                        axis1_scale * velocity_modifier_params.horizontal_axis_1
                        + axis2_scale * velocity_modifier_params.horizontal_axis_2
                        + axis_vertical_scale * velocity_modifier_params.normal
                    )

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(
        self, normal, start_position, end_position, num_layers, start_time, end_time
    ):
        num_layers = 50
        point = [0, 0, 0]
        size = [0, 0, 0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position

        half_length_portion = wp.abs(start_position - end_position) / num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(
                point=point,
                size=size,
                velocity=[0, 0, 0],
                start_time=start_time,
                end_time=end_time_portion * (i + 1),
            )

    # 병렬화
    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            # grid_x, grid_y, grid_z = wp.tid()
            grid_x, grid_y, grid_z, grid_n = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:
                # if outside of padding is going outside, set grid velocity=0
                # if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0] < 0:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    # state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1],
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding # grid_dim_x = n_grid
                    # and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0] > 0
                    and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
                ):
                    # state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1],
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                # if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1] < 0:
                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    # state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        # state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    # and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1] > 0
                    and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
                ):
                    # state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                    #     state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0],
                    #     0.0,
                    #     state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2],
                    # )
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                # if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2] < 0:
                #     state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                #         state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0],
                #         state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1],
                #         0.0,
                #     )
                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                # if (
                #     grid_z >= model.grid_dim_z - padding
                #     and state.grid_v_out[grid_x, grid_y, grid_z, grid_n][2] > 0
                # ):
                #     state.grid_v_out[grid_x, grid_y, grid_z, grid_n] = wp.vec3(
                #         state.grid_v_out[grid_x, grid_y, grid_z, grid_n][0],
                #         state.grid_v_out[grid_x, grid_y, grid_z, grid_n][1],
                #         0.0,
                #     )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide) # 8
        self.modify_bc.append(None) # 8

    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):    
        return 0
        # a cubiod is a rectangular cube
        # centered at `point`
        # dimension is x: point[0]±size[0]
        #              y: point[1]±size[1]
        #              z: point[2]±size[2]
        # all grid nodes lie within the cubiod will have their speed set to velocity
        # the cuboid itself is also moving with const speed = velocity
        # set the speed to zero to fix BC    
        
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        # 병렬화
        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            # grid_x, grid_y, grid_z, n_humans = wp.tid()
            # grid_postprocess, grid 작업 후 BC에서 설정한 velocity 적용
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0], # x-axis length
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            # 만약 reset=1면, end_time 이전은 velocity 적용, 이후는 잠깐 0으로 고정
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        # 위쪽 collide 함수에서 offset을 계산하기 위해 param.point 업데이트
        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide) # 8
        self.modify_bc.append(modify) # 8

    def enforce_particle_velocity_translation(
        self, point, size, velocity, start_time, end_time, index, device="cuda:0"
    ):
        # first select certain particles based on position
        # self.mpm_state.particle_x.numpy().min() # 0.45877
        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])
        velocity_modifier_params.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time
        velocity_modifier_params.index = index
        velocity_modifier_params.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)
        
        wp.launch(
            kernel=selection_enforce_particle_velocity_translation, # mask
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params) # 3

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if state.particle_id[p] == velocity_modifier_params.index:
                    if velocity_modifier_params.mask[p] == 1:
                        state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g) # 3    

    '''
    # FROM : main/set_boundary_conditions
    def modify_posed_human(self, avatar_net, pose_dataset, cano_pts, cano_rot, cano_J, joint_mat, A_mat, knn_indices,
        bone_cano, bone_index, bone_faces, particle_start, index, rot_mats, ori_mean, scale, center, device="cuda"
    ):
        # 여기부터 작성
        # input은 pose -> main_avatar_phys.py/get_avatars 참고해서 smpl로부터 값을 유추하기
        
        # 1. cano_xyz, cano_rot, lbs, cano_num_particles
        human_model = HumanTorchModel()
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
        
        self.human_modify_model.append(human_model) # torch velocity
        
        # 2.
        human_params = HumanModifier()        
        human_params.particle_id = wp.array(np.where(self.mpm_state.particle_id.numpy() == index)[0], dtype=int) # [Human Particles], [ps+0, ps+1, ps+2, ..., ps+N_avatar]       
        self.human_modify_params.append(human_params) # warp velocity

        # 3, velocity 덮어씌우기
        @wp.kernel
        def kinematic_velocity(
            state: MPMStateStruct,
            human_params: HumanModifier,
            kinematic_v: wp.array(dtype=wp.vec3), 
            relR: wp.array(dtype=wp.mat33),
            apply_rot: int,
            vSM: wp.array(dtype=wp.vec3),
        ): # dim = human_n_particles (avatar + bone)
            p = wp.tid()
            id = human_params.particle_id[p]
            state.particle_v[id]   = state.particle_v[id] + kinematic_v[p] - state.particle_vk[id]
            state.particle_vk[id]  = kinematic_v[p]
            state.particle_vko[id] = kinematic_v[p]
            state.particle_F_trial[id] = relR[p] * state.particle_F_trial[id]
            # state.particle_F[id] = relR[p] * state.particle_F[id]
            bid = state.bone_idx[p]
            if bid < 0: # if not bone(if avatar)
                state.particle_vSM[id] = vSM[p]
            
        self.human_modify_changer.append(kinematic_velocity)
                
        # 4. apply particle bone index
        # bone + avatar
        # self.mpm_state.bone_idx # [N]
        ps = particle_start
        # particle_bone_idx = ps + np.array(range(bone_index[-1])) # [74496]
        particle_bone_val = np.zeros(bone_index[-1], dtype=np.int16) # [74496]
        # particle_bone_val = np.zeros_like(particle_bone_idx)
        for i in range(len(bone_index)-1):
            particle_bone_val[ bone_index[i] : bone_index[i+1] ] = i
        
        state_bone_idx = self.mpm_state.bone_idx.numpy()
        state_bone_idx[ps:ps+bone_index[-1]] = particle_bone_val
        self.mpm_state.bone_idx = wp.array(state_bone_idx, dtype=wp.int16, device=device)        
        # self.mpm_state.bone_idx.numpy()[ps:ps+74496]
        
        # @wp.kernel
        # def particle_bone_index(state: MPMStateStruct, idx: wp.array(dtype=wp.int16), val: wp.array(dtype=wp.int16)):
        #     p = wp.tid()
        #     i = idx[p]
        #     state.bone_idx[i] = val[p]
        # wp.launch(
        #     kernel=particle_bone_index,
        #     dim=bone_index[-1], # 74496
        #     inputs=[self.mpm_state, particle_bone_idx_warp, particle_bone_val_warp],
        #     device=device,
        # )
        
        # 위 코드 수정해서 state.bone_idx에 값 제대로 들어가게 수정해야함
        # np.argwhere(self.mpm_state.bone_idx.numpy() == 0)
        # np.argwhere(self.mpm_state.bone_idx.numpy() == 1)
        # np.argwhere(self.mpm_state.bone_idx.numpy() == 2)
        # np.argwhere(self.mpm_state.bone_idx.numpy()[:74495] == 2)
        
        # 5. save bone cano
        # bone_cano_torch = bone_cano.clone().detach() # [74496, 3]
        # bone_cano_torch = torch.tensor(bone_cano, device=device, dtype=torch.float32) # [74496, 3]
        bone_cano_torch = (torch.mm(bone_cano, rot_mats.T) - ori_mean) * scale + center # GT2Sim coordinate system
        bone_cano_wp    = torch2warp_vec3(bone_cano_torch, dvc=device)
        # bone_cano_wp    = wp.from_torch(bone_cano_torch.to(torch.double).detach(), dtype=wp.vec3d)
        
        bone_mass_torch = wp.to_torch(self.mpm_state.particle_mass)[ps:ps+bone_index[-1]]
        x_splits = torch.split(bone_cano_torch, self.bone_p_num.tolist(), dim=0)
        m_splits = torch.split(bone_mass_torch, self.bone_p_num.tolist(), dim=0)
        # bone_cano_c_torch = torch.stack(
        #     [ (x * m.unsqueeze(-1)).sum(dim=0) / m.sum() for x, m in zip(x_splits, m_splits) ], dim=0
        # )
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
            inputs=[self.mpm_state.bone_x0, index, bone_cano_wp],
            device=device,
        )   
        wp.launch(
            kernel=particle_bone_cano,
            dim=20, # 20
            inputs=[self.mpm_state.bone_x0cm, index, bone_cano_c_wp],
            device=device,
        )
        wp.launch(
            kernel=particle_bone_cano,
            dim=bone_index[-1], # 20
            inputs=[self.mpm_state.bone_q, index, bone_cano_q_wp],
            device=device,
        )
        # self.mpm_state.bone_cano.numpy()
        # self.mpm_state.bone_cano_c.numpy()[1]
        # self.mpm_state.bone_cano_p.numpy()[1]
                
        @wp.kernel
        def set_bone_E_nu(
            state: MPMStateStruct, 
            model: MPMModelStruct
        ):
            p = wp.tid()
            bone_idx = state.bone_idx[p]
            if bone_idx >= 0:
                E = 2e6
                nu = 0.3
                model.E[p] = E
                model.nu[p] = nu
                model.mu[p] = E / (2.0 * (1.0 + nu))
                model.lam[p] = (
                    E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                )
        
        wp.launch(
            kernel=set_bone_E_nu,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        
        # 6. avatar_offset
        offset_np = self.mpm_state.avatar_offset.numpy()
        last_offset = offset_np[-1] + human_n_particles
        new_offset_np = np.concatenate([offset_np, [last_offset]]).astype(np.int32)
        self.mpm_state.avatar_offset = wp.array(new_offset_np, dtype=wp.int32, device=device)
    
    '''
    
    '''
    ######################################################################################################################################################################################
    # Inside enforce_joint_velocity & p2g2p per frame
    
    @torch.no_grad()
    def compute_human_particle_velocity(self, particle_x, human_step, list_idx, smplx_dt, maintain_avatar_shape=False):
        # list_idx -> human_idx
        
        # 1. 뼈대 움직임에서 keypoint R, t 계산 -> 현재 global A-pose matrix 계산
        # 1.1 이때 vector 2개로 R 구하는 방법이 필요할 수도 있다. (contribution 2)
        # torch
        avatar_net = self.human_modify_model[list_idx].avatar_net
        # extr = self.human_modify_model[list_idx].extr # [55, 4, 4]
        pose_dataset = self.human_modify_model[list_idx].pose_dataset
        human_n_particles = self.human_modify_model[list_idx].human_n_particles
        
        bone_cano = self.human_modify_model[list_idx].bone_cano # [74496, 3]
        bone_index = self.human_modify_model[list_idx].bone_index # [0, 4495, 8949, ...]
        bone2smplx = self.human_modify_model[list_idx].bone2smplx # [0, 3, 6, 9, 12, ...]
        cano_J = self.human_modify_model[list_idx].cano_J # [55, 3]
        knn_indices = self.human_modify_model[list_idx].knn_indices
        ps = self.human_modify_model[list_idx].particle_start
        
        ori_mean = self.human_modify_model[list_idx].ori_mean
        rot_mats = self.human_modify_model[list_idx].rot_mats
        scale    = self.human_modify_model[list_idx].scale
        center   = self.human_modify_model[list_idx].center
        
        A_now = torch.eye(4, device=self.device).unsqueeze(0).repeat(22, 1, 1)
        # A_mat = self.human_modify_model[list_idx].A_mat[:22]
        
        # cano_J[:, :3] = (torch.mm(cano_J[:, :3], rot_mats.T) - ori_mean) * scale + center
        # bone_cano = (torch.mm(bone_cano, rot_mats.T) - ori_mean) * scale + center
        # avatar particle : particle_x_ori[ps+bone_index[-1]:ps+human_n_particles]
                
        particle_x_ori = ((particle_x - center)/scale + ori_mean) @ rot_mats        
        
        # temp test
        kabsch_A = torch.zeros_like(A_now)
        kabsch_A[:, 3, 3] = 1.0
        
        for i in range(len(bone_index)-1):
            # time1 = time.time()
            R_est, t_est = self.kabsch(bone_cano[bone_index[i]:bone_index[i+1]], particle_x_ori[ps+bone_index[i]:ps+bone_index[i+1]]) # cano, pose
            
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
        
        # kabsch_A <-> A_now[0] relation test !!
        # 완전히 똑같다, kabsch 결과를 A로 대체해도 된다
        # (A_now[0] - kabsch_A).abs().max()
        
        # 2. 현재 global A-pose matrix 에서 다음 global A-pose matrix 계산
        # torch
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
            
            # (now_smpl.A[0, :22] - kabsch_A).abs().max()
            
            parents = pose_dataset.smpl_model.parents[:22]
            joints = torch.unsqueeze(now_smpl.J[:, :22], dim=-1) # Same as cano_J
            joints_homogen = F.pad(joints, [0, 0, 0, 1])
            rel_joints = joints.clone()
            rel_joints[:, 1:] -= joints[:, parents[1:]]        
            next_transl = pose_dataset.transl[next_frame]
            
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
            rel_transforms[0, :, :3, 3] += next_transl # [1, 22, 4, 4]
            A_next = rel_transforms
            # (A_next[0] - next_smpl.A[0, :22]).abs().max() # check !!, yes !!
            
            # 3. AG model network, delta_position
            # torch
            
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
            cano_xyz_now = self.human_modify_model[list_idx].cano_xyz
            cano_rot_now = self.human_modify_model[list_idx].cano_rot
            joint_mat_now = torch.matmul(A_now_55[0], inv_cano_jnt_mats) # [55, 4, 4]
            # joint_mat_now = self.human_modify_model[list_idx].joint_mat # [55, 4, 4], torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
            # A_now # = self.human_modify_model[list_idx].A_mat # [55, 4, 4], 
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
            positions_next = torch.einsum('nxy,ny->nx', pt_mats_next[..., :3, :3], cano_xyz_next) + pt_mats_next[..., :3, 3]
            rot_mats_next = torch.einsum('nxy,nyz->nxz', pt_mats_next[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(cano_rot_next)) # [human_N, 3, 3]
            
            # print(joint_mat_next[0])
            # print(A_next_55[0, 0])
            self.human_modify_model[list_idx].pt_mats_next = pt_mats_next
            self.human_modify_model[list_idx].cano_xyz = cano_xyz_next
            self.human_modify_model[list_idx].cano_rot = cano_rot_next
            # self.human_modify_model[list_idx].joint_mat = joint_mat_next
            # self.human_modify_model[list_idx].A_mat = A_next
                        
            # 4.2 Bone Velocity
            # bone_cano  = self.human_modify_model[list_idx].bone_cano
            # bone_index = self.human_modify_model[list_idx].bone_index
            # smpl_index = self.human_modify_model[list_idx].bone2smplx
            bone_verts_num = bone_cano.shape[0]
            bone_pose1 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
            bone_pose2 = torch.zeros(bone_verts_num, 3, device=bone_cano.device)
            bone_rot1 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
            bone_rot2 = torch.eye(3, device=bone_cano.device).unsqueeze(0).repeat(bone_verts_num, 1, 1)
        
            # data_dir = './AnimatableGaussians/datasets/Actor01/Sequence1'
            # data_dir = './AnimatableGaussians/datasets/Actor07/Sequence1'
            # bone_path = os.path.join(data_dir, 'osso', 'osso_per_parts', 'part_split_meshes.glb')
            # bone = trimesh.load(bone_path)
            # bone_faces = self.human_modify_model[0].bone_faces
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
            alpha = 0.0
            positions_now_total_sim  = particle_x_ori[ps:ps+human_n_particles] # xyz of real MPM simulation
            positions_now_total_pos  = torch.cat([bone_pose1, positions_now]) # xyz of posed gt avatar
            positions_next_total = torch.cat([bone_pose2, positions_next])
            
            rot_mats_now_total  = torch.cat([bone_rot1, rot_mats_now])
            rot_mats_next_total = torch.cat([bone_rot2, rot_mats_next])

            velocity = (positions_next_total - positions_now_total_pos * (1-alpha) - positions_now_total_sim * alpha) / smplx_dt # [373056, 3]
            # velocity = ( (positions_next_total - positions_now_total_pos) + (positions_now_total_pos - positions_now_total_sim) * alpha )/ smplx_dt # [373056, 3]
            
            relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next_total, torch.inverse(rot_mats_now_total))
            # relative_rot_mats = torch.einsum('nxy,nyz->nxz', rot_mats_next, torch.inverse(rot_mats_now))
            
            # velocity += (positions_next - particle_x[ps : ps + human_n_particles]) / smplx_dt * 0.001 # [373056, 3], 아바타, 뼈 유지
            # velocity += (positions_now - particle_x[ps : ps + human_n_particles]) / smplx_dt * 0.001 # [373056, 3], 아바타, 뼈 유지
            
            # 4.0 ply save
            # save_path = "/workspace/physics/PhysGaussian_org/test_data/particle/"
            # particle_x = trimesh.Trimesh(vertices=particle_x_ori[ps+bone_index[-1]:ps+human_n_particles].detach().cpu().numpy())
            # particle_x.export(save_path + str(human_step) + "_particlex.ply")
            # particle_x = trimesh.Trimesh(vertices=positions_now_total[bone_index[-1]:human_n_particles].detach().cpu().numpy())
            # particle_x.export(save_path + str(human_step) + "_positions_now.ply")
            # particle_x = trimesh.Trimesh(vertices=positions_next_total[bone_index[-1]:human_n_particles].detach().cpu().numpy())
            # particle_x.export(save_path + str(human_step) + "_positions_next.ply")
            
            # cano_xyz_now = self.human_modify_model[list_idx].cano_xyz
            # cano_rot_now = self.human_modify_model[list_idx].cano_rot
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
            # grad = self.compute_manual_gradients_parallel(points, knn_indices, L_target) # Edge Length Regularization
            # move_directions = -10*grad
            # velocity[bone_index[-1]:] += move_directions            
            ########################################
            
            ##########################################            
            # 4.6 Final, avatar shape maintain regulzation        
            particle_x_ori # [957730, 3]
            particle_x_ori[ps : ps+human_n_particles] # [447552, 3]
            positions_now_total_sim # [447552, 3]
            positions_now_total_pos # [447552, 3]
            reg_velocity = (positions_now_total_pos - positions_now_total_sim) / smplx_dt 
            # velocity += reg_velocity # 이거 쓰면 심하게 뒤틀림
            ##########################################
            
            velocity = torch.mm(velocity, rot_mats.T) * scale
            # relative_rot_mats = torch.matmul(rot_mats, relative_rot_mats) # [human_n_particles, 3, 3]
        else:
            velocity = torch.zeros([human_n_particles, 3], device=self.device)
            relative_rot_mats = torch.eye(3).unsqueeze(0).repeat(human_n_particles, 1, 1)
            colors_next = torch.zeros([human_n_particles, 3], device=self.device)
        
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
                
                self.human_modify_model[list_idx].check_A_now.append(A_now_55)
                self.human_modify_model[list_idx].check_A_next.append(A_next_55)
                self.human_modify_model[list_idx].check_A_next_woRoot.append(A_next_woRoot_55)
                self.human_modify_model[list_idx].check_velocity.append(velocity[bone_index[-1]:])
                # self.human_modify_model[list_idx].check_bone_now.append(particle_x[ps+bone_index[0]:ps+bone_index[-1]])
                
                now_smpl_A = now_smpl.A; next_smpl_A = next_smpl.A; next_smpl_woRoot_A = next_smpl_woRoot.A
                self.human_modify_model[list_idx].gt_A_now.append(now_smpl_A)
                self.human_modify_model[list_idx].gt_A_next.append(next_smpl_A)
                self.human_modify_model[list_idx].gt_A_next_woRoot.append(next_smpl_woRoot_A)        
                self.human_modify_model[list_idx].gt_velocity.append(velocity_gt)
                
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
        
        rot_mats44 = torch.eye(4, device=self.device); rot_mats44[:3, :3] = rot_mats
        
        A_next_55_local = torch.matmul(A_next_55[0], rot_mats44.T)
        A_next_55_local[:, :3, 3] = (A_next_55_local[:, :3, 3] - ori_mean) * scale + center # [55, 4, 4]
        
        joint_mat_next_local = torch.matmul(joint_mat_next, rot_mats44.T)
        joint_mat_next_local[:, :3, 3] = (joint_mat_next_local[:, :3, 3] - ori_mean) * scale + center # [55, 4, 4]
                
        return velocity, relative_rot_mats, lbs #, joint_mat_next_local, A_next_55 # , torch.flip(colors_next, dims=[1])
'''