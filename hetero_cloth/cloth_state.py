"""ClothStateStruct: cloth-specific state fields, parallel to user's MPMStateStruct.

Indexing convention:
    Total particles N_total = n_existing + n_elements + n_vertices.
    Cloth elements live at [n_existing, n_existing + n_elements).
    Cloth vertices live at [n_existing + n_elements, n_existing + n_elements + n_vertices).

    particle_traditional[i] == 1  if i is a user-existing particle (body / object / sand-like)
    particle_elements[i]    == 1  if i is a cloth element (face centroid)
    particle_vertices[i]    == 1  if i is a cloth vertex node

    Exactly one of the three flags is 1 for each i; others are 0.

Per-element fields (particle_d, particle_R_inv, particle_D_inv, faces) are
allocated with shape n_elements. To index from a particle index i, use
i - n_existing (for elements) or i - n_existing - n_elements (for vertices).
This offset is passed explicitly to kernels rather than baked into arrays.

vertex_force is allocated with shape n_vertices and indexed
[i - n_existing - n_elements] for vertex particles.
"""
import warp as wp


@wp.struct
class ClothStateStruct:
    # per-particle flags (full N_total length, matching MPMStateStruct.particle_x indexing)
    particle_traditional: wp.array(dtype=int)
    particle_vertices: wp.array(dtype=int)
    particle_elements: wp.array(dtype=int)

    # per-element fields (length n_elements; access by element_local_idx = particle_idx - n_existing)
    particle_d: wp.array(dtype=wp.mat33)       # 3D deformation matrix [d1 | d2 | d3]
    particle_R_inv: wp.array(dtype=wp.vec3)    # rest-direction R inverse upper-tri (iD11, iD12, iD22)
    particle_D_inv: wp.array(dtype=wp.mat33)   # rest-direction matrix inverse (full 3x3)
    faces: wp.array(dtype=wp.vec3)             # (v1, v2, v3) per element, stored as vec3 of float-cast ints

    # per-vertex field (length n_vertices; access by vertex_local_idx = particle_idx - n_existing - n_elements)
    vertex_force: wp.array(dtype=wp.vec3)

    # cloth-only material params (per element; user's MPMModelStruct has no gamma/kappa/friction_coeff)
    gamma: wp.array(dtype=float)            # shear stiffness  (R[0,2], R[1,2] terms)
    kappa: wp.array(dtype=float)            # thickness stiffness (R[2,2] compression term)
    friction_coeff: float                   # tan(friction_angle); for anisotropy_return_mapping

    # LBS-pin: top cloth verts kinematically follow body each substep
    # pin_mask[v_local] == 1 -> particle_v[p] is overwritten with pin_target_v[v_local]
    # pin_target_v is computed once per frame from now/next body pose via cloth_lbs_w
    pin_mask: wp.array(dtype=int)           # length n_vertices; 1 = pinned, 0 = free
    pin_target_v: wp.array(dtype=wp.vec3)   # length n_vertices; MPM-space velocity

    # offsets so kernels can convert particle index -> local index
    n_existing: int    # number of pre-cloth particles
    n_elements: int    # number of cloth elements
    n_vertices: int    # number of cloth vertices

    def init(self, n_existing: int, n_elements: int, n_vertices: int,
             device="cuda:0", requires_grad: bool = True):
        n_total = n_existing + n_elements + n_vertices

        self.particle_traditional = wp.zeros(shape=n_total, dtype=int, device=device, requires_grad=False)
        self.particle_vertices    = wp.zeros(shape=n_total, dtype=int, device=device, requires_grad=False)
        self.particle_elements    = wp.zeros(shape=n_total, dtype=int, device=device, requires_grad=False)

        self.particle_d     = wp.zeros(shape=n_elements, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        self.particle_R_inv = wp.zeros(shape=n_elements, dtype=wp.vec3,  device=device, requires_grad=requires_grad)
        self.particle_D_inv = wp.zeros(shape=n_elements, dtype=wp.mat33, device=device, requires_grad=False)
        self.faces          = wp.zeros(shape=n_elements, dtype=wp.vec3,  device=device, requires_grad=False)

        self.vertex_force   = wp.zeros(shape=n_vertices, dtype=wp.vec3, device=device, requires_grad=requires_grad)

        self.gamma = wp.zeros(shape=n_elements, dtype=float, device=device, requires_grad=requires_grad)
        self.kappa = wp.zeros(shape=n_elements, dtype=float, device=device, requires_grad=requires_grad)
        self.friction_coeff = 0.0  # scalar; set via set_cloth_material()

        self.pin_mask     = wp.zeros(shape=n_vertices, dtype=int, device=device, requires_grad=False)
        self.pin_target_v = wp.zeros(shape=n_vertices, dtype=wp.vec3, device=device, requires_grad=False)

        self.n_existing = n_existing
        self.n_elements = n_elements
        self.n_vertices = n_vertices
