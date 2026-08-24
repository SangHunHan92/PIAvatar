"""hetero_cloth: port of MPMAvatar's anisotropic-membrane cloth model into the
user's existing PhysGaussian-derived MPM solver.

Cloth is added as two new particle groups appended at the END of the existing
particle array:
    indices [0,        n_existing)        = user's existing particles (body / objects / etc.)
    indices [n_existing, n_existing + n_E) = cloth elements (face centroids)
    indices [...,        n_existing + n_E + n_V) = cloth vertices

User's MPMStateStruct stays untouched. Cloth-specific fields live in a separate
ClothStateStruct that is passed alongside MPMStateStruct to cloth-aware kernels.
"""
