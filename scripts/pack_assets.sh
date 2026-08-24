#!/usr/bin/env bash
# Build the public asset tarball. Includes redistributed third-party data
# (SMPL-X, AMASS-derived clips, AnimatableGaussians avatarrex_zzr, VPoser) with
# attribution — official sources are linked in ASSETS.md.
# EXCLUDED: third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz — derived
# from MPMAvatar / ActorsHQ data with no redistribution license; add it only
# after obtaining the authors' permission (see ASSETS.md).
set -eo pipefail
cd "$(dirname "$0")/.."
tar --exclude='smpl_models/smplx/smplx_kid_template.npy' \
    --exclude='smpl_models/smplx/*.pkl' \
    --exclude='smpl_models/smplx/smplx_npz.zip' \
    --exclude='osso_per_parts/*.ply' \
    --exclude='a1_s1/a1_canonical_assets.npz' \
    -czf piavatar_assets.tar.gz \
  model smpl_models AnimatableGaussians/datasets third_party/MPMAvatar/data
ls -lh piavatar_assets.tar.gz
