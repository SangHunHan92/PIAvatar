#!/usr/bin/env bash
# Collect every data file / checkpoint the demo configs need from the
# development workspace into this release folder (same relative layout the
# code expects).  Run once from the release root:
#     bash scripts/collect_assets.sh /path/to/Physgaussian3
# The resulting data/, model/, smpl_models/, AnimatableGaussians/datasets/ and
# third_party/MPMAvatar/data/ folders are git-ignored; upload them as
# "piavatar_assets.tar.gz" (see scripts/pack_assets.sh) for public release.
set -euo pipefail
SRC=${1:?usage: collect_assets.sh <path-to-dev-workspace>}
DST=$(cd "$(dirname "$0")/.." && pwd)
cp_() { mkdir -p "$DST/$(dirname "$2")"; cp -rL "$SRC/$1" "$DST/$2"; echo "  + $2"; }

echo "[1/6] OSSO skeletons (Actor01 = fat/cloth body, Actor05 = 2-person scenes)"
for a in Actor01 Actor05; do
  mkdir -p "$DST/AnimatableGaussians/datasets/$a/Sequence1/osso"
  mkdir -p "$DST/AnimatableGaussians/datasets/$a/Sequence1/osso/osso_per_parts"
  cp "$SRC/AnimatableGaussians/datasets/$a/Sequence1/osso/osso_per_parts/part_split_meshes.glb" "$DST/AnimatableGaussians/datasets/$a/Sequence1/osso/osso_per_parts/"
  echo "  + AnimatableGaussians/datasets/$a/Sequence1/osso/osso_per_parts"
done
cp_ AnimatableGaussians/datasets/Actor01/Sequence1/smpl_params.npz AnimatableGaussians/datasets/Actor01/Sequence1/actorshq_smpl_params.npz

echo "[2/6] AMASS (CMU) pose sequences used by the demos"
for p in 05/05_01_poses.npz 11/11_01_poses.npz 11/11_01_poses_modified.npz 01/01_01_poses.npz \
         10/10_01_poses_rightleg_lower_from650_scale0p65.npz 05/05_18_poses.npz 02/02_08_poses.npz; do
  cp_ "AnimatableGaussians/datasets/pose/AMASS/CMU/$p" "AnimatableGaussians/datasets/pose/AMASS/CMU/$p"
done

echo "[3/6] AnimatableGaussians avatarrex_zzr preprocessed data (needed to instantiate the SMPL-X loader)"
for f in calibration_full.json smpl_params.npz smpl_pos_map; do
  cp_ "AnimatableGaussians/datasets/avatarrex_zzr/$f" "AnimatableGaussians/datasets/avatarrex_zzr/$f"
done

echo "[4/6] SMPL-X body models (license: https://smpl-x.is.tue.mpg.de — do NOT redistribute publicly)"
mkdir -p "$DST/smpl_models/smplx"
for f in SMPLX_MALE.npz SMPLX_NEUTRAL.npz SMPLX_FEMALE.npz; do cp "$SRC/smpl_models/smplx/$f" "$DST/smpl_models/smplx/"; done
echo "  + smpl_models/smplx/{MALE,NEUTRAL,FEMALE}.npz"
mkdir -p "$DST/smpl_models/mano"; cp "$SRC"/smpl_models/mano/* "$DST/smpl_models/mano/"; echo "  + smpl_models/mano (hand-pose mapping, from AnimatableGaussians)"

echo "[5/6] Objects"
cp_ model/MeshAsset/football/football.glb model/MeshAsset/football/football.glb
cp_ model/MeshAsset/rucksack/rucksack.glb model/MeshAsset/rucksack/rucksack.glb
mkdir -p "$DST/model/pillow2sofa_whitebg-trained/point_cloud"
cp -r "$SRC/model/pillow2sofa_whitebg-trained/point_cloud/iteration_30000" "$DST/model/pillow2sofa_whitebg-trained/point_cloud/"
cp "$SRC/model/pillow2sofa_whitebg-trained/cameras.json" "$DST/model/pillow2sofa_whitebg-trained/" 2>/dev/null || true
echo "  + model/pillow2sofa_whitebg-trained (3DGS pillow, iteration_30000)"

echo "[6/6] MPMAvatar a1_s1 assets for the loose-garment demo"
for f in a1_s1/a1_amass_spin.npz a1_s1/a1_canonical_assets.npz body_models/TR00_E096.pt; do
  cp_ "third_party/MPMAvatar/data/$f" "third_party/MPMAvatar/data/$f"
done
mkdir -p "$DST/third_party/MPMAvatar/data/body_models"
ln -sfn ../../../../smpl_models/smplx "$DST/third_party/MPMAvatar/data/body_models/smplx"
echo "  + third_party/MPMAvatar/data/body_models/smplx -> smpl_models/smplx (symlink)"

echo; echo "done. total:"; du -sh "$DST"
