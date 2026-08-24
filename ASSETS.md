# Assets guide

The GitHub repository contains **code only** (≈13 MB tracked). Everything else
is classified below.

## A. Not needed to run / created automatically — nothing to download

| Path | Size | Note |
|---|---|---|
| `3dgrut/` | ~300 MB | Cloned and patched automatically by `setup_env.sh` (nv-tlabs/3dgrut@43947a7). |
| `output/` | grows with use | Rendered videos/frames + per-frame ply written by the demo runs. Fully reproducible with `scripts/run_demos.sh`. |

## B. Required — all in one archive: `piavatar_assets.tar.gz` (≈550 MB)

Download from the link in the README and extract in the repository root.
Built with `bash scripts/pack_assets.sh`.

Ours:

| Path | Size | What it is |
|---|---|---|
| `AnimatableGaussians/datasets/Actor{01,05}/Sequence1/osso/osso_per_parts/part_split_meshes.glb` | 3.3 MB ×2 | OSSO skeleton meshes fitted to the demo bodies (bone particles) |
| `AnimatableGaussians/datasets/Actor01/Sequence1/actorshq_smpl_params.npz` | 1.5 MB | body shape params for the cloth demo body |
| `model/pillow2sofa_whitebg-trained/point_cloud/iteration_30000/` | 151 MB | 3DGS reconstruction of the pillow (our BlenderNeRF capture) |
| `model/MeshAsset/{football,rucksack}/*.glb` | 15 MB | object meshes, via Sketchfab (links and credits below) |

Redistributed third-party data (kept in the archive for convenience — each item
remains under its original license; official sources below):

| Path | Size | Source / license |
|---|---|---|
| `smpl_models/smplx/SMPLX_{MALE,NEUTRAL,FEMALE}.npz` | 313 MB | SMPL-X, © Max Planck Institute for Intelligent Systems — https://smpl-x.is.tue.mpg.de (research-only license) |
| `AnimatableGaussians/datasets/pose/AMASS/CMU/*` | ~22 MB | AMASS (CMU subset, SMPL-X G) — https://amass.is.tue.mpg.de (research-only license). `11_01_poses_modified.npz` and `10_01_poses_rightleg_lower_from650_scale0p65.npz` are our hand-edited derivatives. |
| `AnimatableGaussians/datasets/avatarrex_zzr/` | ~100 MB | AvatarReX preprocessed data from Animatable Gaussians — https://github.com/lizhe00/AnimatableGaussians (PREPROCESSED_DATASET.md) |
| `third_party/MPMAvatar/data/body_models/TR00_E096.pt` | 2.6 MB | VPoser v1.0 checkpoint, © MPI — https://smpl-x.is.tue.mpg.de |
| `smpl_models/mano/` | 60 KB | SMPL-X↔MANO hand mapping, from the Animatable Gaussians repository |
| `third_party/MPMAvatar/data/a1_s1/a1_amass_spin.npz` | 272 KB | pose clip we synthesized from AMASS sequences |

## C. NOT in the archive — `a1_canonical_assets.npz` (loose-garment demo)

`third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz` (8 MB) is derived
from [MPMAvatar](https://github.com/KAISTChangmin/MPMAvatar)'s a1_s1 tracked
data, which in turn derives from [ActorsHQ](https://actors-hq.com/)
(request-form license). The MPMAvatar repository publishes **no license file**,
so we do not redistribute this file until the authors grant permission.

To run the loose-garment demo, obtain MPMAvatar's a1_s1 assets from their
repository's download links and place/derive
`third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz`
(cloth template vertices/faces/LBS weights + body canonical assets in the
schema read by `hetero_cloth/cloth_integration.py`). All other demos run
without it.

## Object mesh credits (Sketchfab)

- `football.glb`, `rucksack.glb` — CC-licensed Sketchfab assets; **[add the
  exact asset URLs and creator names here before release]**.

## Optional avatar checkpoints (not needed by any bundled demo)

The bundled demos all use procedural SMPL-X bodies, so no learned avatar
checkpoint is required. Clothed Animatable-Gaussians avatars (ActorsHQ
Actor01/05/06) used for other paper results are not distributed.
