# PIAvatar: Physically Interactive Avatars via Deformation Gradient Decoupling

Official code for **PIAvatar** — an MPM-based avatar simulation framework that
(1) decouples the user-defined kinematic velocity from the deformation-gradient
update so that pose-driven motion induces no restorative stress, and
(2) embeds an OSSO skeleton inside the avatar so that its pose can be tracked in
closed form (Kabsch) even while it deforms under contact.

This repository contains the simulator and ready-to-run configurations for the
experiments in the paper:

| Experiment | Entry script | Config(s) |
|---|---|---|
| Basic MPM vs. ours (kinematic deformation decoupling) | `simulation.py` | `configs/main_method/rucksack_walk.json` |
| Avatar–avatar interaction (low kick, high five) | `simulation.py` | `configs/main_method/{low_kick,high_five}.json` |
| Permanent deformation / **stickiness artifact** | `simulation_separable_contact.py` | `configs/stickiness/{a,b,c}_*.json` |
| **Self-penetration** (multi-field contact between body parts) | `simulation_lbs_decouple.py` | `configs/self_penetration/{a,b}_*.json` |
| **Soft-tissue** (belly jiggle) | `simulation.py` | `configs/soft_tissue/belly_jump.json` |
| **Loose-garment** simulation (coupled thin-shell cloth) | `simulation_with_cloth.py` | `configs/cloth/spin_jump.json` |

The VLM-driven parameter optimisation is not part of this release.

---

## 1. Installation

Tested on Ubuntu 20.04, NVIDIA RTX 4090 (driver 535, CUDA 11.8), Python 3.11,
PyTorch 2.1.2+cu118, warp-lang 1.7.1. An OptiX-capable GPU is required for the
3DGRUT renderer.

```bash
git clone https://github.com/SangHunHan92/PIAvatar.git
cd PIAvatar
bash setup_env.sh            # creates conda env "piavatar" and builds all CUDA extensions
conda activate piavatar
```

`setup_env.sh` performs, in order:

1. `conda env create -f environment.yml` — Python 3.11, CUDA 11.8 toolkit, faiss-gpu, ffmpeg, and the pinned `requirements.txt`.
2. `kaolin==0.17.0` (NVIDIA wheel for torch 2.1.2 + cu118) and `pytorch3d==0.7.8`.
3. CUDA extensions from source: `diff-gaussian-rasterization`, `simple-knn`, `diff_gaussian_rasterization_depth_alpha`, `fused`/`upfirdn2d` (AnimatableGaussians StyleUNet).
4. Clones [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) at commit `43947a7`, applies `third_party/3dgrut_piavatar.patch`, and installs it in editable mode (`threedgrut_playground` is the ray-traced renderer used for all results).
5. Registers `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` in the env's `activate.d` — required, otherwise kaolin/open3d fail with `GLIBCXX_3.4.29 not found` on Ubuntu 20.04.

Manual install: follow the same steps in `setup_env.sh` — every version is pinned in `environment.yml` / `requirements.txt`.

---

## 2. Assets

The repository contains code only. Download
[`piavatar_assets.tar.gz`](https://drive.google.com/file/d/1lne-a_8KgmWxA3vScRxJiqS4hm5l8sNI/view?usp=sharing)
(≈550 MB) and extract it in the repository root:

```bash
pip install gdown
gdown 1lne-a_8KgmWxA3vScRxJiqS4hm5l8sNI
tar -xzf piavatar_assets.tar.gz
```

It contains
everything the demos need, including redistributed third-party data (SMPL-X,
AMASS clips, AvatarReX preprocessed data, VPoser) under their original
licenses; sources and credits are listed in [ASSETS.md](ASSETS.md).

Exception: the loose-garment demo additionally needs
`third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz`, which we cannot
redistribute (MPMAvatar/ActorsHQ-derived, no license granted) — see ASSETS.md
§C. All other demos run out of the box.

The archive restores the layout below, which the configs reference with
relative paths (always run from the repo root).

```
PIAvatar/
├── smpl_models/smplx/SMPLX_{MALE,NEUTRAL,FEMALE}.npz     # (*)
├── smpl_models/mano/smplx_*hand_to_mano_*.npz             # hand-pose mapping (from AnimatableGaussians)
├── AnimatableGaussians/datasets/
│   ├── Actor01/Sequence1/osso/osso_per_parts/part_split_meshes.glb   # OSSO skeleton (fat body, cloth body)
│   ├── Actor05/Sequence1/osso/osso_per_parts/part_split_meshes.glb   # OSSO skeleton (2-person scenes)
│   ├── Actor01/Sequence1/actorshq_smpl_params.npz
│   ├── avatarrex_zzr/{calibration_full.json,smpl_params.npz,smpl_pos_map/}  # AG preprocessed data (SMPL-X loader)
│   └── pose/AMASS/CMU/**/*_poses.npz                                  # (*)
├── model/MeshAsset/{football,rucksack}/*.glb               # objects (Sketchfab, CC-BY)
├── model/pillow2sofa_whitebg-trained/point_cloud/...       # 3DGS pillow (BlenderNeRF)
├── data/render/                                            # renderer bootstrap (ships with the repo)
├── data/render/{gs_seed/point_cloud.ply, mesh_assets/sphere.obj, cameras.json}   # renderer bootstrap
└── third_party/MPMAvatar/data/
    ├── a1_s1/a1_amass_spin.npz                            # pose clip (a1_canonical_assets.npz: see ASSETS.md §C)
    └── body_models/{TR00_E096.pt, smplx -> ../../../../smpl_models/smplx}        # VPoser ckpt + symlink
```

(*) redistributed under the original SMPL-X / AMASS research licenses — see ASSETS.md for sources and credits.

Maintainers: `bash scripts/collect_assets.sh <dev-workspace>` gathers the files
from the development tree, `bash scripts/pack_assets.sh` builds the tarball.

---

## 3. Running the demos

```bash
conda activate piavatar
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_demos.sh            # everything, or: main | stickiness | selfpen | soft | cloth
```

All entry scripts accept `--config <json>`, `--output_path <dir>` (overrides the
config's `sim_params.output_path`) and `--method ours|vanila`.

**Where results go.** Every run writes to `output/<sim_params.name>/` (relative to the repo root):

```
output/<name>/
├── 3dgrut/output.mp4        # ray-traced video (used for all results)
├── 3dgrut/0000.png ...      # per-frame ray-traced renders
├── output.mp4, 0000.png ... # 3DGS-rasterizer video / frames
├── simulation_ply/          # per-frame particle positions (if sim_params.output_ply)
├── cloth_ply/               # per-frame cloth vertices (cloth demo only)
├── config.json              # exact config used
└── *.py                     # snapshot of the solver sources used for the run
```

Output directories of the bundled configs:

| Config | Output directory |
|---|---|
| `main_method/rucksack_walk.json` | `output/main_rucksack_walk/` (with `--method vanila`: `output/main_rucksack_walk_basic_mpm/`) |
| `main_method/low_kick.json`, `high_five.json` | `output/main_low_kick/`, `output/main_high_five/` |
| `stickiness/{a,b,c}_*.json` | `output/stickiness_a_alpha0/`, `output/stickiness_b_single_field/`, `output/stickiness_c_ours/` |
| `self_penetration/{a,b}_*.json` | `output/selfpen_a_single_field/`, `output/selfpen_b_multi_field/` |
| `soft_tissue/belly_jump*.json` | `output/soft_tissue_belly_jump/`, `output/soft_tissue_belly_jump_noatt/` |
| `cloth/spin_jump.json` | `output/cloth_spin_jump/` |

### 3.1 Main method — kinematic deformation decoupling

```bash
# ours: F ← F (F^k)^-1, the kinematic part never produces stress
python simulation.py --config configs/main_method/rucksack_walk.json
# basic MPM: same scene, deformation gradient updated by the full velocity field
python simulation.py --config configs/main_method/rucksack_walk.json \
       --method vanila --output_path output/main_rucksack_walk_basic_mpm
# avatar–avatar interactions
python simulation.py --config configs/main_method/low_kick.json
python simulation.py --config configs/main_method/high_five.json
```

### 3.2 Stickiness / permanent deformation

Fat SMPL-X avatar (`betas=[0.6,4.8,4.2,3.8,…]`) hit by a 0.5 kg soccer ball.
The three configs differ only in two keys:

| Case | Config | `velocity_alpha` (shape preservation) | `use_separable_contact` (multi-field contact) |
|---|---|---|---|
| (a) permanent dent | `a_no_shape_preservation.json` | 0.0 | true |
| (b) sticky contact | `b_single_field_contact.json` | 0.2 | false |
| (c) ours | `c_ours.json` | 0.4 | true |

```bash
for c in a_no_shape_preservation b_single_field_contact c_ours; do
  python simulation_separable_contact.py --config configs/stickiness/$c.json
done
```

### 3.3 Self-penetration

Two SMPL-X avatars and a 3DGS pillow. The red avatar is knocked down; with a
single grid field its legs fuse on contact. `simulation_lbs_decouple.py` assigns
each particle a *super-tag* (a group of parent/child joints derived from its LBS
weights) and runs the Bardenhagen multi-field contact between super-tags, so
different body parts of the same avatar are separate momentum fields.

```bash
python simulation_lbs_decouple.py --config configs/self_penetration/a_single_field.json   # use_lbs_decouple=false
python simulation_lbs_decouple.py --config configs/self_penetration/b_multi_field.json    # use_lbs_decouple=true
```

### 3.4 Soft-tissue deformation

Fat avatar performing a jump (AMASS CMU 01_01, frames 330–810). The kinematic
velocity of belly particles is attenuated (`sim_params.belly_attenuation=true`:
LBS weight of spine joint 3 zeroed, joint 6 scaled by 1/5) so the belly is driven
by neighbouring tissue rather than prescribed directly, and inertial jiggle emerges.

```bash
python simulation.py --config configs/soft_tissue/belly_jump.json
python simulation.py --config configs/soft_tissue/belly_jump_no_attenuation.json   # ablation
```

### 3.5 Loose-garment simulation

SMPL-X body (MPMAvatar `a1_s1` shape) wearing the MPMAvatar dress template,
driven by the AMASS *spin + jump* clip. The body runs in the volumetric MPM
solver, the dress in a coupled thin-shell MPM solver (QR-decomposed deformation
gradient, `hetero_cloth/`), both on the same grid with multi-field contact and
a body-mesh collider.

```bash
python simulation_with_cloth.py --config configs/cloth/spin_jump.json
```

Requires `third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz`, which is
not in the asset archive — see [ASSETS.md](ASSETS.md) §C.

Cloth parameters live under `"cloth"` in the config (`thickness`, `density`, `E`,
`nu`, `gamma`, `kappa`, `outward_offset`, `lbs_pin_top_pct`, `use_body_mesh_collider`).

---

## 4. Config reference

Top-level: `substep_dt`, `frame_dt`, `smplx_dt`, `frame_num`, `n_grid`, `g`,
camera (`mpm_space_viewpoint_center`, `init_azimuthm`, `init_elevation`, `init_radius`, …), `smplx_path`.
`frame_dt` is the render/output interval, `smplx_dt` the interval between consecutive poses of the
input sequence — `smplx_dt = 2 × frame_dt` plays the pose sequence at half speed.

`sim_params`

| key | meaning |
|---|---|
| `use_separable_contact` | per-subject multi-field grid with free-slip / no-tension contact resolve (Bardenhagen 2000). |
| `contact_eps` | mass threshold below which a subject is ignored at a node. |
| `use_lbs_decouple` | (`simulation_lbs_decouple.py` only) super-tag fields per body part. |
| `belly_attenuation` | soft-tissue: attenuate kinematic velocity on the belly (spine joints 3/6). |
| `interior_mass_scale`, `bone_mass_scale` | volume scale for filled-interior / bone particles (only with `particle_filling`). |
| `smplx_direct_A` | read joint transforms from SMPL-X directly (default) instead of the Kabsch fit of bone particles. |
| `render_img`, `render_3dgrut`, `compile_video`, `white_bg`, `output_ply`, `output_h5` | outputs. |

`subject_conditions[]` (one per avatar / object)

| key | meaning |
|---|---|
| `human`, `type` (`mesh`/`gaussian`), `model` (`smplx`, `smplx_a1`, `animatable_gaussians`, `mesh_glb`, `gaussian`) | subject kind. |
| `pose_path`, `start_frame`, `end_frame`, `gender`, `betas` | SMPL-X pose sequence (AMASS npz) and shape. |
| `osso_path` | OSSO skeleton directory (bone particles, 10× voxel-downsampled). |
| `velocity_type` (`tgt`/`rel`/`gt`) | how the kinematic velocity is built from consecutive poses. |
| `velocity_alpha` | α of the velocity-level shape-preservation term. |
| `wide_z_arm`, `wide_z_leg`, `wide_y_arm` | canonical-pose limb spreading (deg) to avoid self-contact at rest. |
| `density`, `E`, `nu`, `material` | constitutive parameters (`jelly` = neo-Hookean, `metal`, `sand`, …). |
| `rotation_degree`, `rotation_axis`, `center`, `scale`, `initial_velocity` | placement in the unit simulation cube. |

---

## 5. Repository layout

```
simulation.py                     main method: deformation-gradient decoupling + skeletal pose regression (single-field grid)
simulation_separable_contact.py   + per-subject multi-field contact (Bardenhagen) — stickiness demos
simulation_lbs_decouple.py        + per-body-part (super-tag) multi-field contact — self-penetration demo
simulation_with_cloth.py          + coupled thin-shell cloth solver — loose-garment demo
mpm_solver_warp/                  Warp MPM solver; one file set per entry script
                                   {mpm_solver_warp, mpm_utils, warp_utils, mpm_human_utils}[_separable_contact|_lbs_decouple].py
hetero_cloth/                     thin-shell cloth solver, body-mesh collider, MPMAvatar a1 body loader
particle_filling/                 optional volumetric / thin-shell particle filling
utils/                            config decoding, camera, rendering helpers, SMPL-X mesh construction
AnimatableGaussians/              vendored AG code (SMPL-X implementation, pose dataset, avatar net)
gaussian-splatting/               vendored INRIA 3DGS (rasterizer, scene I/O)
third_party/3dgrut_piavatar.patch small patch on nv-tlabs/3dgrut@43947a7
configs/                          demo configurations (see table above)
scripts/                          run_demos.sh, collect_assets.sh, pack_assets.sh
```

---

## 6. Acknowledgements & licenses

Built on [PhysGaussian](https://github.com/XPandora/PhysGaussian) /
[warp-mpm](https://github.com/zeshunzong/warp-mpm),
[Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians),
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting),
[3DGRUT](https://github.com/nv-tlabs/3dgrut), [OSSO](https://github.com/MarilynKeller/OSSO),
[MPMAvatar](https://github.com/KAISTChangmin/MPMAvatar) and [SMPL-X](https://smpl-x.is.tue.mpg.de).
Each vendored component keeps its original license in its directory.
Redistributed data (SMPL-X, AMASS, AvatarReX preprocessed, VPoser) remains
under its original license — see [ASSETS.md](ASSETS.md).

## License

The PIAvatar code is released under the
[Creative Commons Attribution-NonCommercial 4.0 International](LICENSE.txt)
license, following [2K2K](https://github.com/SangHunHan92/2K2K).

## Citation

```bibtex
@article{han2025piavatar,
  title   = {PIAvatar: Physically Interactive Avatars via Deformation Gradient Decoupling},
  author  = {Han, Sang-Hun and Park, Min-Gyu and Shin, Jisu and Shin, Seunghyun and Park, Jin-Hwi and Jeon, Hae-Gon},
  journal = {arXiv preprint},
  year    = {2025}
}
```
