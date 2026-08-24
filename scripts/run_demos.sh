#!/usr/bin/env bash
# Run every demo in the paper order. Each run writes output/<name>/{3dgrut/output.mp4, output.mp4, config.json}.
# Usage:  bash scripts/run_demos.sh [all|main|stickiness|selfpen|soft|cloth]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
what=${1:-all}
run() { echo ">>> $*"; python "$@"; }

if [[ $what == all || $what == main ]]; then
  run simulation.py --config configs/main_method/rucksack_walk.json
  run simulation.py --config configs/main_method/rucksack_walk.json --method vanila --output_path output/main_rucksack_walk_basic_mpm
  run simulation.py --config configs/main_method/low_kick.json
  run simulation.py --config configs/main_method/high_five.json
fi
if [[ $what == all || $what == stickiness ]]; then
  run simulation_separable_contact.py --config configs/stickiness/a_no_shape_preservation.json
  run simulation_separable_contact.py --config configs/stickiness/b_single_field_contact.json
  run simulation_separable_contact.py --config configs/stickiness/c_ours.json
fi
if [[ $what == all || $what == selfpen ]]; then
  run simulation_lbs_decouple.py --config configs/self_penetration/a_single_field.json
  run simulation_lbs_decouple.py --config configs/self_penetration/b_multi_field.json
fi
if [[ $what == all || $what == soft ]]; then
  run simulation.py --config configs/soft_tissue/belly_jump.json
fi
if [[ $what == all || $what == cloth ]]; then
  if [ -f third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz ]; then
    run simulation_with_cloth.py --config configs/cloth/spin_jump.json
  else
    echo "[skip] loose-garment demo: third_party/MPMAvatar/data/a1_s1/a1_canonical_assets.npz not found (see ASSETS.md §C)"
  fi
fi
