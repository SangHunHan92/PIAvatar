#!/usr/bin/env bash
# One-shot environment setup. Run from the repository root:
#     bash setup_env.sh
set -eo pipefail   # no -u: conda compiler activation scripts reference unset vars
ROOT=$(cd "$(dirname "$0")" && pwd)
ENV_NAME=${ENV_NAME:-piavatar}

# 1. conda env + pip requirements -------------------------------------------
conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml" || echo "(env exists, continuing)"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"      # win over any env hard-coded in ~/.bashrc
PIP="$CONDA_PREFIX/bin/python -m pip"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
# kaolin/open3d need the env's libstdc++ ahead of the system one (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# conda's gcc does not search the system lib dir: make libcuda.so (driver) visible to the JIT linker (3dgrut OptiX tracers)
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/piavatar.sh" <<EOT
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="\$CONDA_PREFIX/lib:\$CONDA_PREFIX/lib/stubs:/usr/lib/x86_64-linux-gnu:\${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
EOT

# 2. CUDA wheels that must match torch 2.1.2 + cu118 ------------------------
$PIP install "setuptools>=68,<70" wheel ninja   # torch 2.1 cpp_extension still imports pkg_resources
$PIP install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html
$PIP install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# 3. CUDA extensions from source --------------------------------------------
$PIP install --no-build-isolation "$ROOT/gaussian-splatting/submodules/diff-gaussian-rasterization"
$PIP install --no-build-isolation "$ROOT/gaussian-splatting/submodules/simple-knn"
$PIP install --no-build-isolation "$ROOT/AnimatableGaussians/gaussians/diff_gaussian_rasterization_depth_alpha"
(cd "$ROOT/AnimatableGaussians/network/styleunet" && "$CONDA_PREFIX/bin/python" setup.py install && rm -rf build dist ./*.egg-info)   # fused + upfirdn2d (two setup() calls -> legacy install)

# 4. 3DGRUT renderer (NVIDIA, Apache-2.0) -----------------------------------
if [ ! -d "$ROOT/3dgrut" ]; then
  git clone --recursive https://github.com/nv-tlabs/3dgrut.git "$ROOT/3dgrut"
  git -C "$ROOT/3dgrut" checkout 43947a7
  git -C "$ROOT/3dgrut" apply "$ROOT/third_party/3dgrut_piavatar.patch"
fi
$PIP install --no-build-isolation -r "$ROOT/3dgrut/requirements.txt" "kornia==0.8.1" "kornia_rs==0.1.9"   # kornia>=0.8.2 breaks on torch 2.1
$PIP install -e "$ROOT/3dgrut"
# threedgrut_playground is not declared in 3dgrut's setup.py -> expose the checkout on sys.path
echo "$ROOT/3dgrut" > "$("$CONDA_PREFIX/bin/python" -c 'import site;print(site.getsitepackages()[0])')/threedgrut_playground.pth"

# 5. sanity -----------------------------------------------------------------
$CONDA_PREFIX/bin/python - <<'PY'
import torch, warp, taichi, kaolin, pytorch3d, smplx, trimesh, threedgrut_playground
print("torch", torch.__version__, "cuda", torch.version.cuda, "| warp", warp.__version__, "| taichi", taichi.__version__,
      "| kaolin", kaolin.__version__, "| pytorch3d", pytorch3d.__version__)
import diff_gaussian_rasterization, simple_knn, diff_gaussian_rasterization_depth_alpha, fused, upfirdn2d
print("CUDA extensions OK")
PY
echo "Environment ready:  conda activate $ENV_NAME"
