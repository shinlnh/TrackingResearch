#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/HELIOS/MyECOTracker}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCH_BOX_URL="${TORCH_BOX_URL:-https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl}"

mkdir -p "$PROJECT_ROOT" "$HOME/opt/libomp_pkgs" "$HOME/opt/libomp_root" \
         "$HOME/opt/openmpi_pkgs" "$HOME/opt/openmpi_root" \
         "$HOME/opt/hwloc_pkgs" "$HOME/opt/hwloc_root" \
         "$HOME/opt/torch_wheels"

rm -rf "$VENV_DIR"
python3 -m virtualenv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/pip" install --no-cache-dir "numpy==1.19.4" "pillow<9" pyyaml tqdm visdom

cd "$HOME/opt/libomp_pkgs"
apt-get download libomp5-8 libomp-8-dev >/dev/null
for deb in ./*.deb; do dpkg-deb -x "$deb" "$HOME/opt/libomp_root"; done

cd "$HOME/opt/openmpi_pkgs"
apt-get download libopenmpi2 >/dev/null
for deb in ./*.deb; do dpkg-deb -x "$deb" "$HOME/opt/openmpi_root"; done

cd "$HOME/opt/hwloc_pkgs"
apt-get download libhwloc5 libhwloc-plugins >/dev/null
for deb in ./*.deb; do dpkg-deb -x "$deb" "$HOME/opt/hwloc_root"; done

cd "$HOME/opt/torch_wheels"
wget --content-disposition -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl "$TORCH_BOX_URL"

export LD_LIBRARY_PATH="$HOME/opt/libomp_root/usr/lib/llvm-8/lib:$HOME/opt/openmpi_root/usr/lib/aarch64-linux-gnu:$HOME/opt/openmpi_root/usr/lib/aarch64-linux-gnu/openmpi/lib:$HOME/opt/hwloc_root/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$VENV_DIR/bin/pip" install --no-cache-dir ./torch-1.10.0-cp36-cp36m-linux_aarch64.whl

echo
echo "Bootstrap complete."
echo "Next:"
echo "  source \"$PROJECT_ROOT/jetson/activate_verified936_env.sh\""
echo "  bash \"$PROJECT_ROOT/jetson/run_verified936.sh\" smoke"
