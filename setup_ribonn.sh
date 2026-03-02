#!/usr/bin/env bash
# Setup script for RiboNN as a dependency of RNAFlow.
# Clones the RiboNN repository and downloads pretrained checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIBONN_DIR="${SCRIPT_DIR}/external/RiboNN"

echo "=== RNAFlow: RiboNN Setup ==="

# Clone RiboNN if not already present
if [ ! -d "${RIBONN_DIR}" ]; then
    echo "Cloning RiboNN repository..."
    mkdir -p "${SCRIPT_DIR}/external"
    git clone https://github.com/Sanofi-Public/RiboNN.git "${RIBONN_DIR}"
else
    echo "RiboNN already cloned at ${RIBONN_DIR}"
fi

# Install RiboNN dependencies (assumes conda/mamba is available)
echo ""
echo "To complete RiboNN setup, run:"
echo "  cd ${RIBONN_DIR}"
echo "  make install"
echo "  mamba activate RiboNN"
echo ""
echo "Pretrained models are automatically downloaded from:"
echo "  https://zenodo.org/records/17258709"
echo ""
echo "After activation, add RiboNN src to your PYTHONPATH:"
echo "  export PYTHONPATH=\"${RIBONN_DIR}/src:\${PYTHONPATH:-}\""
echo ""
echo "=== Setup complete ==="
