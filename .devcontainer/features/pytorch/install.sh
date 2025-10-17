#!/usr/bin/env bash

set -e

VERSION=${VERSION:-"2.2.0"}
CUDA=${CUDA:-"false"}

# Configure pip to allow breaking system packages
python3 -m pip config set global.break-system-packages true

echo "Installing PyTorch ${VERSION} (CUDA support: ${CUDA})"

# Install PyTorch based on version and CUDA preference
if [ "$CUDA" = "true" ]; then
    # Install CUDA version of PyTorch
    if [ "$VERSION" = "latest" ]; then
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        python3 -m pip install torch==$VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    # Install CPU-only version
    if [ "$VERSION" = "latest" ]; then
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        python3 -m pip install torch==$VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Make script executable
chmod +x $0

echo "PyTorch installation completed successfully!"