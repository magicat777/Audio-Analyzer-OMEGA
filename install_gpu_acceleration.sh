#!/bin/bash
# GPU Acceleration Installation Script for OMEGA-4

echo "OMEGA-4 GPU Acceleration Setup"
echo "=============================="

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Checking CUDA version..."
    
    # Get CUDA version from nvidia-smi
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1-2)
    
    if [ -n "$cuda_version" ]; then
        echo "CUDA version detected: $cuda_version"
        
        # Determine which CuPy package to install
        cuda_major=$(echo $cuda_version | cut -d'.' -f1)
        
        if [ "$cuda_major" = "11" ]; then
            echo "Installing CuPy for CUDA 11.x..."
            pip install cupy-cuda11x
        elif [ "$cuda_major" = "12" ]; then
            echo "Installing CuPy for CUDA 12.x..."
            pip install cupy-cuda12x
        else
            echo "Unsupported CUDA version. Please install CuPy manually."
            echo "Visit: https://docs.cupy.dev/en/stable/install.html"
            exit 1
        fi
        
        # Verify installation
        echo ""
        echo "Verifying GPU acceleration..."
        python3 -c "
import sys
try:
    import cupy as cp
    print(f'✓ CuPy installed successfully')
    print(f'✓ Number of GPUs: {cp.cuda.runtime.getDeviceCount()}')
    device = cp.cuda.Device(0)
    print(f'✓ GPU Name: {device.name.decode()}')
    print(f'✓ Compute Capability: {device.compute_capability}')
    print('')
    print('GPU acceleration is ready!')
except Exception as e:
    print(f'✗ Error: {e}')
    print('GPU acceleration setup failed.')
    sys.exit(1)
"
    else
        echo "Could not detect CUDA version. Please check your NVIDIA drivers."
        exit 1
    fi
else
    echo "No NVIDIA GPU detected. GPU acceleration will not be available."
    echo "The visualizer will still work using CPU-based processing."
    exit 0
fi

echo ""
echo "Installation complete!"
echo ""
echo "To use GPU acceleration, the visualizer will automatically detect and use it."
echo "You can verify GPU usage by pressing 'D' while the visualizer is running."