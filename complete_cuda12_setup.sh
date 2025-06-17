#!/bin/bash
# Script to complete CUDA 12.4 setup after installation

echo "CUDA 12.4 Post-Installation Setup"
echo "================================="

# Step 1: Verify CUDA installation
echo "Step 1: Verifying CUDA 12.4 installation..."
if [ -d "/usr/local/cuda-12.4" ]; then
    echo "✓ CUDA 12.4 directory found"
else
    echo "✗ CUDA 12.4 directory not found! Installation may have failed."
    exit 1
fi

# Step 2: Set up environment variables
echo -e "\nStep 2: Setting up environment variables..."
CUDA_SETUP="
# CUDA 12.4 environment variables
export PATH=/usr/local/cuda-12.4/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
"

# Add to bashrc if not already present
if ! grep -q "CUDA 12.4 environment" ~/.bashrc; then
    echo "$CUDA_SETUP" >> ~/.bashrc
    echo "✓ Added CUDA environment to ~/.bashrc"
else
    echo "✓ CUDA environment already in ~/.bashrc"
fi

# Apply immediately
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4

# Step 3: Verify nvcc version
echo -e "\nStep 3: Checking nvcc version..."
/usr/local/cuda-12.4/bin/nvcc --version

# Step 4: Reinstall CuPy for CUDA 12
echo -e "\nStep 4: Installing CuPy for CUDA 12..."
pip3 uninstall -y cupy cupy-cuda11x cupy-cuda12x 2>/dev/null
pip3 install cupy-cuda12x

# Step 5: Test GPU acceleration
echo -e "\nStep 5: Testing GPU acceleration..."
python3 << 'EOF'
import sys
try:
    import cupy as cp
    print(f"✓ CuPy version: {cp.__version__}")
    
    # Check CUDA version
    cuda_version = cp.cuda.runtime.runtimeGetVersion()
    print(f"✓ CUDA Runtime Version: {cuda_version // 1000}.{(cuda_version % 1000) // 10}")
    
    # Test basic operation
    a = cp.array([1, 2, 3])
    b = a * 2
    print(f"✓ Basic GPU operation: {a} * 2 = {b}")
    
    # Test FFT
    signal = cp.random.randn(1024).astype(cp.float32)
    fft_result = cp.fft.rfft(signal)
    print(f"✓ GPU FFT successful! Result shape: {fft_result.shape}")
    
    # Get GPU info
    device = cp.cuda.Device()
    print(f"✓ GPU Name: {device.name}")
    print(f"✓ Compute Capability: {device.compute_capability}")
    print(f"✓ Total Memory: {device.mem_info[1] / 1024**3:.1f} GB")
    
    print("\n🎉 GPU acceleration is fully functional!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Step 6: Update ldconfig
echo -e "\nStep 6: Updating library cache..."
sudo ldconfig /usr/local/cuda-12.4/lib64

echo -e "\n================================="
echo "Setup complete! Next steps:"
echo "1. Run: source ~/.bashrc"
echo "2. Test OMEGA-4: python3 omega4_main.py"
echo "3. GPU acceleration should now work with your RTX 4080!"
echo "================================="