#!/bin/bash
# Script to re-enable GPU acceleration when CUDA is properly configured

echo "OMEGA-4 GPU Re-enablement Script"
echo "================================"

# Check current status
echo "Current system status:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

echo "Checking CUDA libraries..."
echo "NVRTC libraries found:"
ls -la /usr/lib/x86_64-linux-gnu/libnvrtc.so* 2>/dev/null || echo "  None found in standard location"
echo ""

echo "To enable GPU acceleration, you need to:"
echo "1. Install CUDA toolkit 12.x to match your driver:"
echo "   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
echo "   sudo sh cuda_12.4.0_550.54.14_linux.run"
echo ""
echo "2. Or create symbolic links (temporary fix):"
echo "   sudo ln -s /usr/lib/x86_64-linux-gnu/libnvrtc.so.11.5.119 /usr/lib/x86_64-linux-gnu/libnvrtc.so.12"
echo ""
echo "3. Once CUDA is fixed, edit these files:"
echo "   - omega4/optimization/gpu_accelerated_fft.py"
echo "   - omega4/optimization/batched_fft_processor.py"
echo "   Replace 'CUPY_AVAILABLE = False' with the original try/except import block"
echo ""
echo "4. Test GPU:"
echo "   python3 -c 'import cupy; print(cupy.array([1,2,3]) + 1)'"
echo ""
echo "Current workaround: CPU processing is active and performing well!"