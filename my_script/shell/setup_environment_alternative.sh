#!/bin/bash

# Alternative setup script for SGG-Benchmark Environment with PyTorch fix
echo "Setting up SGG-Benchmark environment (alternative method)..."

# Remove existing environment if it exists
conda env remove --name sgg_benchmark -y

# Create conda environment
conda create --name sgg_benchmark python=3.11 -y

# Activate and install conda dependencies (except PyTorch)
conda activate sgg_benchmark && conda install ipython scipy h5py ninja cython matplotlib tqdm pandas -y

# Install PyTorch via pip to avoid TBB conflicts
conda activate sgg_benchmark && pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# Change directory to project root
cd SGG-Benchmark

# Install other requirements
conda activate sgg_benchmark && pip install -r requirements.txt

# Install CLIP from GitHub
conda activate sgg_benchmark && pip install git+https://github.com/openai/CLIP.git

# Build the project
conda activate sgg_benchmark && pip install -e . --no-build-isolation --config-settings editable_mode=compat
# conda activate sgg_benchmark && pip install -e . --no-build-isolation
# conda activate sgg_benchmark && python setup.py build develop

# Check
conda activate sgg_benchmark && python -c "import sgg_benchmark; print('SGG-Benchmark installed successfully')"
conda activate sgg_benchmark && python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
conda activate sgg_benchmark && python -c "from sgg_benchmark.config import cfg; print('Config loaded successfully')"
conda activate sgg_benchmark && python -c "from ultralytics import YOLO; print('Ultralytics YOLO available')"

echo "Environment setup complete!"
echo "Activate with: conda activate sgg_benchmark"
echo ""
echo "Test PyTorch installation:"
echo "conda activate sgg_benchmark && python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"