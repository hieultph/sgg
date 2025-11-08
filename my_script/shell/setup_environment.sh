#!/bin/bash

# Setup SGG-Benchmark Environment
echo "Setting up SGG-Benchmark environment..."

# Create conda environment
conda create --name sgg_benchmark python=3.11 -y

# Activate environment (for current shell session)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sgg_benchmark

# Install PyTorch and dependencies
conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install ipython scipy h5py ninja cython matplotlib tqdm pandas -y

# Install pip requirements
pip install -r requirements.txt

# Build CUDA extensions (optional - code works without them)
echo "Building CUDA extensions (optional)..."
FORCE_CUDA=1 python setup.py build

# Copy compiled extensions
if [ -f "build/lib.linux-x86_64-cpython-311/sgg_benchmark/_C.cpython-311-x86_64-linux-gnu.so" ]; then
    cp build/lib.linux-x86_64-cpython-311/sgg_benchmark/_C.cpython-311-x86_64-linux-gnu.so sgg_benchmark/ 2>/dev/null || echo "Note: CUDA extensions may have library compatibility issues - code will use fallback implementations"
fi

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. conda activate sgg_benchmark"
echo "2. Download dataset (see dataset_guide.md)"
echo "3. Download GloVe embeddings"
echo "4. Update paths in sgg_benchmark/config/paths_catalog.py"
echo "5. Start training! (see READY_TO_TRAIN.md)"