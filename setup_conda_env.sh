#!/bin/bash
# Script to set up Conda environment for Consistent Character Generator

# Exit on error
set -e

echo "Setting up Conda environment for Consistent Character Generator..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
  echo "Conda not found. Please install Miniconda or Anaconda first."
  echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
  exit 1
fi

# Create conda environment
ENV_NAME="consistent_character"
PYTHON_VERSION="3.10"
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install CUDA toolkit (if needed)
read -p "Do you want to install CUDA toolkit 11.8? (y/n): " install_cuda
if [[ $install_cuda == "y" || $install_cuda == "Y" ]]; then
  conda install -y -c conda-forge cudatoolkit=11.8
fi

# Install PyTorch with appropriate CUDA version
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio

# Install core dependencies
echo "Installing core dependencies..."
pip install \
  einops \
  transformers \
  safetensors \
  aiohttp \
  accelerate \
  pyyaml \
  Pillow \
  scipy \
  tqdm \
  psutil \
  kornia \
  websocket-client \
  diffusers \
  albumentations \
  opencv-python \
  requests

# Install optional dependencies
read -p "Do you want to install additional dependencies for ComfyUI nodes? (y/n): " install_extras
if [[ $install_extras == "y" || $install_extras == "Y" ]]; then
  echo "Installing additional dependencies for ComfyUI nodes..."
  pip install \
    cmake \
    imageio \
    joblib \
    matplotlib \
    pilgram \
    scikit-learn \
    rembg \
    numba \
    pandas \
    numexpr \
    insightface \
    onnx \
    segment-anything \
    piexif \
    ultralytics \
    timm \
    importlib_metadata \
    filelock \
    numpy \
    scikit-image \
    python-dateutil \
    mediapipe \
    svglib \
    fvcore \
    yapf \
    omegaconf \
    ftfy \
    addict \
    yacs \
    trimesh[easy] \
    librosa \
    color-matcher \
    facexlib
fi

# Create a script to activate the environment
ACTIVATE_SCRIPT="activate_consistent_character_env.sh"
echo "#!/bin/bash" > $ACTIVATE_SCRIPT
echo "# Activate the consistent_character conda environment" >> $ACTIVATE_SCRIPT
echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\"" >> $ACTIVATE_SCRIPT
echo "conda activate $ENV_NAME" >> $ACTIVATE_SCRIPT
chmod +x $ACTIVATE_SCRIPT

echo ""
echo "====================================================="
echo "Conda environment '$ENV_NAME' has been set up!"
echo "To activate the environment, run:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "To use the consistent character generator:"
echo "1. Start ComfyUI server with:"
echo "   cd ComfyUI && python main.py --listen 0.0.0.0"
echo "2. In a new terminal, activate the environment and run:"
echo "   python consistent_character.py path/to/subject_image.jpg"
echo "====================================================="