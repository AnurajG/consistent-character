# Setting up ComfyUI for Consistent Character Generator

This guide explains how to set up ComfyUI for use with the consistent_character.py script.

## Automatic Setup (Recommended)

The simplest way to set up ComfyUI is to use the provided setup script:

```bash
# Make the script executable
chmod +x setup_comfyui.sh

# Run the setup script
./setup_comfyui.sh
```

The script will:
1. Clone the ComfyUI repository
2. Create a Python virtual environment
3. Install all required dependencies
4. Download pose images
5. Create start scripts for ComfyUI and the generator

After running the setup script, you can:
1. Start ComfyUI server: `./start_comfyui.sh`
2. Run the generator: `./run_generator.sh path/to/your/image.jpg`

## Manual Setup

If you prefer to set up manually, follow these steps:

### 1. Clone ComfyUI Repository

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Install ComfyUI dependencies
pip install -r ComfyUI/requirements.txt

# Install additional dependencies
pip install pillow opencv-python numpy websocket-client requests
```

### 4. Download Pose Images

```bash
# Create directories
mkdir -p ./inputs/poses
mkdir -p ./outputs

# Download pose images
curl -L "https://weights.replicate.delivery/default/fofr/character/pose_images.tar" -o pose_images.tar
# OR
wget "https://weights.replicate.delivery/default/fofr/character/pose_images.tar" -O pose_images.tar

# Extract pose images
tar -xf pose_images.tar -C ./inputs/poses
rm pose_images.tar
```

### 5. Start ComfyUI Server

```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --output-directory ../outputs --input-directory ../inputs
```

### 6. Run the Generator

In a new terminal:

```bash
# Activate the virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Run the generator
python consistent_character.py path/to/your/image.jpg
```

## Troubleshooting

### ComfyUI won't start

- Check if you have the required Python version (3.10+ recommended)
- Ensure you've installed all dependencies
- Check for any error messages during startup

### Generator can't connect to ComfyUI

- Make sure ComfyUI server is running
- Verify the server address in consistent_character.py (default: 127.0.0.1:8188)
- Check if your firewall is blocking the connection

### Missing pose images

If pose images aren't downloaded automatically:
1. Download pose_images.tar manually from:
   https://weights.replicate.delivery/default/fofr/character/pose_images.tar
2. Extract it to the ./inputs/poses directory

### Model weight issues

If you encounter model weight issues:
1. ComfyUI will download some models automatically
2. For custom models, check the ComfyUI documentation for details on model placement