#!/bin/bash
# Script to set up ComfyUI for the Consistent Character Generator

# Exit on error
set -e

echo "Setting up ComfyUI for Consistent Character Generator..."

# Check if git is installed
if ! command -v git &> /dev/null; then
  echo "Error: git is not installed. Please install git first."
  exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo "Error: Python 3 is not installed. Please install Python 3 first."
  exit 1
fi

# Clone ComfyUI repository if it doesn't exist
if [ ! -d "ComfyUI" ]; then
  echo "Cloning ComfyUI repository..."
  git clone https://github.com/comfyanonymous/ComfyUI.git
else
  echo "ComfyUI directory already exists. Updating..."
  cd ComfyUI
  git pull
  cd ..
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies for ComfyUI
echo "Installing dependencies for ComfyUI..."
pip install torch torchvision torchaudio
pip install -r ComfyUI/requirements.txt

# Install dependencies for consistent_character.py
echo "Installing dependencies for consistent_character.py..."
pip install pillow opencv-python numpy websocket-client requests

# Create required directories
mkdir -p ./inputs/poses
mkdir -p ./outputs

# Download pose images
echo "Downloading pose images..."
if command -v curl &> /dev/null; then
  echo "Using curl to download pose images..."
  curl -L "https://weights.replicate.delivery/default/fofr/character/pose_images.tar" -o pose_images.tar
elif command -v wget &> /dev/null; then
  echo "Using wget to download pose images..."
  wget "https://weights.replicate.delivery/default/fofr/character/pose_images.tar" -O pose_images.tar
else
  echo "Neither curl nor wget is installed. Cannot download pose images."
  echo "Please download pose images manually from:"
  echo "https://weights.replicate.delivery/default/fofr/character/pose_images.tar"
  echo "Then extract them to ./inputs/poses/"
fi

# Extract pose images if downloaded
if [ -f "pose_images.tar" ]; then
  echo "Extracting pose images..."
  tar -xf pose_images.tar -C ./inputs/poses
  rm pose_images.tar
  echo "Pose images extracted to ./inputs/poses/"
fi

# Copy workflow_api.json to current directory if needed
if [ ! -f "workflow_api.json" ] && [ -f "ComfyUI/workflow_api.json" ]; then
  echo "Copying workflow_api.json to current directory..."
  cp ComfyUI/workflow_api.json .
fi

# Create a start script for ComfyUI
cat > start_comfyui.sh << 'EOL'
#!/bin/bash
# Script to start ComfyUI server

# Activate virtual environment
source venv/bin/activate

# Start ComfyUI server
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --output-directory ../outputs --input-directory ../inputs

# Deactivate virtual environment when done
deactivate
EOL

# Make the start script executable
chmod +x start_comfyui.sh

# Create a script to run consistent_character.py
cat > run_generator.sh << 'EOL'
#!/bin/bash
# Script to run consistent_character.py

# Activate virtual environment
source venv/bin/activate

# Run consistent_character.py with arguments
python consistent_character.py "$@"

# Deactivate virtual environment when done
deactivate
EOL

# Make the run script executable
chmod +x run_generator.sh

echo ""
echo "====================================================="
echo "ComfyUI setup complete!"
echo ""
echo "To start ComfyUI server, run:"
echo "  ./start_comfyui.sh"
echo ""
echo "After starting ComfyUI server, in a new terminal, run:"
echo "  ./run_generator.sh path/to/your/image.jpg"
echo ""
echo "Additional options:"
echo "  --prompt \"Your prompt here\""
echo "  --num-outputs 5"
echo "  --pose-type \"Half-body poses\""
echo "====================================================="

# Deactivate virtual environment
deactivate