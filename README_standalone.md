# Consistent Character Standalone

Create images of a given character in different poses without needing the Replicate platform.

## Overview

This standalone version of the Consistent Character generator allows you to:

1. Generate multiple images of the same person in different poses
2. Maintain consistent appearance across all generated images
3. Run locally with minimal setup

## Prerequisites

- Python 3.10
- ComfyUI installation
- CUDA-capable GPU (recommended)

## Setup

### 1. Clone the ComfyUI repository

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
```

### 2. Set up the environment

Option A: With Conda (recommended):

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate consistent_character

# Or use the setup script
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

Option B: With pip:

```bash
pip install -r requirements.txt
```

### 3. Download required pose images

```bash
python consistent_character.py --download-poses
```

### 4. Start the ComfyUI server

```bash
cd ComfyUI
python main.py --listen 0.0.0.0
```

## Usage

Basic usage:

```bash
python consistent_character.py path/to/subject_image.jpg
```

With additional options:

```bash
python consistent_character.py path/to/subject_image.jpg \
  --prompt "A portrait photo of a woman with short curly hair" \
  --num-outputs 5 \
  --pose-type "Half-body poses"
```

### Command Line Options

- `image`: Path to the subject image (required)
- `--prompt`: Description of the character (default: "A portrait photo of a person")
- `--negative-prompt`: Things to avoid in the generated images
- `--pose-type`: Type of poses to use ("Headshot poses", "Half-body poses", or "Both headshots and half-body poses")
- `--num-outputs`: Number of images to generate (default: 3)
- `--images-per-pose`: Number of variations per pose (default: 1)
- `--no-random`: Don't randomize pose selection
- `--seed`: Random seed for reproducible results
- `--comfyui-address`: ComfyUI server address (default: 127.0.0.1:8188)
- `--download-poses`: Download pose reference images

## Examples

Generate three half-body poses:
```bash
python consistent_character.py my_portrait.jpg --prompt "A photo of a man in a suit"
```

Generate headshot poses:
```bash
python consistent_character.py my_portrait.jpg --pose-type "Headshot poses" --num-outputs 5
```

Use a specific seed for reproducibility:
```bash
python consistent_character.py my_portrait.jpg --seed 42
```

## Troubleshooting

- Make sure ComfyUI server is running before executing the script
- Verify that the workflow_api.json file is in the current directory
- Check that pose images are downloaded and available in the inputs/poses directory

## Credits

Based on the [Consistent Character](https://github.com/fofr/cog-consistent-character) project.