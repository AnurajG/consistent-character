#!/usr/bin/env python
# Standalone script for creating consistent character images in different poses

import os
import argparse
import json
import random
import time
import shutil
from PIL import Image, ExifTags
from typing import List, Dict, Optional, Set
import subprocess
import sys

# Constants
OUTPUT_DIR = "./outputs"
INPUT_DIR = "./inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
POSE_PATH = f"{INPUT_DIR}/poses"
MAX_HEADSHOTS = 14
MAX_POSES = 30
WORKFLOW_JSON = "workflow_api.json"
BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"

class ConsistentCharacter:
    def __init__(self, comfyui_address: str = "127.0.0.1:8188"):
        """Initialize the consistent character generator"""
        self.comfyui_address = comfyui_address
        self.setup_directories()
        
        # Import required modules only after checking if ComfyUI is installed
        try:
            import websocket
            import requests
            import uuid
            import urllib.request
        except ImportError:
            print("Required packages not found. Please install them with:")
            print("pip install websocket-client requests pillow")
            sys.exit(1)
            
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.safety_checker_enabled = True
        
        # Initialize available poses
        self.headshots = self.list_pose_filenames(type="headshot")
        self.poses = self.list_pose_filenames(type="pose")
        self.all = self.headshots + self.poses

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [OUTPUT_DIR, INPUT_DIR, POSE_PATH, COMFYUI_TEMP_OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)
            
    def check_comfyui_server(self) -> bool:
        """Check if ComfyUI server is running"""
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://{self.comfyui_address}/history/123"
            ) as response:
                return response.status == 200
        except Exception:
            return False
            
    def connect_to_comfyui(self):
        """Connect to the ComfyUI websocket server"""
        import websocket
        if not self.check_comfyui_server():
            print("ComfyUI server is not running. Please start it with:")
            print("cd ComfyUI && python main.py --listen 0.0.0.0")
            sys.exit(1)
            
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(f"ws://{self.comfyui_address}/ws?clientId={self.client_id}")
            print(f"Connected to ComfyUI server at {self.comfyui_address}")
        except Exception as e:
            print(f"Failed to connect to ComfyUI server: {str(e)}")
            sys.exit(1)
    
    def list_pose_filenames(self, type="headshot") -> List[Dict[str, str]]:
        """Get list of available pose filenames"""
        if type == "headshot":
            max_value = MAX_HEADSHOTS
            prefix = "headshot"
        elif type == "pose":
            max_value = MAX_POSES
            prefix = "pose"
        else:
            raise ValueError("Invalid type specified. Use 'headshot' or 'pose'.")

        return [
            {
                "kps": f"{POSE_PATH}/{prefix}_kps_{str(i).zfill(5)}_.png",
                "openpose": f"{POSE_PATH}/{prefix}_open_pose_{str(i).zfill(5)}_.png",
                "dwpose": f"{POSE_PATH}/{prefix}_dw_pose_{str(i).zfill(5)}_.png",
            }
            for i in range(1, max_value + 1)
        ]

    def get_poses(self, number_of_outputs: int, is_random: bool = True, type: str = "Half-body poses") -> List[Dict[str, str]]:
        """Get a list of poses to use for generation"""
        if type == "Headshot poses":
            return self.get_filenames(self.headshots, number_of_outputs, is_random)
        elif type == "Half-body poses":
            return self.get_filenames(self.poses, number_of_outputs, is_random)
        else:
            return self.get_filenames(self.all, number_of_outputs, is_random)

    def get_filenames(self, filenames: List[Dict[str, str]], length: int, use_random: bool = True) -> List[Dict[str, str]]:
        """Get a subset of filenames for generation"""
        if length > len(filenames):
            length = len(filenames)
            print(f"Using {length} as the max number of files.")

        files_copy = filenames.copy()
        if use_random:
            random.shuffle(files_copy)

        return files_copy[:length]

    def process_input_image(self, image_path: str, output_filename: str = "subject.png") -> str:
        """Process and save input image, handling orientation"""
        try:
            image = Image.open(image_path)

            # Handle image orientation
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation, do not rotate
                pass

            # Save processed image
            output_path = os.path.join(INPUT_DIR, output_filename)
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error processing input image: {str(e)}")
            sys.exit(1)
            
    def update_workflow(self, workflow: Dict, **kwargs):
        """Update nodes in the workflow JSON based on input parameters"""
        # Update positive prompt
        positive_prompt = workflow["9"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        # Update negative prompt
        negative_prompt = workflow["10"]["inputs"]
        negative_prompt["text"] = (
            f"(nsfw:2), nipple, nude, naked, {kwargs['negative_prompt']}, lowres, "
            "two people, child, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, "
            "jpeg artifacts, signature, watermark, username, blurry, multiple view, reference sheet"
        )

        # Update seed
        sampler = workflow["11"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        # Update batch size
        empty_latent_image = workflow["29"]["inputs"]
        empty_latent_image["batch_size"] = kwargs["number_of_images_per_pose"]

        # Update pose images
        kps_input_image = workflow["94"]["inputs"]
        kps_input_image["image"] = kwargs["pose"]["kps"]

        dwpose_input_image = workflow["95"]["inputs"]
        dwpose_input_image["image"] = kwargs["pose"]["dwpose"]
            
    def queue_prompt(self, prompt: Dict) -> str:
        """Queue a prompt for execution in ComfyUI"""
        import urllib.request
        import json
        
        try:
            # Send the prompt to ComfyUI
            p = {"prompt": prompt, "client_id": self.client_id}
            data = json.dumps(p).encode("utf-8")
            req = urllib.request.Request(
                f"http://{self.comfyui_address}/prompt?{self.client_id}", data=data
            )

            output = json.loads(urllib.request.urlopen(req).read())
            return output["prompt_id"]
        except Exception as e:
            print(f"ComfyUI error: {str(e)}")
            sys.exit(1)
            
    def wait_for_prompt_completion(self, workflow: Dict, prompt_id: str):
        """Wait for prompt execution to complete"""
        import json
        
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
                    elif data["prompt_id"] == prompt_id:
                        node = workflow.get(data["node"], {})
                        meta = node.get("_meta", {})
                        class_type = node.get("class_type", "Unknown")
                        print(
                            f"Processing node {data['node']}: {meta.get('title', 'Unknown')} ({class_type})"
                        )
            else:
                continue
                
    def get_history(self, prompt_id: str) -> Dict:
        """Get history of prompt execution"""
        import urllib.request
        import json
        
        with urllib.request.urlopen(
            f"http://{self.comfyui_address}/history/{prompt_id}"
        ) as response:
            output = json.loads(response.read())
            return output[prompt_id]["outputs"]
            
    def run_workflow(self, workflow: Dict):
        """Run a workflow and wait for completion"""
        print("Running workflow...")
        prompt_id = self.queue_prompt(workflow)
        self.wait_for_prompt_completion(workflow, prompt_id)
        output_json = self.get_history(prompt_id)
        print("Workflow completed")
        return output_json
        
    def get_output_files(self) -> List[str]:
        """Get list of output files sorted by modification time"""
        files = []
        for filename in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(path):
                files.append(path)
                
        # Sort by modification time (newest first)
        return sorted(files, key=os.path.getmtime, reverse=True)
        
    def generate(self, 
                 subject_image_path: str,
                 prompt: str = "A portrait photo of a person", 
                 negative_prompt: str = "",
                 pose_type: str = "Half-body poses",
                 number_of_outputs: int = 3,
                 number_of_images_per_pose: int = 1,
                 randomise_poses: bool = True,
                 seed: Optional[int] = None):
        """Generate consistent character images with different poses"""
        # Process input image
        self.process_input_image(subject_image_path)
        
        # Generate a random seed if not specified
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Using random seed: {seed}")
        
        # Get poses based on parameters
        poses = self.get_poses(number_of_outputs, randomise_poses, pose_type)
        if not poses:
            print("No poses available. Please check pose files.")
            return []
            
        # Connect to ComfyUI server
        self.connect_to_comfyui()
        
        # Load workflow JSON
        try:
            with open(WORKFLOW_JSON, "r") as file:
                workflow = json.loads(file.read())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading workflow JSON: {str(e)}")
            print(f"Make sure {WORKFLOW_JSON} exists in the current directory")
            return []
            
        # Clear output directory
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        output_images = []
        
        # Process each pose
        for i, pose in enumerate(poses):
            print(f"\nGenerating image {i+1}/{len(poses)}...")
            
            # Check if pose files exist
            if not all(os.path.exists(p) for p in pose.values()):
                print(f"Warning: Pose files missing for {pose}. Skipping.")
                continue
                
            # Update workflow with parameters
            self.update_workflow(
                workflow,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                type=pose_type,
                number_of_outputs=number_of_outputs,
                number_of_images_per_pose=number_of_images_per_pose,
                randomise_poses=randomise_poses,
                pose=pose,
            )
            
            # Run workflow
            self.run_workflow(workflow)
            
            # Get new output files
            new_files = self.get_output_files()
            if new_files:
                output_images.extend(new_files)
                print(f"Generated image saved to: {new_files[0]}")
            
        print(f"\nGeneration complete! {len(output_images)} images created.")
        return output_images

def download_weights(weights_to_download: Optional[List[str]] = None):
    """Download model weights specified in weights.json
    
    Args:
        weights_to_download: Optional list of specific weight names to download.
                            If None, downloads a default set of necessary weights.
    """
    try:
        # Import the weight manifest and downloader
        try:
            from weights_manifest import WeightsManifest
            from weights_downloader import WeightsDownloader
            weights_downloader = WeightsDownloader()
        except ImportError:
            print("Could not import weights_manifest or weights_downloader modules.")
            print("Ensure weights_manifest.py and weights_downloader.py are in the current directory.")
            return False
            
        if not os.path.exists("weights.json"):
            print("weights.json not found in the current directory.")
            return False
            
        # Default essential weights if none specified
        if weights_to_download is None:
            weights_to_download = [
                # Core SDXL models
                "sd_xl_base_1.0.safetensors",
                "sd_xl_refiner_1.0.safetensors",
                # ControlNet models needed for poses
                "control_v11p_sd15_openpose.pth",
                "control_v11p_sd15_openpose_fp16.safetensors",
                "OpenPoseXL2.safetensors",
                # Removing backgrounds
                "RMBG-1.4/model.pth"
            ]
            
        print(f"Downloading {len(weights_to_download)} model weights...")
        for weight in weights_to_download:
            try:
                weights_downloader.download_weights(weight)
            except Exception as e:
                print(f"Error downloading weight {weight}: {str(e)}")
                
        print("Weight downloading complete")
        return True
    except Exception as e:
        print(f"Error in download_weights: {str(e)}")
        return False

def download_pose_images():
    """Download pose images from the repository"""
    print("Downloading pose images...")
    try:
        # Check if curl is available
        subprocess.run(["curl", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create pose directory
        os.makedirs(POSE_PATH, exist_ok=True)
        
        # Download pose images archive
        subprocess.run([
            "curl", "-L", 
            "https://weights.replicate.delivery/default/fofr/character/pose_images.tar",
            "-o", "pose_images.tar"
        ], check=True)
        
        # Extract pose images
        subprocess.run(["tar", "-xf", "pose_images.tar", "-C", POSE_PATH], check=True)
        
        # Remove downloaded archive
        os.remove("pose_images.tar")
        print("Pose images downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading pose images: {str(e)}")
        print("Please download pose images manually and place them in the 'inputs/poses' directory")
        return False

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Generate consistent character images in different poses")
    parser.add_argument("image", nargs="?", help="Path to the subject image")
    parser.add_argument("--prompt", default="A portrait photo of a person", help="Positive prompt describing the character")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for things to avoid")
    parser.add_argument("--pose-type", choices=["Headshot poses", "Half-body poses", "Both headshots and half-body poses"], 
                       default="Half-body poses", help="Type of poses to use")
    parser.add_argument("--num-outputs", type=int, default=3, help="Number of images to generate")
    parser.add_argument("--images-per-pose", type=int, default=1, help="Number of variations per pose")
    parser.add_argument("--no-random", action="store_false", dest="randomise", help="Don't randomize pose selection")
    parser.add_argument("--seed", type=int, help="Random seed for generation")
    parser.add_argument("--comfyui-address", default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--download-poses", action="store_true", help="Download pose images")
    parser.add_argument("--download-weights", action="store_true", help="Download necessary model weights")
    parser.add_argument("--download-specific-weights", nargs="+", help="Download specific model weights")
    parser.add_argument("--setup", action="store_true", help="Setup mode: download all necessary resources")
    
    args = parser.parse_args()
    
    # Setup mode: download all necessary resources
    if args.setup:
        print("Setting up Consistent Character generator...")
        download_pose_images()
        download_weights()
        print("Setup complete!")
        return
        
    # Download weights if requested
    if args.download_weights:
        download_weights()
        
    # Download specific weights if requested
    if args.download_specific_weights:
        download_weights(args.download_specific_weights)
    
    # Download pose images if requested
    if args.download_poses:
        download_pose_images()
    
    # If no image provided and in download-only mode, exit
    if args.image is None:
        if args.download_weights or args.download_poses or args.download_specific_weights:
            return
        else:
            parser.error("the following arguments are required: image")
    
    # Check if workflow JSON exists
    if not os.path.exists(WORKFLOW_JSON):
        print(f"Error: {WORKFLOW_JSON} not found")
        print("Please ensure the workflow API JSON file is in the current directory")
        return
    
    # Initialize generator
    generator = ConsistentCharacter(comfyui_address=args.comfyui_address)
    
    # Generate images
    start_time = time.time()
    output_images = generator.generate(
        subject_image_path=args.image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        pose_type=args.pose_type,
        number_of_outputs=args.num_outputs,
        number_of_images_per_pose=args.images_per_pose,
        randomise_poses=args.randomise,
        seed=args.seed
    )
    elapsed_time = time.time() - start_time
    
    if output_images:
        print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
        print(f"Output images saved to {os.path.abspath(OUTPUT_DIR)}")
    else:
        print("No images were generated. Please check the error messages above.")

if __name__ == "__main__":
    main()