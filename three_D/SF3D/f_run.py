import argparse
import os
from contextlib import nullcontext

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

def generate_toy(image_path):
    print(image_path)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    output_dir = "/userhome/30/zyzhong2/controllable_diffusion/three_D/stable_fast_3d/out"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = SF3D.from_pretrained(
        'stabilityai/stable-fast-3d',
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()

    # Process single image
    rembg_session = rembg.new_session()
    image = remove_background(Image.open(image_path).convert("RGBA"), rembg_session)
    image = resize_foreground(image, 0.85)
    

    image.save(os.path.join(output_dir, "input.png"))

    # Generate 3D model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16) if "cuda" in device else nullcontext():
        mesh, glob_dict = model.run_image(
            [image],  # Still needs to be a list
            bake_resolution=1024,
            # remesh=args.remesh_option,
            vertex_count=-1,
        )

    # Print memory usage
    if torch.cuda.is_available():
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    elif torch.backends.mps.is_available():
        print(f"Peak Memory: {torch.mps.driver_allocated_memory() / 1024 / 1024:.2f} MB")

    # Save output
    out_mesh_path = os.path.join(output_dir, "mesh.glb")
    mesh.export(out_mesh_path, include_normals=True)



if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    generate_toy()