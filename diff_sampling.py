import torch
import argparse
from diffusers import DiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDPMScheduler

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
parser.add_argument(
    "--num_steps", type=int, default=50, help="Number of sampling steps (default: 50)"
)
parser.add_argument(
    "--guidance_scale", type=float, default=7.5, help="Guidance scale (default: 7.5)"
)
parser.add_argument(
    "--scheduler", type=str, help="Scheduler choice"
)
args = parser.parse_args()
torch.manual_seed(42)

# Load the Stable Diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
# Set the scheduler
if args.scheduler == "DDIM":
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
elif args.scheduler == "DPM":
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
elif args.scheduler == "LMS":
    pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
elif args.scheduler == "Euler":
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
elif args.scheduler == "EulerAncestral":
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
elif args.scheduler == "DDPM":
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator(device="cuda").manual_seed(8)
# Example prompt (you can modify this as needed)
prompt = "A Chinese artwork of a cat with a box and a knife in his hand, white and orange breastplate, wearina torn clothes,full body mascot, inspired by Dong Kingman, bassist, brown pants, artist rendition, aliased, artistic rendition,instruments, block head, trading card, by Noami"
# Generate an image
image = pipeline(prompt, num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale).images[0]

# Save the generated image
image.save("/images/task2.png")