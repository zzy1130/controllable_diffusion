from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# Generate the image
generated_image = pipeline("a digital photograph of a fluffy gray and white English shorthair cat, 4k, detailed, full-body.").images[0]

# Optionally, save the image
generated_image.save("/images/task1_gen.png")