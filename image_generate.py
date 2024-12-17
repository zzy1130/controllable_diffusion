from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# Generate the image
generated_image = pipeline(prompt="A digital illustration of a cozy indoor scene, 8K, featuring a modern rocking chair with a stylish leather design. A fluffy gray and white cat is lying beneath the chair. The setting includes a soft, textured rug and a warm ambiance, with houseplants and a fireplace in the background",
                           negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, ugly, cat missing").images[0]

# Optionally, save the image
generated_image.save("/userhome/30/zyzhong2/controllable_diffusion/out/task1_gen.png")