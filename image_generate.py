from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# Generate the image
generated_image = pipeline(prompt="a Chinese-style painting of a cat holding a piece of paper, wrenches, style as nendoroid, full body portrait, maintenance photo, wiry, brown paper, capacitors, app design, father figure image, summer clothes, square nose, mechanised, of a full body, ehime, full body painting, box",
                           negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, ugly, cat missing").images[0]

# Optionally, save the image
generated_image.save("/userhome/30/zyzhong2/controllable_diffusion/out/task1_gen.png")