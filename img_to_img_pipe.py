from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
from diffusers.utils import load_image
import torch
from PIL import Image

pipe_prior = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
)
pipe_prior.to("cuda")

prompt = "Make the cat in the image become a small dog"
image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

pipe = KandinskyImg2ImgPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
)
pipe.to("cuda")

# init_image = load_image(
#     ""
# )
# Specify the path to your local image
image_path = "/userhome/30/zyzhong2/diffussion/32381734091359_.pic_hd.jpg"  # Update with your image path

# Load the image
init_image = Image.open(image_path)


image = pipe(
    prompt,
    image=init_image,
    image_embeds=image_emb,
    negative_image_embeds=zero_image_emb,
    height=768,
    width=768,
    num_inference_steps=100,
    strength=0.2,
).images

image[0].save("/images/img2img_output.png")