from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector, HEDdetector, PidiNetDetector
import torch
import cv2
from PIL import Image
import numpy as np
import argparse
from transformers import pipeline
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description="Generate images using Controlnet")
parser.add_argument(
    "--condition", type=str, default='', help="Condition choice"
)
parser.add_argument(
    "--multiple_condition", type=str, help="Multiple Condition choice"
)
args = parser.parse_args()

control_images = []
controlnets = []
target_size = (512, 512) 
if args.multiple_condition:
    conditions = args.multiple_condition.split('+')
else:
    conditions = [args.condition]
for condition in conditions:
    if condition == 'canny':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/controllable_diffusion/images/beauty.png"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        # zero_start = image.shape[1] // 4
        # zero_end = zero_start + image.shape[1] // 2
        # image[:, zero_start:zero_end] = 0

        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        # image = image.resize(target_size)
        image.save('/userhome/30/zyzhong2/diffussion/control_image/canny_control.png')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        control_images.append(image)
        controlnets.append(controlnet)
    elif condition == 'pose':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/diffussion/images/man_pose.png"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        checkpoint = "lllyasviel/control_v11p_sd15_openpose"
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        image = processor(image, hand_and_face=True, preprocessor=None)
        image.save("/userhome/30/zyzhong2/diffussion/control_image/pose_control.png")
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        # image = image.resize(target_size)
        control_images.append(image)
        controlnets.append(controlnet)
    elif condition == 'depth':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/controllable_diffusion/images/beauty.png"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        depth_estimator = pipeline('depth-estimation')
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        image.save('/userhome/30/zyzhong2/controllable_diffusion/control_image/depth_control.png')
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )
        # image = image.resize(target_size)
        control_images.append(image)
        controlnets.append(controlnet)
    elif condition == 'normal_map':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/controllable_diffusion/images/crowd_image.png"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

        image = depth_estimator(image)['predicted_depth'][0]

        image = image.numpy()

        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.1

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
        )
        image.save('/userhome/30/zyzhong2/controllable_diffusion/control_image/normal_map_control.png')
        # image = image.resize(target_size)
        control_images.append(image)
        controlnets.append(controlnet)
    elif condition == 'scribble':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/diffussion/images/keep_down.png"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        checkpoint = "lllyasviel/control_v11p_sd15_scribble"
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(image, scribble=True)
        control_image.save("/userhome/30/zyzhong2/diffussion/images/scribble_control.png")
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        # control_image = control_image.resize(target_size)
        control_images.append(control_image)
        controlnets.append(controlnet)
    elif condition == 'hed':
        checkpoint = "lllyasviel/control_v11p_sd15_softedge"
        image = Image.open('/userhome/30/zyzhong2/controllable_diffusion/images/beauty.png')
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(image, safe=True)
        control_image.save("/userhome/30/zyzhong2/controllable_diffusion/control_image/control_hed.png")
        # control_image = control_image.resize(target_size)
        control_images.append(control_image)
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        controlnets.append(controlnet)
    elif condition == 'pix2pix':
        # Specify the path to your local image
        image_path = "/userhome/30/zyzhong2/diffussion/images/IMG_47CF507AEBD7-1.jpeg"  # Update with your image path
        # Load the image
        image = Image.open(image_path)
        checkpoint = "lllyasviel/control_v11e_sd15_ip2p"
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        # image = image.resize(target_size)
        control_images.append(image)
        controlnets.append(controlnet)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnets, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(42)
# Adjust the control strength and guidance scale
# control_strength = 0.1  # Lower this value to reduce control from the Canny edges
# guidance_scale = 1  # Adjust as needed
if condition[0] != 'pix2pix':
    prompt = "A portrait of a woman in DisyneyLand, high appearance level, gorgeous, 8k, detailed"
    inference_step = 20
else:
    prompt = "Make it become a Chinese artwork of a cat with a box in his hand, white and orange breastplate, wearina torn clothes,full body mascot, inspired by Dong Kingman, bassist, brown pants, artist rendition, aliased, artistic rendition,instruments, block head, trading card, by Noami"
    inference_step = 30
base_size = control_images[0].size

# Resize all other images in the list
resized_images = []
for img in control_images:
    # Resize the image to match the base size
    resized_img = img.resize(base_size)
    resized_images.append(resized_img)

# Generate the image
out_image = pipe(
    prompt, 
    num_inference_steps=inference_step, 
    image=resized_images,
    controlnet = controlnets,
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, ugly",
    # control_strength=control_strength,  # Adjust control strength
    # guidance_scale=guidance_scale  # Adjust guidance scale
).images[0]

if not args.multiple_condition:
    out_image.save(f"/userhome/30/zyzhong2/controllable_diffusion/out/controlnet_{conditions[0]}.png")
else:
    out_image.save(f"/userhome/30/zyzhong2/controllable_diffusion/out/controlnet_{args.multiple_condition}.png")
