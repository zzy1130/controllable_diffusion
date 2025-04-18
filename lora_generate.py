import torch
from diffusers import DiffusionPipeline, LCMScheduler, ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline
import matplotlib.pyplot as plt
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, DPTImageProcessor
from controlnet_aux import OpenposeDetector, HEDdetector, PidiNetDetector
from PIL import Image
from transformers import pipeline
import numpy as np
from clip_inter import ImageInterrogator
import argparse
import cv2


parser = argparse.ArgumentParser(description="Generate images using Controlnet")
parser.add_argument(
    "--image_path", type=str, default='/userhome/30/zyzhong2/controllable_diffusion/images/smelly_cat.png', help="Condition choice"
)
parser.add_argument(
    "--condition", type=str, default='depth', help="Condition choice"
)
class ImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0", device="cuda",method='canny'):
        if method == 'soft_edge':
            checkpoint = "lllyasviel/control_v11p_sd15_softedge"
            controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        elif method == 'canny':
            checkpoint = "diffusers/controlnet-canny-sdxl-1.0"
            controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        elif method == 'depth':
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
            )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        # Initialize the diffusion pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_name,
            vae=vae,
            variant="fp16",
            torch_dtype=torch.float16,
            controlnet=controlnet
        ).to(device)

        # Set the scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # Load LoRAs
        self.load_lora_weights()

    def load_lora_weights(self):
        # Load the LoRA weights
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
        self.pipe.load_lora_weights("ming-yang/sdxl_chinese_ink_lora", adapter_name="Chinese Ink")

    def load_control_image(self, image_path, method='depth'):
        # Load the control image based on the specified method
        control_image = Image.open(image_path)

        if method == 'depth':
            depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            image = Image.open(image_path).convert("RGB")  # Ensure RGB format
            inputs = feature_extractor(images=image, return_tensors="pt")
            
            # Move to GPU and add batch dimension if needed
            pixel_values = inputs.pixel_values.to("cuda")
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension if missing
            
            # Generate depth map
            with torch.no_grad(), torch.autocast("cuda"):
                depth_map = depth_estimator(pixel_values).predicted_depth
            
            # Process depth map
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),  # Add channel dimension
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            
            # Normalize
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Convert to RGB format
            image = torch.cat([depth_map] * 3, dim=1)  # Repeat depth map across channels
            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]  # Convert to HWC format
            image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
            
            # Save and return
            image.save("/userhome/30/zyzhong2/controllable_diffusion/control_image/control_depth.png")
            return image
        elif method == 'canny':
            control_image = np.array(control_image)
            control_image = cv2.Canny(control_image, 100, 200)
            control_image = control_image[:, :, None]
            control_image = np.concatenate([control_image, control_image, control_image], axis=2)
            control_image = Image.fromarray(control_image)
            control_image.save("/userhome/30/zyzhong2/controllable_diffusion/control_image/control_canny.png")
            return control_image
        elif method == 'soft_edge':
            processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
            processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
            control_image = processor(control_image, safe=True)
            control_image.save("/userhome/30/zyzhong2/controllable_diffusion/control_image/control_hed.png")
            return control_image
        # Additional methods (e.g., 'pose', 'scribble') can be added here as needed

    def generate_image(self, prompt, control_image, method='soft_edge'):
        # Combine LoRAs
        self.pipe.set_adapters(["lcm", "Chinese Ink"], adapter_weights=[1.0, 0.8])
        # Generate the image
        generator = torch.manual_seed(1)
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, ugly, letter in the image"
        images = self.pipe(prompt, num_inference_steps=10, generator=generator, negative_prompt=negative_prompt, image=control_image, controlnet_conditioning_scale=0.9).images[0]
        
        return images

# Example usage
if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(42)
    method = args.condition
    image_path = args.image_path
    interrogator = ImageInterrogator()
    interrogator.load_image(image_path)
    prompt = interrogator.interrogate()
    prompt = "Chinese Ink, " + prompt + ", 8k"
    print(prompt)
    generator = ImageGenerator(method=method)
    control_image = generator.load_control_image(image_path, method=method)
    output_image = generator.generate_image(prompt, control_image, method=method)
    output_image.save('/userhome/30/zyzhong2/controllable_diffusion/out/lora_output.png')