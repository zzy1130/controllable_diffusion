import torch
from diffusers import DiffusionPipeline, LCMScheduler, ControlNetModel
import matplotlib.pyplot as plt
from controlnet_aux import OpenposeDetector, HEDdetector
from PIL import Image
from transformers import pipeline
import numpy as np
from clip_inter import ImageInterrogator

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
#                                          variant="fp16",
#                                          torch_dtype=torch.float16
#                                          ).to("cuda")
# # set scheduler
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # load LoRAs
# pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
# pipe.load_lora_weights("ming-yang/sdxl_chinese_ink_lora", adapter_name="Chinese Ink")

# image_path = "/userhome/30/zyzhong2/diffussion/images/IMG_47CF507AEBD7-1.jpeg"  # Update with your image path
# # Load the image
# control_image = Image.open(image_path)
# #pose
# # processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
# # control_image = processor(control_image, hand_and_face=True, preprocessor=None)
# # control_image.save("/userhome/30/zyzhong2/diffussion/images/pose_control.png")
# # # Load ControlNet model
# # controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose").to("cuda")
# #scribble
# # checkpoint = "lllyasviel/control_v11p_sd15_scribble"
# # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
# # control_image = processor(control_image, scribble=True)
# # control_image.save("/userhome/30/zyzhong2/diffussion/images/scribble_control.png")
# # controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
# #depth
# depth_estimator = pipeline('depth-estimation')
# control_image = depth_estimator(control_image)['depth']
# control_image = np.array(control_image)
# control_image = control_image[:, :, None]
# control_image = np.concatenate([control_image, control_image, control_image], axis=2)
# control_image = Image.fromarray(control_image)
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
# )
# # Combine LoRAs
# pipe.set_adapters(["lcm", "Chinese Ink"], adapter_weights=[1.0, 0.8])

# prompt = "Chinese Ink, A cat with a box and a knife in his hand, white and orange breastplate, wearina torn clothes,full body mascot, inspired by Dong Kingman, bassist, brown pants, artist rendition, aliased, artistic rendition,instruments, block head, trading card, by Noami, a character portrait, 8k"
# generator = torch.manual_seed(1)
# images = pipe(prompt, num_inference_steps=20, generator=generator, controlnet=controlnet, image=control_image).images[0]
# images.save('lora_output.png')

class ImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0", device="cuda"):
        # Initialize the diffusion pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            variant="fp16",
            torch_dtype=torch.float16
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
            depth_estimator = pipeline('depth-estimation')
            control_image = depth_estimator(control_image)['depth']
            control_image = np.array(control_image)
            control_image = control_image[:, :, None]
            control_image = np.concatenate([control_image, control_image, control_image], axis=2)
            control_image = Image.fromarray(control_image)
            return control_image

        # Additional methods (e.g., 'pose', 'scribble') can be added here as needed

    def generate_image(self, prompt, control_image):
        # Load the ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )

        # Combine LoRAs
        self.pipe.set_adapters(["lcm", "Chinese Ink"], adapter_weights=[1.0, 0.8])

        # Generate the image
        generator = torch.manual_seed(1)
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, ugly"
        images = self.pipe(prompt, num_inference_steps=20, generator=generator, controlnet=controlnet, negative_prompt=negative_prompt, image=control_image).images[0]
        
        return images

# Example usage
if __name__ == "__main__":
    image_path = "/userhome/30/zyzhong2/controllable_diffussion/images/IMG_47CF507AEBD7-1.png"
    interrogator = ImageInterrogator()
    interrogator.load_image(image_path)
    prompt = interrogator.interrogate()
    print(prompt)
    # prompt = "Chinese Ink, A cat with a box and a knife in his hand, white and orange breastplate, wearing torn clothes, full body mascot, inspired by Dong Kingman, bassist, brown pants, artist rendition, aliased, artistic rendition, instruments, block head, trading card, by Noami, a character portrait, 8k"
    generator = ImageGenerator()
    control_image = generator.load_control_image(image_path, method='depth')
    output_image = generator.generate_image(prompt, control_image)
    output_image.save('/userhome/30/zyzhong2/controllable_diffussion/out/lora_output.png')