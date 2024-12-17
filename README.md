# controllable_diffusion
## Contact ##

- For any question and enquiry, please feel free to reach out to Zhong Zhiyi(zhongzy@connect.hku.hk)

## Overview

**Hardware & System Requirements**

- Hardware (HKU GPU Farm)
  - `lora_generate` is attempted with NVIDIA GeForce RTX 3080 with 24G RAM
  -  Other code can be run with NVIDIA GeForce RTX 1080 Ti
- System
  - Linux

## Commands to run code

### Task 1: Image Generation with Stable Diffusion

```
python image_generate.py
```

This program will generate an image based on the prompt. You can change prompt in the program.


### Task 2: Different Sampling Strategies
Example implementation:
```
python diff_sampling.py --num_steps 50 --guidance_scale 7 --scheduler DDIM
```
--num_steps is the number of sampling steps, --guidance_scale defines the guidance scale, and --scheduler defines the scheduler (you can choose from DDIM, DPM, LMS, Euler, DDPM)

### Task 3 and 4: ControlNet with Different Control Signals
Single Control:
```
python controlnet.py --condition canny
```
--condition specifies the condition choice. You can choose among canny, depth, normal_map, pose, scribble, hed, and pix2pix.
Multiple controls:
```
python controlnet.py --multiple_condition canny+pose
```
--multiple_control can accept at most two conditions separated by '+'
What should be noted is that you should specify the image file path for control signals in the code.

### Task 5: Image-text-image Generation with LoRA
Example:
```
python lora_generate.py --image_path ./images/IMG_47CF507AEBD7-1.png
```
--image_path specifies the path to the original image. When the path is passed into image_path, the program will go through all steps automatically to generate the final output.
-- --

After this, all have been done. Thanks for your patience!
