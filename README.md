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

## Commands to run code/>

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
num_steps is the number of sampling steps, guidance_scale defines the guidance scale, and scheduler defines the scheduler (you can choose from DDIM, DPM, LMS, Euler, DDPM)

### Task 3 and 4: ControlNet with Different Control Signals
-- --

After this, all have been done. Thanks for your patience!
