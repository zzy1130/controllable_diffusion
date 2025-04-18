---
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/blob/main/LICENSE.txt
language:
- en
---
# HunyuanDiT LoRA

Language: **English**

## Instructions

 The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2).

 We provide two types of trained LoRA weights for you to test.
 
 Then download the model using the following commands:
 
```bash
cd HunyuanDiT
# Use the huggingface-cli tool to download the model.
huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora

# Quick start
python sample_t2i.py --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain --infer-mode fa
```


## Training

We provide three types of weights for fine-tuning LoRA, `ema`, `module` and `distill`, and you can choose according to the actual effect. By default, we use `ema` weights. 

Here is an example for LoRA with HunYuanDiT v1.2, we load the `distill` weights into the main model and perform LoRA fine-tuning through the `resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt` setting. 

If multiple resolution are used, you need to add the `--multireso` and `--reso-step 64 ` parameter. 

If you want to train LoRA with HunYuanDiT v1.1, you could add `--use-style-cond`, `--size-cond 1024 1024` and `--beta-end 0.03`.

```bash
model='DiT-g/2'                                                   # model type
task_flag="lora_porcelain_ema_rank64"                             # task flag
resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt     # resume checkpoint
index_file=dataset/porcelain/jsons/porcelain.json                 # the selected data indices
results_dir=./log_EXP                                             # save root for results
batch_size=1                                                      # training batch size
image_size=1024                                                   # training image resolution
grad_accu_steps=2                                                 # gradient accumulation steps
warmup_num_steps=0                                                # warm-up steps
lr=0.0001                                                         # learning rate
ckpt_every=100                                                    # create a ckpt every a few steps.
ckpt_latest_every=2000                                            # create a ckpt named `latest.pt` every a few steps.
rank=64                                                           # rank of lora
max_training_steps=2000                                           # Maximum training iteration steps

PYTHONPATH=./ deepspeed hydit/train_deepspeed.py \
    --task-flag ${task_flag} \
    --model ${model} \
    --training-parts lora \
    --rank ${rank} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --lr ${lr} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --uncond-p-t5 0 \
    --index-file ${index_file} \
    --random-flip \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --max-training-steps ${max_training_steps}\
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --qk-norm \
    --rope-img base512 \
    --rope-real \
    "$@"
```

Recommended parameter settings

|     Parameter     |  Description  |          Recommended Parameter Value                               | Note|
|:---------------:|:---------:|:---------------------------------------------------:|:--:|
|   `--batch_size` |    Training batch size    |        1        | Depends on GPU memory| 
|   `--grad-accu-steps` |    Size of gradient accumulation    |       2        | - |
|   `--rank` |    Rank of lora    |       64        | Choosing from 8-128|
|   `--max-training-steps` |    Training steps  |       2000        | Depend on training data size, for reference apply 2000 steps on 100 images|
|   `--lr` |    Learning rate  |        0.0001        | - |



## Inference

### Using Gradio

Make sure you have activated the conda environment before running the following command.

> ⚠️ Important Reminder:  
> We recommend not using prompt enhance, as it may lead to the disappearance of style words. 

```shell
# jade style

# Using Flash Attention for acceleration.
python app/hydit_app.py --infer-mode fa --load-key ema --lora-ckpt ./ckpts/t2i/lora/jade

# You can disable the enhancement model if the GPU memory is insufficient.
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 
python app/hydit_app.py --infer-mode fa --no-enhance --load-key ema --lora-ckpt  ./ckpts/t2i/lora/jade 

# Start with English UI
python app/hydit_app.py --infer-mode fa --lang en --load-key ema --lora-ckpt ./ckpts/t2i/lora/jade 

# porcelain style 

# Using Flash Attention for acceleration.
python app/hydit_app.py --infer-mode fa --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain

# You can disable the enhancement model if the GPU memory is insufficient.
# The enhancement will be unavailable until you restart the app without the `--no-enhance` flag. 
python app/hydit_app.py --infer-mode fa --no-enhance --load-key ema --lora-ckpt  ./ckpts/t2i/lora/porcelain

# Start with English UI
python app/hydit_app.py --infer-mode fa --lang en --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain
```


### Using Command Line

We provide several commands to quick start: 

```shell
# jade style

# Prompt Enhancement + Text-to-Image. Torch mode
python sample_t2i.py --infer-mode fa --prompt "玉石绘画风格，一只猫在追蝴蝶" --load-key ema --lora-ckpt ./ckpts/t2i/lora/jade

# Only Text-to-Image. Torch mode
python sample_t2i.py --infer-mode fa --prompt "玉石绘画风格，一只猫在追蝴蝶" --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/jade

# Generate an image with other image sizes.
python sample_t2i.py --infer-mode fa --prompt "玉石绘画风格，一只猫在追蝴蝶" --image-size 1280 768 --load-key ema --lora-ckpt ./ckpts/t2i/lora/jade

# porcelain style 

# Prompt Enhancement + Text-to-Image. Torch mode
python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶" --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain

# Only Text-to-Image. Torch mode
python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶" --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain

# Generate an image with other image sizes.
python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只猫在追蝴蝶"  --image-size 1280 768 --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain 
```


Regarding how to use the LoRA weights we trained in diffusion, we provide the following script. To ensure compatibility with the diffuser, some modifications are made, which means that LoRA cannot be directly loaded. 

```python
import torch
from diffusers import HunyuanDiTPipeline

num_layers = 40
def load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale):
    for i in range(num_layers):
        Wqkv = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_A.weight"]) 
        q, k, v = torch.chunk(Wqkv, 3, dim=0)
        transformer_state_dict[f"blocks.{i}.attn1.to_q.weight"] += lora_scale * q
        transformer_state_dict[f"blocks.{i}.attn1.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn1.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.to_out.0.weight"] += lora_scale * out_proj

        q_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_q.weight"] += lora_scale * q_proj

        kv_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_A.weight"])
        k, v = torch.chunk(kv_proj, 2, dim=0)
        transformer_state_dict[f"blocks.{i}.attn2.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn2.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn2.to_out.0.weight"] += lora_scale * out_proj
    
    q_proj = torch.matmul(lora_state_dict["pooler.q_proj.lora_B.weight"], lora_state_dict["pooler.q_proj.lora_A.weight"])
    transformer_state_dict["time_extra_emb.pooler.q_proj.weight"] += lora_scale * q_proj
    
    return transformer_state_dict

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

from safetensors import safe_open

lora_state_dict = {}
with safe_open("./ckpts/t2i/lora/jade/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        lora_state_dict[k[17:]] = f.get_tensor(k) # remove 'basemodel.model'

transformer_state_dict = pipe.transformer.state_dict()
transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=1.0)
pipe.transformer.load_state_dict(transformer_state_dict)

prompt = "玉石绘画风格，一只猫在追蝴蝶"
image = pipe(
    prompt, 
    num_inference_steps=100,
    guidance_scale=6.0, 
).images[0]
image.save('img.png')
```

More example prompts can be found in [example_prompts.txt](example_prompts.txt)
