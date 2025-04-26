from flask import Flask, request, jsonify
from PIL import Image
import base64
import numpy as np
import io
import os
import torch
from werkzeug.utils import secure_filename
import torch
from diffusers import DiffusionPipeline, LCMScheduler, HunyuanDiTPipeline, AutoPipelineForText2Image
from PIL import Image
from transformers import pipeline
import numpy as np
from clip_inter import ImageInterrogator
from lora_generate import ImageGenerator
from diffusers import AutoPipelineForText2Image
from flask_cors import CORS
from blue_and_white import load_hunyuan_dit_lora
from safetensors import safe_open
import json
import requests
from three_D.adjust import gen_chop
from three_D.stable_fast_3d.f_run import generate_toy

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def story_to_bnw(prompt):
    pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
    pipe.to("cuda")

    from safetensors import safe_open

    lora_state_dict = {}
    with safe_open("/userhome/30/zyzhong2/controllable_diffusion/HunyuanDiT/ckpts/t2i/lora/jade/adapter_model.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            lora_state_dict[k[17:]] = f.get_tensor(k) # remove 'basemodel.model'

    transformer_state_dict = pipe.transformer.state_dict()
    transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=1.0)
    pipe.transformer.load_state_dict(transformer_state_dict)

    prompt = "Porcelain style, "+prompt
    image = pipe(
        prompt, 
        num_inference_steps=50,
        guidance_scale=7, 
    ).images[0]
    image.save('/userhome/30/zyzhong2/controllable_diffusion/out2front/bnw.png')
    return image
   
def story_to_disney(prompt):
    url =  "https://stablediffusionapi.com/api/v3/dreambooth"  
    print('original: ', prompt)
    prompt = "ultra realistic close up portrait (("+prompt+")), blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city, Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K"

    payload = json.dumps({  
    "key":  "",  
    "model_id":  "disney-pixar-cartoon",  
    "prompt":  prompt, 
    "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
    "width":  "512",  
    "height":  "512",  
    "samples":  "1",  
    "num_inference_steps":  "30",  
    "safety_checker":  "no",  
    "enhance_prompt":  "yes",  
    "seed":  None,  
    "guidance_scale":  7.5,  
    "multi_lingual":  "no",  
    "panorama":  "no",  
    "self_attention":  "no",  
    "upscale":  "no",  
    "embeddings":  "embeddings_model_id",  
    "lora":  "lora_model_id",  
    "webhook":  None,  
    "track_id":  None  
    })  
    
    headers =  {  
    'Content-Type':  'application/json'  
    }  
    
    response = requests.request("POST", url, headers=headers, data=payload) 
    response_dict = json.loads(response.text)
    print(response_dict)
    return response_dict['output'][0]

def story_to_ink(prompt):
    print(prompt)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         variant="fp16",
                                         torch_dtype=torch.float16
                                         ).to("cuda")
    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    prompt = "Chinese-ink artwork" + prompt + ", "
    # load LoRAs
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    pipe.load_lora_weights("ming-yang/sdxl_chinese_ink_lora", adapter_name="Chinese Ink")

    # Combine LoRAs
    pipe.set_adapters(["lcm", "Chinese Ink"], adapter_weights=[1.0, 0.8])

    generator = torch.manual_seed(1)
    image = pipe(prompt, num_inference_steps=10, guidance_scale=5, generator=generator).images[0]
    image.save('/userhome/30/zyzhong2/controllable_diffusion/out2front/ink.png')
    return image

@app.route('/text-generate', methods=['POST'])
def generate_with_story():
    try:
        # Get data from request
        data = request.json
        # image_data = data.get('originalImage')  # Base64 or URL
        story = data.get('story')
        style = data.get('style')
        thr_model = data.get('model')
        
        if style == 1:
            output_image = story_to_ink(story)
        elif style == 2:
            output_image = story_to_bnw(story)
        elif style == 3:
            output_image = story_to_disney(story)
            if thr_model == 1:
                generate_toy(output_image)
                with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/stable_fast_3d/out/mesh.glb", "rb") as f:
                    glb_data = base64.b64encode(f.read()).decode('utf-8')
            else:
                gen_chop(output_image)
                with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_chopstick_pair.glb", "rb") as f:
                    glb_data = base64.b64encode(f.read()).decode('utf-8')
            
            print(glb_data)
            return jsonify({"artworkUrl": output_image, "glbData": f"data:model/gltf-binary;base64,{glb_data}"})
        
        out_path="/userhome/30/zyzhong2/controllable_diffusion/out/model.png"
        output_image.save(out_path)
        if thr_model == 1:
            generate_toy(out_path)
            with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/stable_fast_3d/out/mesh.glb", "rb") as f:
                glb_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            gen_chop(out_path)
            with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_chopstick_pair.glb", "rb") as f:
                glb_data = base64.b64encode(f.read()).decode('utf-8')

        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        output_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"artworkUrl": f"data:image/png;base64,{output_image_b64}", "glbData": f"data:model/gltf-binary;base64,{glb_data}"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_from_image():
    try:
        
        # # Check if file was uploaded
        # if 'file' not in request.files:
        # print(request.json)
        data = request.json
        print(data)
        image_data = data.get('image')  # Base64 or URL
        style = data.get('style')
        thr_model=data.get('model')
        image_data = image_data.split(',')[1]
        # story = data.get('story')
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        temp_image_path = '/userhome/30/zyzhong2/controllable_diffusion/input_frontend/temp_image.png'
        image.save(temp_image_path)
        # output_image = Image.open('/userhome/30/zyzhong2/controllable_diffusion/out/lora_output.png')
        interrogator = ImageInterrogator()
        interrogator.load_image(temp_image_path)
        prompt = interrogator.interrogate()
        # prompt = "Chinese Ink, " + prompt + ", 8k"
        print(prompt)
        # generator = ImageGenerator(method='depth')
        # control_image = generator.load_control_image(temp_image_path, method='depth')
        # output_image = generator.generate_image(prompt, control_image, method='depth')
        # print(output_image)
        # output_image.save('/userhome/30/zyzhong2/controllable_diffusion/out/lora_frontend.png')
        if style == 1:
            output_image = story_to_ink(prompt)
        elif style == 2:
            output_image = story_to_bnw(prompt)
        elif style == 3:
            output_image = story_to_disney(prompt)
            if thr_model == 1:
                generate_toy(output_image)
                with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/stable_fast_3d/out/mesh.glb", "rb") as f:
                    glb_data = base64.b64encode(f.read()).decode('utf-8')
            else:
                gen_chop(output_image)
                with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_chopstick_pair.glb", "rb") as f:
                    glb_data = base64.b64encode(f.read()).decode('utf-8')
            
            print(glb_data)
            return jsonify({"artworkUrl": output_image, "glbData": f"data:model/gltf-binary;base64,{glb_data}"})
        
        out_path="/userhome/30/zyzhong2/controllable_diffusion/out/model.png"
        output_image.save(out_path)
        if thr_model == 1:
            generate_toy(out_path)
            with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/stable_fast_3d/out/mesh.glb", "rb") as f:
                glb_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            gen_chop(out_path)
            with open("/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_chopstick_pair.glb", "rb") as f:
                glb_data = base64.b64encode(f.read()).decode('utf-8')

        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        output_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"artworkUrl": f"data:image/png;base64,{output_image_b64}", "glbData": f"data:model/gltf-binary;base64,{glb_data}"})
        
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)

