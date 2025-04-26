import trimesh
import numpy as np
from PIL import Image, ImageEnhance
import os
from trimesh.exchange import gltf
from PIL import Image
import requests
from io import BytesIO

# === 参数设置 ===
FRONT_IMAGE_PATH = '/userhome/30/zyzhong2/controllable_diffusion/three_D/input/silver.png'
# TOP_IMAGE_PATH = 'https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-f8acd759-2ec9-4a12-9d10-94e06eb706d9.jpg'
OUTPUT_PATH = '/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_chopstick_pair.glb'
TEXTURE_PATH = '/userhome/30/zyzhong2/controllable_diffusion/three_D/glb_out/square_combined_texture.png'

TOTAL_HEIGHT = 20.0
TOP_HEIGHT = 5.0
WIDTH_TOP = 0.6
WIDTH_BOTTOM = 0.2
SECTIONS = 64
DISTANCE_BETWEEN = 1.0

BRIGHTNESS_FACTOR = 1.4
CONTRAST_FACTOR = 1.2

def gen_chop(image_path):
    front_img = Image.open(FRONT_IMAGE_PATH).convert('RGB')
    if "http" not in image_path:
        # === 加载图像 ===  
        top_img = Image.open(image_path).convert('RGBA')
    else:
        response = requests.get(image_path)
        top_img = Image.open(BytesIO(response.content))


    # 增强亮度/对比度
    enhancer = ImageEnhance.Brightness(front_img)
    front_img = enhancer.enhance(BRIGHTNESS_FACTOR)
    front_img = ImageEnhance.Contrast(front_img).enhance(CONTRAST_FACTOR)

    top_img = ImageEnhance.Brightness(top_img).enhance(BRIGHTNESS_FACTOR)
    top_img = ImageEnhance.Contrast(top_img).enhance(CONTRAST_FACTOR)

    # === 合成上下分段纹理贴图 ===
    texture_width = max(top_img.width, front_img.width)
    texture_top = top_img.resize((texture_width, top_img.height))
    texture_front = front_img.resize((texture_width, front_img.height))
    texture_height = texture_top.height + texture_front.height
    combined_texture = Image.new('RGBA', (texture_width, texture_height), (255, 255, 255, 0))
    combined_texture.paste(texture_top, (0, 0))
    combined_texture.paste(texture_front, (0, texture_top.height))
    combined_texture.save(TEXTURE_PATH)
    combined_texture = combined_texture.convert('RGB')

    # === 方形筷子生成函数 ===
    def create_square_chopstick(x_offset=0.0):
        vertices = []
        faces = []
        uvs = []
        rings = []

        for i in range(SECTIONS):
            t = i / (SECTIONS - 1)
            z = TOTAL_HEIGHT * t
            w = WIDTH_TOP * (1 - t) + WIDTH_BOTTOM * t
            x0, x1 = -w / 2 + x_offset, w / 2 + x_offset
            y0, y1 = -w / 2, w / 2

            ring = [
                len(vertices) + 0,
                len(vertices) + 1,
                len(vertices) + 2,
                len(vertices) + 3,
            ]
            rings.append(ring)

            vertices.extend([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])

            # 分段 UV 映射（顶部图像占 25%，材质图占 75%）
            if t <= TOP_HEIGHT / TOTAL_HEIGHT:
                # 映射到纹理上方的 0~0.25 区域
                v = 1 - (t / (TOP_HEIGHT / TOTAL_HEIGHT)) * 0.25
            else:
                # 映射到纹理下方的 0.25~1 区域
                vt = (t - TOP_HEIGHT / TOTAL_HEIGHT) / (1 - TOP_HEIGHT / TOTAL_HEIGHT)
                v = 0.75 * (1 - vt)

            uvs.extend([[0, v], [0.33, v], [0.66, v], [1, v]])

        for i in range(len(rings) - 1):
            r1 = rings[i]
            r2 = rings[i + 1]
            for j in range(4):
                a = r1[j]
                b = r1[(j + 1) % 4]
                c = r2[(j + 1) % 4]
                d = r2[j]
                faces.append([a, b, c])
                faces.append([a, c, d])

        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=combined_texture)
        return mesh

    # === 创建一对方筷子 ===
    left = create_square_chopstick(x_offset=-DISTANCE_BETWEEN / 2)
    right = create_square_chopstick(x_offset=+DISTANCE_BETWEEN / 2)

    scene = trimesh.Scene()
    scene.add_geometry(left, node_name='square_chopstick_left')
    scene.add_geometry(right, node_name='square_chopstick_right')

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(gltf.export_glb(scene))

    print(f'✅ 方形筷子对已生成并保存至：{OUTPUT_PATH}')