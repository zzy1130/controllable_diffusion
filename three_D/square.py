import trimesh
import numpy as np
from PIL import Image
import os
from trimesh.exchange import gltf

# === 参数设置 ===
FRONT_IMAGE_PATH = 'input/material/silver.jpg'
TOP_IMAGE_PATH = 'input/top/abalone.png'
OUTPUT_PATH = 'glb_out/square_chopstick_pair.glb'
TEXTURE_PATH = 'glb_out/square_combined_texture.png'

TOTAL_HEIGHT = 20.0
TOP_HEIGHT = 5.0
WIDTH_TOP = 0.6
WIDTH_BOTTOM = 0.2
SECTIONS = 64
DISTANCE_BETWEEN = 1.0

# === 加载图像 ===
front_img = Image.open(FRONT_IMAGE_PATH).convert('RGB')
top_img = Image.open(TOP_IMAGE_PATH).convert('RGBA')

texture_width = max(top_img.width, front_img.width)
texture_height = top_img.height + front_img.height
combined_texture = Image.new('RGBA', (texture_width, texture_height), (255, 255, 255, 0))
combined_texture.paste(top_img.resize((texture_width, top_img.height)), (0, 0))
combined_texture.paste(front_img.resize((texture_width, front_img.height)), (0, top_img.height))
os.makedirs('glb_out', exist_ok=True)
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
        v = 1 - t
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