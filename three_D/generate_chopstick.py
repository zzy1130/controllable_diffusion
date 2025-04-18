import trimesh
import numpy as np
from PIL import Image
import os
from trimesh.exchange import gltf

# === 参数设置 ===
FRONT_IMAGE_PATH = 'https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-233715a9-ef49-49ac-a0ec-8abe74c76647.jpg'
TOP_IMAGE_PATH = 'input/top/top.png'
OUTPUT_PATH = 'glb_out/chopstick_pair.glb'
TEXTURE_PATH = 'glb_out/combined_texture.png'

# 筷子尺寸
TOTAL_HEIGHT = 20.0
TOP_HEIGHT = 5.0
RADIUS = 0.3
BOTTOM_RADIUS = 0.1
SECTIONS = 64
DISTANCE_BETWEEN = 1.0  # 两根筷子的水平间距

# === 加载图像 ===
front_img = Image.open(FRONT_IMAGE_PATH).convert('RGB')
top_img = Image.open(TOP_IMAGE_PATH).convert('RGBA')

# === 合成纹理贴图 ===
texture_width = max(top_img.width, front_img.width)
texture_height = top_img.height + front_img.height

combined_texture = Image.new('RGBA', (texture_width, texture_height), (255, 255, 255, 0))
combined_texture.paste(top_img.resize((texture_width, top_img.height)), (0, 0))
combined_texture.paste(front_img.resize((texture_width, front_img.height)), (0, top_img.height))

# 保存合成图（调试用）
os.makedirs('glb_out', exist_ok=True)
combined_texture.save(TEXTURE_PATH)

# 转换为 RGB（GLB 要求）
combined_texture = combined_texture.convert('RGB')

# === 函数：生成一根筷子的网格 ===
def create_chopstick_mesh(x_offset=0.0):
    z = np.linspace(0, TOTAL_HEIGHT, SECTIONS)
    r = np.linspace(RADIUS, BOTTOM_RADIUS, SECTIONS)

    vertices = []
    faces = []
    uvs = []

    for i in range(SECTIONS):
        v = i / (SECTIONS - 1)
        for j in range(SECTIONS):
            u = j / SECTIONS
            theta = 2 * np.pi * j / SECTIONS
            x = r[i] * np.cos(theta) + x_offset
            y = r[i] * np.sin(theta)
            z_pos = z[i]
            vertices.append([x, y, z_pos])
            uvs.append([u, 1 - v])

    for i in range(SECTIONS - 1):
        for j in range(SECTIONS):
            next_j = (j + 1) % SECTIONS
            a = i * SECTIONS + j
            b = i * SECTIONS + next_j
            c = (i + 1) * SECTIONS + next_j
            d = (i + 1) * SECTIONS + j
            faces.append([a, b, c])
            faces.append([a, c, d])

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=combined_texture)
    return mesh

# === 创建两根筷子 ===
left_chopstick = create_chopstick_mesh(x_offset=-DISTANCE_BETWEEN / 2)
right_chopstick = create_chopstick_mesh(x_offset=+DISTANCE_BETWEEN / 2)

# === 合并场景并导出 GLB ===
scene = trimesh.Scene()
scene.add_geometry(left_chopstick, node_name='chopstick_left')
scene.add_geometry(right_chopstick, node_name='chopstick_right')

glb_bytes = gltf.export_glb(scene)

with open(OUTPUT_PATH, 'wb') as f:
    f.write(glb_bytes)

print(f'✅ 筷子模型对已生成并保存至：{OUTPUT_PATH}')