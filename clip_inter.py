from PIL import Image
from clip_interrogator import Config, Interrogator

class ImageInterrogator:
    def __init__(self, clip_model_name="ViT-L-14/openai"):
        # Initialize the Interrogator with the specified CLIP model
        self.config = Config(clip_model_name=clip_model_name)
        self.interrogator = Interrogator(self.config)

    def load_image(self, image_path):
        # Load and convert the image to RGB
        self.image = Image.open(image_path).convert('RGB')

    def interrogate(self):
        # Interrogate the loaded image and return the result
        if hasattr(self, 'image'):
            return self.interrogator.interrogate(self.image)
        else:
            raise ValueError("Image not loaded. Please load an image first using load_image() method.")

if __name__ == "__main__":
    image_path = ''
    interrogator = ImageInterrogator()
    interrogator.load_image(image_path)
    result = interrogator.interrogate()
    print(result)