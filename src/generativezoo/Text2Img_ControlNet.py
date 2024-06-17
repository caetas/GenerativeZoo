from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import argparse
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Text2Img LoRA")
    parser.add_argument("--cnet_model_path", type=str, default="./../../models/Text2Img_ControlNet/pokemons/checkpoint-200/controlnet", help="Path to ControlNet model")
    parser.add_argument("--cond_image_path", type=str, default="./../../data/processed/pokemons/conditioning_images/0003_mask.png", help="Path to conditioning image")
    args = parser.parse_args()
    return args

args = parse_args()

controlnet = ControlNetModel.from_pretrained(args.cnet_model_path, torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image(args.cond_image_path)

while True:
    text = input("Enter prompt (0 to exit): ")
    if text == "0":
        break
    image = pipeline(text, num_inference_steps=20, image=control_image).images[0]
    plt.imshow(image)
    plt.show()
    new_cond = input("Enter new conditioning image (0 to keep the same): ")
    if new_cond == "0":
        continue
    control_image = load_image(new_cond)