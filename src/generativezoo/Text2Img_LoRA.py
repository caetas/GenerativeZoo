from diffusers import AutoPipelineForText2Image
import torch
import argparse
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Text2Img LoRA")
    parser.add_argument("--lora_model_path", type=str, default="./../../models/Text2Img_Lora/naruto/checkpoint-60", help="Path to LoRA model")
    args = parser.parse_args()
    return args

args = parse_args()
pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(args.lora_model_path, weight_name="pytorch_lora_weights.safetensors")

while True:
    text = input("Enter prompt (0 to exit): ")
    if text == "0":
        break
    image = pipeline(text).images[0]
    plt.imshow(image)
    plt.show()