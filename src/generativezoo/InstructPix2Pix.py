import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.utils import load_image
import argparse
from matplotlib import pyplot as plt

def parse_args():
    arg_parser = argparse.ArgumentParser(description="InstructPix2Pix")
    arg_parser.add_argument("--pix2pix_model", type=str, default='./../../models/InstructPix2Pix/checkpoint-20/unet', help="The name of the Pix2Pix model to use")
    arg_parser.add_argument("--image_path", type=str, default="./../../data/processed/pokemons/conditioning_images/0003_mask.png", help="The path to the image to edit")
    return arg_parser.parse_args()

args = parse_args()

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", unet = UNet2DConditionModel.from_pretrained(args.pix2pix_model, torch_dtype=torch.float16), torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

original_image = load_image(args.image_path)

while True:
    text = input("Enter prompt (0 to exit): ")
    if text == "0":
        break
    image = pipeline(text,image=original_image,num_inference_steps=num_inference_steps,image_guidance_scale=image_guidance_scale,guidance_scale=guidance_scale,generator=generator).images[0]
    plt.imshow(image)
    plt.show()
    new_original = input("Enter new original image (0 to keep the same): ")
    if new_original == "0":
        continue
    original_image = load_image(new_original)
