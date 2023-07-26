# Imports
from diffusers import StableDiffusionPipeline, DiffusionPipeline  
import torch
from PIL import Image
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="input image path")

args = parser.parse_args()
input_image = args.input

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-1"

if device=="cuda":
    torch_dtype = torch.float16 
else:
    torch_dtype = torch.float32

pipe_df = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
pipe_sdf = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)

# Enable memory efficient attention
pipe_df.enable_xformers_memory_efficient_attention()  

# Load image
original_image = Image.open(input_image).convert("RGB").resize((768, 512))

# Define prompt
prompt = "Van Gogh style painting"  

# Benchmark
models = [pipe_df, pipe_sdf]
if device == 'cuda':
    print(torch.cuda.memory_allocated())

for pipe in models:

    start = time.time()

    for i in range(4):
        
        # Generate image
        images = pipe(prompt, image=original_image, strength=0.75, guidance_scale=7.5).images

    end = time.time()
    total_time = end - start
    
    print(f"4 generations of {pipe.model_id} took {total_time:.2f} seconds")
    print(f"Average generation time: {total_time/4:.2f} seconds")
    
    # Save image
    images[0].save(f"{pipe.model_id}_style.png")