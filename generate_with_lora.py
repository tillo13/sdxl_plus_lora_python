import os
import warnings
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from datetime import datetime
import time

# GLOBAL VARIABLES
MODEL_ID = "martyn/sdxl-turbo-mario-merge-top-rated"
MODEL_FILENAME = "topRatedTurboxlLCM_v10.safetensors"
LORA_MODEL_ID = 'ntc-ai/SDXL-LoRA-slider.huge-anime-eyes'
LORA_WEIGHT_NAME = 'huge anime eyes.safetensors'
PROMPT = "a sat anime cat with sad cat eyes"
NEGATIVE_PROMPT = "happy"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.5  # Typically 7.5 for good quality
LORA_STRENGTH = 3.0
NUMBER_OF_LOOPS = 3
SEED = 42

# Set up logging
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used when initializing")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

# Create output directory
output_dir = "generated_images"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")

# Download the model
log("Downloading model...")
model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME)
log("Model downloaded.")

# Load the Stable Diffusion model from Hugging Face
log("Loading pipeline...")
pipeline = StableDiffusionXLPipeline.from_single_file(model_path)
pipeline.to(device)
log("Pipeline loaded.")

# Change the scheduler to Euler Ancestral Discrete Scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

# Function to generate and save an image
def generate_image(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, filepath, seed, apply_lora=False, lora_strength=2.0):
    if apply_lora:
        log("Applying LoRA weights...")
        adapter_name = f"huge_anime_eyes_{seed}"  # Unique adapter name
        pipeline.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME, adapter_name=adapter_name)
        pipeline.set_adapters([adapter_name], adapter_weights=[lora_strength])
        log("LoRA weights applied.")
    else:
        log(f"Generating image without LoRA ({filepath})...")

    generator = torch.Generator(device=device).manual_seed(seed)  # Ensure reproducibility with the same seed
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator  # Add the same generator for reproducibility
    ).images[0]

    image.save(filepath)
    log(f"Image saved at: {filepath}")

# Timer start
start_time = time.time()

# Parameters for image generation
for i in range(NUMBER_OF_LOOPS):
    current_seed = SEED + i  # Increment the seed for each loop

    # Generate and save the image without applying LoRA first
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath_without_lora = os.path.join(output_dir, f"{timestamp}_without_lora_{i}.png")
    generate_image(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        filepath=filepath_without_lora,
        seed=current_seed,
        apply_lora=False
    )

    # Apply LoRA and save the image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath_with_lora = os.path.join(output_dir, f"{timestamp}_with_lora_{i}.png")
    generate_image(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        filepath=filepath_with_lora,
        seed=current_seed,
        apply_lora=True,
        lora_strength=LORA_STRENGTH
    )

# Timer end
end_time = time.time()
elapsed_time = end_time - start_time

# Summary log
log("== SUMMARY ==")
log(f"Total iterations: {NUMBER_OF_LOOPS}")
log(f"Elapsed time: {elapsed_time:.2f} seconds")