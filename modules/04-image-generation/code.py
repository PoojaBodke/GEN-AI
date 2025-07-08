"""
Generate images from text prompts using Stable Diffusion.
"""

from diffusers import StableDiffusionPipeline

# Load model (first time it will download)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Use your GPU if available
pipe.to("cuda")

# Prompt to generate
prompt = "A futuristic city skyline at sunset"

# Generate image
image = pipe(prompt).images[0]

# Save to file
image.save("generated_image.png")

print("Image saved as generated_image.png")
