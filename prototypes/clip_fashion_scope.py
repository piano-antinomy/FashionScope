import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image where makeup is applied
image = Image.open("/Users/yingjianding/Documents/imagesForFashionScope/CHANELROUGE16022023.jpg")

# Define makeup categories or descriptions to recognize
makeup_descriptions = [
    "red lipstick", "eyeliner", "foundation", "blush", "mascara", "nude lipstick", "cat-eye makeup",
    # Additional descriptions:
    "smokey eye", "natural makeup", "bold eyebrows", "glitter eyeshadow", "contouring",
    "highlighter", "bronzer", "winged eyeliner", "false eyelashes", "matte finish",
    "dewy skin", "glossy lips", "dark lipstick", "pastel eyeshadow", "no makeup look",
    "dramatic eye makeup", "rosy cheeks", "heavy foundation", "bright eyeshadow",
    "gothic makeup", "colorful eyeliner", "subtle lip color", "shimmery eyeshadow", 
    # specific lipsticks
    "Chanel Rouge Allure Velvet Luminous Matte Lip Color", "Chanel Rouge Coco Flash", "Chanel Rouge Allure Ink", 
    "gucci rouge agrave l egrave vres voile sheer lipstick P452737", "gucci rouge a l eacute gant satin lipstick P452737"
]

# Preprocess image and text for CLIP
inputs = processor(text=makeup_descriptions, images=image, return_tensors="pt", padding=True)

# Perform inference using CLIP
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-to-text similarity
    probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

# Print the probabilities for each makeup description
for desc, prob in zip(makeup_descriptions, probs[0]):
    print(f"{desc}: {prob.item():.4f}")
