from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter

# --- Helper function to wrap text for drawing ---
def wrap_text(text, max_width, draw, font):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

# --- Paths ---
yolo_model_path = r"C:\nuv\AI_SURVILANCE_05\best (6).pt"
image_path = r"C:\nuv\AI_SURVILANCE_05\test\images\armas-1109-_jpg.rf.d6ffbb0571e9a204f1a5822ed1e526ff.jpg"

# --- Load YOLOv8 model and run inference ---
model = YOLO(yolo_model_path)
results = model(image_path, save=True)

# Get saved YOLO output image path
output_dir = results[0].save_dir
filename = os.path.basename(image_path)
output_image_path = os.path.join(output_dir, filename)

# --- Load BLIP model and processor ---
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image for BLIP captioning
image_for_blip = Image.open(output_image_path).convert("RGB")

# Generate BLIP caption
inputs = processor(image_for_blip, return_tensors="pt")
out = blip_model.generate(**inputs)
vlm_caption = processor.decode(out[0], skip_special_tokens=True)

# Get detected classes from YOLO results
detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

# --- Custom descriptive caption generation ---
class_counts = Counter(detected_classes)
description_parts = []

if class_counts["person"] >= 2:
    if "gun" in class_counts or "pistol" in class_counts:
        description_parts.append("One person is aiming a gun at another person")
    elif "knife" in class_counts:
        description_parts.append("One person is threatening another person with a knife")
    elif "violence" in class_counts:
        description_parts.append("Two people are engaged in a violent act")
    else:
        description_parts.append("Multiple people are present")

elif class_counts["person"] == 1:
    if "gun" in class_counts or "pistol" in class_counts:
        description_parts.append("A person is holding a gun")
    elif "knife" in class_counts:
        description_parts.append("A person is holding a knife")
    else:
        description_parts.append("A person is present")

if not description_parts:
    description_parts.append("Scene contains: " + ", ".join(detected_classes))

# Combine descriptions
combined_caption = f"{' | '.join(description_parts)} || VLM caption: {vlm_caption}"

# --- Draw combined caption on the image ---
img_pil = Image.open(output_image_path).convert("RGB")
draw = ImageDraw.Draw(img_pil)

# Choose a font
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    font = ImageFont.load_default()

# Wrap and position text
max_width = img_pil.width - 20
lines = wrap_text(combined_caption, max_width, draw, font)

y_text = img_pil.height - (len(lines) * 25) - 10
if y_text < 0:
    y_text = 10

# Draw background for text
text_height = sum((draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] + 5) for line in lines)
overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
overlay_draw = ImageDraw.Draw(overlay)
overlay_draw.rectangle(((5, y_text - 5), (img_pil.width - 5, y_text + text_height + 5)), fill=(0, 0, 0, 150))
img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
draw = ImageDraw.Draw(img_pil)

# Draw text
for line in lines:
    draw.text((10, y_text), line, font=font, fill=(255, 255, 255))
    bbox = draw.textbbox((0, 0), line, font=font)
    line_height = bbox[3] - bbox[1]
    y_text += line_height + 5

# Save and show final image
final_output_path = os.path.join(output_dir, "vlm_captioned_" + filename)
img_pil.save(final_output_path)
img_pil.show()

# Terminal output
print("Combined Caption:")
print(combined_caption)
print(f"\nFinal image saved at: {final_output_path}")
