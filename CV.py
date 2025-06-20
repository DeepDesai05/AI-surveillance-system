from ultralytics import YOLO
import cv2
import os
from matplotlib import pyplot as plt

# Load trained model
model_path = r"C:\nuv\AI_SURVILANCE_05\best (6).pt"
model = YOLO(model_path)

# Load test image
image_path = r"C:\nuv\AI_SURVILANCE_05\test\images\V_333_F69_jpg.rf.008c7175b118fa20592d051540717f48.jpg"

# Run inference (this will auto-save to 'runs/detect/predict/')
results = model(image_path, save=True)

# Get path to saved prediction image
output_dir = results[0].save_dir  # Directory where prediction was saved
filename = os.path.basename(image_path)
output_image_path = os.path.join(output_dir, filename)

# Read and display the result
result_img = cv2.imread(output_image_path)
result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

plt.imshow(result_img)
plt.axis('off')
plt.title("YOLOv8 Prediction")
plt.show()
