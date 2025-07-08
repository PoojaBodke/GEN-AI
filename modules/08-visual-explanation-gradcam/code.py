"""
Grad-CAM demo with a pre-trained ResNet.
"""

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and preprocess the image
img = Image.open("cat.jpg")
input_tensor = preprocess(img).unsqueeze(0)

# Register hook to get gradients and feature maps
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Forward pass
output = model(input_tensor)
class_idx = output.argmax()

# Backward pass to get gradients
model.zero_grad()
output[0, class_idx].backward()

# Compute Grad-CAM
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
gradcam = torch.zeros(features.shape[2:], dtype=torch.float32)

for i in range(features.shape[1]):
    gradcam += pooled_gradients[i] * features[0, i, :, :]

gradcam = torch.relu(gradcam)
gradcam /= gradcam.max()

# Convert to numpy & resize
gradcam = gradcam.detach().numpy()
gradcam = cv2.resize(gradcam, (img.width, img.height))

# Overlay
heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
img_np = np.array(img)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# Show and save
plt.imshow(overlay)
plt.axis('off')
plt.savefig("gradcam_result.png")
print("Saved Grad-CAM visualization as gradcam_result.png")
