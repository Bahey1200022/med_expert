
import os
import torch
import numpy as np
import cv2
import config

from torchvision import transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="resnet34",        # Choose encoder, e.g., resnet34, mobilenet_v2, efficientnet-b7, etc.
        encoder_weights="imagenet",     # Use 'imagenet' pre-trained weights for encoder initialization
        in_channels=3,                  # Model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1                       # Model output channels (number of classes in your dataset)
    ).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model



def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(image, (256, 256))  # Resize for display purposes
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image, original_image



def segment_image(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8)
    return output

