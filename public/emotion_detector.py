import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define the new model architecture from the second file
class ResEmoteNet(nn.Module):
    """
    Custom ResEmoteNet architecture based on ResNet backbone
    with attention mechanisms for emotion detection
    """
    def __init__(self, num_classes=7):
        super(ResEmoteNet, self).__init__()
        # Use ResNet18 as backbone
        resnet = models.resnet18(weights=None)  # No pre-trained weights needed for inference
        
        # Remove the last FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Spatial attention module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling and classification layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.features(x)
        
        # Apply attention mechanism
        attention_mask = self.attention(features)
        refined_features = features * attention_mask
        
        # Global average pooling
        x = self.avg_pool(refined_features)
        x = x.view(x.size(0), -1)
        
        # Classification layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Load the model
def load_emotion_model():
    model_path = r"C:\Users\Ghanim\Desktop\test\backend\Ai_Model\final_resemotenet.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    # Initialize model with the new architecture
    model = ResEmoteNet(num_classes=7)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            print("No face detected")
            return None
        
        # Get the largest face
        largest_face = None
        largest_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)
        
        x, y, w, h = largest_face
        face_img = image[y:y+h, x:x+w]
        
        # Convert to PIL image
        face_img = Image.fromarray(face_img)
        
        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Apply preprocessing
        input_tensor = preprocess(face_img)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        return input_batch
    
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

# Predict emotion
def predict_emotion(model, image_path):
    try:
        # Preprocess image
        input_batch = preprocess_image(image_path)
        if input_batch is None:
            return -1  # Error code
        
        # Predict
        with torch.no_grad():
            output = model(input_batch)
            
        # Get the predicted class
        _, predicted = torch.max(output, 1)
        
        return predicted.item()
    
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python emotion_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        sys.exit(1)
    
    # Load model
    model = load_emotion_model()
    if model is None:
        sys.exit(1)
    
    # Predict emotion
    emotion = predict_emotion(model, image_path)
    
    # Output only the emotion index for parsing
    print(emotion)