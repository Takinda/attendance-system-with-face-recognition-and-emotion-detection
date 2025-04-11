import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the same model architecture as trained
class ResEmoteNet(nn.Module):
    """
    Custom ResEmoteNet architecture based on ResNet backbone
    with attention mechanisms for emotion detection
    """
    def __init__(self, num_classes):
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

# Function to detect and display emotion in real-time
def real_time_emotion_detection():
    # Set up emotion labels (update these to match your trained model's emotions)
    emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
    
    # Define the image transformation pipeline (same as used during inference)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = r'C:\Users\Ghanim\Desktop\project codes\خريط\final_resemotenet.pth'  # Update this to your model path
    model = ResEmoteNet(num_classes=len(emotions))
    
    try:
        # Try to load model weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    # Face detector using OpenCV's Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region with some margin
            face_margin = int(0.1 * w)  # 10% margin
            x1 = max(0, x - face_margin)
            y1 = max(0, y - face_margin)
            x2 = min(frame.shape[1], x + w + face_margin)
            y2 = min(frame.shape[0], y + h + face_margin)
            
            # Extract face
            face_img = frame[y1:y2, x1:x2]
            
            # Convert to PIL for transformation
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            # Apply transformation and prepare for model
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
            
            # Get emotion label and confidence
            emotion = emotions[predicted_idx]
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display emotion text
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Show emotion confidence bars
            bar_width = 100
            bar_height = 15
            bar_gap = 5
            bar_x = x1
            bar_y = y2 + 10
            
            for i, emo in enumerate(emotions):
                prob = probabilities[i].item()
                # Draw emotion label
                cv2.putText(frame, emo, (bar_x, bar_y + i*(bar_height+bar_gap)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Draw empty bar
                cv2.rectangle(frame, 
                             (bar_x + 70, bar_y + i*(bar_height+bar_gap) - bar_height + 5), 
                             (bar_x + 70 + bar_width, bar_y + i*(bar_height+bar_gap)), 
                             (255, 255, 255), 1)
                # Fill bar according to probability
                filled_width = int(prob * bar_width)
                cv2.rectangle(frame, 
                             (bar_x + 70, bar_y + i*(bar_height+bar_gap) - bar_height + 5), 
                             (bar_x + 70 + filled_width, bar_y + i*(bar_height+bar_gap)), 
                             (0, 255, 0), -1)
        
        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Function to test with still images
def test_with_image(image_path):
    # Set up emotion labels (update these to match your trained model's emotions)
    emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
    
    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'best_resemotenet.pth'  # Update this to your model path
    model = ResEmoteNet(num_classes=len(emotions))
    
    try:
        # Try to load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Load and process the image
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Get emotion label and confidence
        emotion = emotions[predicted_idx]
        
        # Display the results
        print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
        print("All emotion probabilities:")
        for i, emo in enumerate(emotions):
            print(f"  {emo}: {probabilities[i].item():.4f}")
        
        # Display the image with prediction
        img_cv = cv2.imread(image_path)
        text = f"{emotion}: {confidence:.2f}"
        cv2.putText(img_cv, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection (Still Image)', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    # Comment or uncomment based on what you want to do
    
    # For real-time emotion detection with webcam:
    real_time_emotion_detection()
    
    # For testing with a still image:
    # test_with_image('path/to/your/image.jpg')