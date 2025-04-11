import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import traceback
import time

# Define the SEBlock for Squeeze-and-Excitation
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define the ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResEmoteNet model
class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(256)
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 7)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.se(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

class EmotionDetector:
    def __init__(self, model_path='Ai_Model/raf-db.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResEmoteNet().to(self.device)
        
        # Load the pre-trained model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # If the saved model is a state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            # If the saved model is directly the state_dict
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.eval()  # Set the model to evaluation mode
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define emotion labels (RAF-DB dataset has 7 emotions)
        self.emotions = {
            0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happiness',
            4: 'Sadness',
            5: 'Anger',
            6: 'Neutral'
        }
    
    def preprocess_image(self, image):
        """
        Preprocess the image for the model.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            # Convert OpenCV BGR to RGB
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def detect_emotion(self, image):
        """
        Detect emotion from an image.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            
        Returns:
            dict: Detected emotion and confidence scores
        """
        # Handle different input types
        if isinstance(image, str):
            if os.path.isfile(image):
                image = Image.open(image).convert('RGB')
            else:
                raise FileNotFoundError(f"Image file not found: {image}")
        
        # Preprocess the image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class and its probability
            prob_values, pred_indices = torch.max(probabilities, 1)
            emotion_idx = pred_indices.item()
            confidence = prob_values.item()
            
            # Get all emotion probabilities
            all_probs = probabilities[0].cpu().numpy()
            emotion_probs = {self.emotions[i]: float(all_probs[i]) for i in range(len(self.emotions))}
            
            return {
                'emotion': self.emotions[emotion_idx],
                'confidence': confidence,
                'all_emotions': emotion_probs
            }
    
    def detect_emotion_from_face(self, frame, face_cascade_path='haarcascade_frontalface_default.xml'):
        """
        Detect emotion from faces in a video frame.
        
        Args:
            frame: Video frame (numpy array)
            face_cascade_path: Path to Haar cascade XML file for face detection
            
        Returns:
            list: List of dictionaries containing face location and emotion
        """
        # Check if the cascade file exists, download if not
        if not os.path.exists(face_cascade_path):
            print(f"Cascade file not found at {face_cascade_path}, downloading...")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, face_cascade_path)
                print(f"Downloaded cascade file to {face_cascade_path}")
            except Exception as e:
                print(f"Failed to download cascade file: {e}")
                print("Using a simple face detection method instead...")
                # Return a simple face detection (whole frame as face)
                height, width = frame.shape[:2]
                face_img = frame
                emotion_result = self.detect_emotion(face_img)
                emotion_result['face'] = (0, 0, width, height)
                return [emotion_result]
        
        # Load the face detector
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Check if face cascade loaded correctly
        if face_cascade.empty():
            print(f"Error: Could not load face cascade from {face_cascade_path}")
            print("Using a simple face detection method instead...")
            # Return a simple face detection (whole frame as face)
            height, width = frame.shape[:2]
            face_img = frame
            emotion_result = self.detect_emotion(face_img)
            emotion_result['face'] = (0, 0, width, height)
            return [emotion_result]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve face detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with more conservative parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Higher value for fewer detections
            minNeighbors=8,   # Higher value for more reliable detections
            minSize=(60, 60), # Larger minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces detected, try with slightly more lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # If multiple faces detected, choose the largest one or the most central one
        if len(faces) > 1:
            # Calculate the center of the frame
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            
            # Find the face with the largest area or closest to center
            max_area = 0
            best_face = None
            min_distance = float('inf')
            
            for face in faces:
                x, y, w, h = face
                area = w * h
                
                # Calculate distance to center
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                distance = ((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2) ** 0.5
                
                # Prefer larger faces that are close to center
                if area > max_area and distance < min_distance * 1.5:
                    max_area = area
                    best_face = face
                    min_distance = distance
                elif distance < min_distance and area > max_area * 0.7:
                    min_distance = distance
                    best_face = face
            
            if best_face is not None:
                faces = [best_face]
        
        results = []
        # If still no faces detected, use the whole frame
        if len(faces) == 0:
            height, width = frame.shape[:2]
            face_img = frame
            emotion_result = self.detect_emotion(face_img)
            emotion_result['face'] = (0, 0, width, height)
            results.append(emotion_result)
        else:
            for (x, y, w, h) in faces:
                # Expand the face region slightly for better emotion detection
                x_expanded = max(0, x - int(w * 0.1))
                y_expanded = max(0, y - int(h * 0.1))
                w_expanded = min(frame.shape[1] - x_expanded, int(w * 1.2))
                h_expanded = min(frame.shape[0] - y_expanded, int(h * 1.2))
                
                # Extract face
                face_img = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                
                if face_img.size == 0:  # If face extraction failed
                    continue
                
                # Detect emotion
                emotion_result = self.detect_emotion(face_img)
                
                # Add face location (use the expanded coordinates)
                emotion_result['face'] = (x_expanded, y_expanded, w_expanded, h_expanded)
                results.append(emotion_result)
        
        return results

# Example usage
if __name__ == "__main__":
    try:
        print("Starting ResEmoteNet Emotion Detector...")
        
        # Initialize the emotion detector
        detector = EmotionDetector(model_path='Ai_Model/raf-db.pth')
        
        print("Starting real-time emotion detection using webcam...")
        print("The program will automatically download face detection files if needed.")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            print("Make sure your camera is connected and not being used by another application.")
            exit()
            
        print("Press 'q' to quit")
        
        # For calculating FPS
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:  # Update FPS every 10 frames
                current_time = time.time()
                fps = frame_count / (current_time - prev_time)
                prev_time = current_time
                frame_count = 0
            
            try:
                # Detect emotions in the frame
                face_results = detector.detect_emotion_from_face(frame)
                
                # Draw rectangles around faces and display emotions
                for result in face_results:
                    x, y, w, h = result['face']
                    emotion = result['emotion']
                    confidence = result['confidence']
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion text
                    text = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Display all emotions in a bar on the side
                    bar_height = 20
                    max_bar_width = 100
                    start_x = frame.shape[1] - max_bar_width - 10
                    start_y = 30
                    
                    for i, (emo, prob) in enumerate(result['all_emotions'].items()):
                        # Draw emotion label
                        cv2.putText(frame, emo, (start_x - 70, start_y + i*bar_height), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw probability bar
                        bar_width = int(prob * max_bar_width)
                        
                        # Use color based on emotion
                        if emo == 'Happiness':
                            color = (0, 255, 0)  # Green for happiness
                        elif emo == 'Sadness':
                            color = (255, 0, 0)  # Blue for sadness
                        elif emo == 'Anger':
                            color = (0, 0, 255)  # Red for anger
                        elif emo == 'Fear':
                            color = (255, 0, 255)  # Purple for fear
                        elif emo == 'Surprise':
                            color = (0, 255, 255)  # Yellow for surprise
                        elif emo == 'Disgust':
                            color = (255, 255, 0)  # Cyan for disgust
                        else:
                            color = (200, 200, 200)  # Gray for neutral
                            
                        cv2.rectangle(frame, 
                                    (start_x, start_y + i*bar_height - 10), 
                                    (start_x + bar_width, start_y + i*bar_height), 
                                    color, 
                                    -1)  # Filled rectangle
                        
                        # Display probability value
                        cv2.putText(frame, f"{prob:.2f}", 
                                  (start_x + bar_width + 5, start_y + i*bar_height), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            except Exception as e:
                # Display error on the frame
                error_msg = f"Error: {str(e)}"
                cv2.putText(frame, error_msg, (10, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print(f"Error during detection: {e}")
                traceback.print_exc()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display status (running/error)
            cv2.putText(frame, "Status: Running", (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display the resulting frame
            cv2.imshow('ResEmoteNet Emotion Detection', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure PyTorch, OpenCV, and PIL are installed")
        print("2. Check that your model file exists at 'Ai_Model/raf-db.pth'")
        print("3. Make sure your webcam is working and not in use by another application")
        print("4. If the face detection file is missing, the program will try to download it")