import sys
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, min_face_size=40, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load the FaceNet model for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to generate face embeddings
def get_embedding(face_img):
    min_size = 160
    if face_img.shape[0] < min_size or face_img.shape[1] < min_size:
        face_img = cv2.resize(face_img, (min_size, min_size))

    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding = model(face_tensor).numpy()[0]  # Get the first item from batch
    return embedding

def recognize_face(image_path):
    # Check if the pickle file exists
    if not os.path.exists("public/known_faces.pkl"):
        return "Unknown"
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown"
    
    # Convert to RGB for the model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    boxes, _ = mtcnn.detect(img_rgb)
    
    if boxes is None:
        return "Unknown"
    
    # Get the largest face in the image (assuming closest to camera)
    largest_area = 0
    largest_face = None
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_face = img_rgb[y1:y2, x1:x2]
    
    if largest_face is None:
        return "Unknown"
    
    # Get embedding for the detected face
    try:
        face_embedding = get_embedding(largest_face)
    except Exception:
        return "Unknown"
    
    # Load known faces from pickle file
    try:
        with open("public/known_faces.pkl", "rb") as f:
            known_faces = pickle.load(f)
    except Exception:
        return "Unknown"
    
    if not known_faces:
        return "Unknown"
    
    # Compare with known faces
    best_match = "Unknown"
    best_dist = float("inf")
    threshold = 0.8  # Adjust this threshold as needed
    
    for name, known_embedding in known_faces.items():
        # Calculate Euclidean distance
        dist = np.linalg.norm(face_embedding - known_embedding)
        
        if dist < threshold and dist < best_dist:
            best_match = name
            best_dist = dist
    
    return best_match

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Unknown")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("Unknown")
        sys.exit(1)
    
    recognized_name = recognize_face(image_path)
    # Print just the name without any additional text or newlines
    print(recognized_name, end='')