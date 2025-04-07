import sys
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
import shutil  # Import shutil for deleting folders and their contents

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, min_face_size=40, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load the FaceNet model for face recognition
model = InceptionResnetV1(pretrained='vggface2').eval()

# Directory where known faces are stored
KNOWN_FACES_DIR = "public"

# Function to generate face embeddings
def get_embedding(face_img):
    min_size = 160  # FaceNet expects 160x160 images
    if face_img.shape[0] < min_size or face_img.shape[1] < min_size:
        face_img = cv2.resize(face_img, (min_size, min_size))

    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding = model(face_tensor).numpy()
    return embedding

# Function to load known faces and their embeddings
def load_known_faces():
    if os.path.exists("public/known_faces.pkl"):
        with open("public/known_faces.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# Function to save known faces
def save_known_faces(known_faces):
    with open("public/known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# Load stored faces
known_faces = load_known_faces()

# Process the images in the user's folder
def process_user_images(user_folder, user_id):
    images = [f for f in os.listdir(user_folder) if f.endswith(".jpg")]
    
    print(f"Processing {len(images)} images for user {user_id} in folder {user_folder}")

    for image in images:
        img_path = os.path.join(user_folder, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image {image}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img_rgb[y1:y2, x1:x2]
                embedding = get_embedding(face)
                known_faces[user_id] = embedding  # Save the face to the known faces
        else:
            print(f"No face detected in image {image}")

    save_known_faces(known_faces)

    # Delete the user folder after processing
    print(f"Deleting user folder: {user_folder}")
    shutil.rmtree(user_folder)  # Use shutil.rmtree to delete folder and its contents

# Main entry point
if __name__ == "__main__":
    user_folder = sys.argv[1]
    user_id = sys.argv[2]

    process_user_images(user_folder, user_id)
