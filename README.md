# Face Recognition and Emotion Detection Installation
## this project was made as graduation project in **Computer Engineering Department - University of technology** in Baghdad - Iraq



This guide provides step-by-step instructions for setting up both the Python environment for face recognition and the Node.js backend server.

## Python Dependencies Installation

The face recognition system requires several Python packages. Install them using pip:

```bash
# Install required Python packages
pip install opencv-python
pip install torch torchvision
pip install numpy
pip install facenet-pytorch
pip install pickle-mixin
```

If you're using a GPU and want to install PyTorch with CUDA support, visit the official PyTorch website (https://pytorch.org/get-started/locally/) to get the appropriate installation command for your system.

## Node.js Backend Installation

1. Create a new directory for your backend project:

```bash
mkdir project
cd project
```

2. Install the dependencies:

```bash
npm install
```

## Project Structure

Create the following basic structure for your project:

```
project/
│
├── database/
│   ├── users.mjs
│   └── attendance.mjs
|── public/
|   ├── adduser.html
|   ├── attend.html
|   ├── index.html
|   ├── showattendance.html
|   ├── emotion_status.html
|   ├── index.js
|   ├── emotion_detector.py
|   └── face_re.py
├── server.mjs
├── image_process.py

```

Start the server to verify it works:

```bash
npm start
```

If everything is set up correctly, you should see "Server running on port 3000" in your console.

## Troubleshooting

### Common PyTorch Issues
- If you encounter CUDA-related errors, make sure you have the correct PyTorch version for your CUDA installation
- For CPU-only installation, use: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`


# About The "final_resemotenet.pth"
this is the emotion detection model based on ResEmoteNet model from "https://github.com/ArnabKumarRoy02/ResEmoteNet"
the dataset that got used is **RAF_DB** the public version on kaggle "https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset"
