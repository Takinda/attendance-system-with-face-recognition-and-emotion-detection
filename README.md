# Face Recognition System Installation Guide

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
mkdir face-recognition-backend
cd face-recognition-backend
```

2. Create a `package.json` file with the following content:

```json
{
  "name": "backend",
  "version": "1.0.0",
  "main": "server.mjs",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "start": "nodemon server.mjs"
  },
  "author": "",
  "license": "ISC",
  "description": "",
  "dependencies": {
    "child_process": "^1.0.2",
    "cors": "^2.8.5",
    "express": "^4.21.2",
    "fs": "^0.0.1-security",
    "mongoose": "^8.13.1",
    "multer": "^1.4.5-lts.2",
    "nodemon": "^3.1.9"
  }
}
```

3. Install the dependencies:

```bash
npm install
```

## Project Structure

Create the following basic structure for your project:

```
face-recognition-project/
│
├── backend/
│   ├── server.mjs
│   ├── package.json
│   └── node_modules/
│
└── python/
    ├── face_recognition.py
    ├── train_model.py
    └── utils.py
```

## Environment Setup Verification

After installation, verify your setup:

### Python Verification

Create a simple test script `test_python_deps.py`:

```python
import sys
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

# Print versions to verify
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print("FaceNet PyTorch and Pickle are installed")

# Test MTCNN and InceptionResnetV1 initialization
try:
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("Successfully initialized MTCNN and InceptionResnetV1")
except Exception as e:
    print(f"Error initializing models: {e}")
```

Run the script to verify all dependencies are installed correctly:

```bash
python test_python_deps.py
```

### Node.js Verification

Create a simple `server.mjs` file:

```javascript
import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import fs from 'fs';
import mongoose from 'mongoose';
import multer from 'multer';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('Face Recognition Backend is running!');
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
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

### Common Node.js Issues
- If nodemon isn't recognized, install it globally: `npm install -g nodemon`
- If you encounter permission issues on Linux/Mac, use `sudo` for global installations

## Next Steps

After successful installation, you can proceed with:
1. Implementing the face recognition scripts
2. Setting up the MongoDB database connection
3. Creating API endpoints for your application
4. Testing the complete system
