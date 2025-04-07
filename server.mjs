import express from 'express';
import mongoose from 'mongoose';
import multer from 'multer';
import fs from 'fs';
import { spawn } from 'child_process';
import { Users } from "./database/users.mjs";
import { Attendance } from "./database/attendance.mjs";
import path from "path";

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static("public"));

// Create necessary directories
const tempDir = path.join("temp_images");
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
}

const userImagesDir = path.join("user_images");
if (!fs.existsSync(userImagesDir)) {
    fs.mkdirSync(userImagesDir, { recursive: true });
}

// Connect to MongoDB
mongoose.connect("mongodb://localhost:27017/project_4th")
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch(err => {
    console.error("MongoDB connection error:", err);
  });

// Get all users and serve the main page
app.get("/", async (req, res) => {
    try {
        const users = await Users.find({});
        console.log("Users fetched:", users);
        res.sendFile(__dirname + "/public/index.html");
    } catch (err) {
        console.error("Error fetching users:", err);
        res.status(500).send("Error fetching users");
    }
});

// Serve the attendance page
app.get("/attend", (req, res) => {
    res.sendFile(__dirname + "/public/attend.html");
});

// Serve the records page
app.get("/records", (req, res) => {
    res.sendFile(__dirname + "/public/records.html");
});

// API endpoint to get all users
app.get("/api/users", async (req, res) => {
    try {
        const users = await Users.find({});
        res.json(users);
    } catch (err) {
        console.error("Error fetching users:", err);
        res.status(500).json({ error: "Error fetching users" });
    }
});

// Create a new user
app.post("/", async(req, res) => {
    const name = req.body.name;
    try {
        const user = await Users.create({ name: name });
        await user.save();
        res.status(200).json({ success: true, user });
        console.log("User created:", user);
    } catch (err) {
        console.error("Error creating user:", err);
        res.status(500).json({ error: "Error creating user" });
    }
});

// Create an attendance record with duplication prevention
app.post("/attendance", async(req, res) => {
    const { UserID, date } = req.body;
    
    if (!UserID || !date) {
        return res.status(400).json({ 
            success: false, 
            message: "Missing required fields" 
        });
    }
    
    try {
        // Get the current date without time (start of the day)
        const today = new Date(date);
        today.setHours(0, 0, 0, 0);
        
        // End of the current day
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        console.log(`Checking attendance for user ${UserID} between ${today} and ${tomorrow}`);
        
        // Check if attendance already exists for today
        const existingAttendance = await Attendance.findOne({ 
            UserID: UserID, 
            Date: { 
                $gte: today, 
                $lt: tomorrow 
            } 
        });
        
        if (existingAttendance) {
            console.log(`Attendance already recorded for user ${UserID} today`);
            return res.status(409).json({ 
                success: false, 
                message: "Attendance already recorded for today" 
            });
        }
        
        // Create new attendance record
        const entry = await Attendance.create({ 
            UserID: UserID, 
            Date: date 
        });
        await entry.save();
        
        console.log(`Attendance recorded for user ${UserID}:`, entry);
        res.status(200).json({ success: true, entry });
    } catch (err) {
        console.error("Error recording attendance:", err);
        res.status(500).json({ error: "Error recording attendance" });
    }
});

// Get attendance records for a specific date
app.get("/attendance/:date", async(req, res) => {
    const date = req.params.date;
    try {
        const startDate = new Date(date);
        startDate.setHours(0, 0, 0, 0);
        
        const endDate = new Date(startDate);
        endDate.setDate(endDate.getDate() + 1);
        
        console.log(`Fetching attendance from ${startDate} to ${endDate}`);
        
        // Get attendance records without using populate
        const attendance = await Attendance.find({ 
            Date: { 
                $gte: startDate, 
                $lt: endDate 
            } 
        });
        
        // Map the results to include username directly
        const mappedAttendance = await Promise.all(attendance.map(async (record) => {
            // Use the UserID directly which is now a string (name)
            return {
                _id: record._id,
                UserID: record.UserID,
                Date: record.Date,
                createdAt: record.createdAt
            };
        }));
        
        res.json(mappedAttendance);
    } catch (err) {
        console.error("Error fetching attendance:", err);
        res.status(500).json({ error: "Error fetching attendance" });
    }
});

// Temporary storage for images
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // If it's a recognition request, store in a temp directory
        if (req.body && req.body.recognition === "true") {
            const tempDir = path.join("temp_images");
            // Ensure directory exists
            if (!fs.existsSync(tempDir)) {
                fs.mkdirSync(tempDir, { recursive: true });
            }
            cb(null, tempDir);
        } else if (req.body && req.body.user_id) {
            // For user registration images
            const userFolder = path.join("user_images", req.body.user_id);
            // Ensure directory exists
            if (!fs.existsSync(userFolder)) {
                fs.mkdirSync(userFolder, { recursive: true });
            }
            cb(null, userFolder);
        } else {
            // Fallback to temp directory
            const defaultDir = path.join("temp_images");
            // Ensure directory exists
            if (!fs.existsSync(defaultDir)) {
                fs.mkdirSync(defaultDir, { recursive: true });
            }
            cb(null, defaultDir);
        }
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        const ext = path.extname(file.originalname || 'unknown.jpg');
        cb(null, `face_image_${timestamp}${ext}`);
    }
});

const upload = multer({ storage });

// Directly recognize a face image
app.post("/recognize-face", upload.single("face_image"), (req, res) => {
    if (!req.file) {
        console.error("No file received for recognition");
        return res.status(400).json({ success: false, error: "No file uploaded" });
    }
    
    const imagePath = req.file.path;
    console.log(`Processing image for face recognition: ${imagePath}`);
    
    // Call face_rec.py to process the image (use 'python' or 'python3' as appropriate for your system)
    const pythonProcess = spawn("python", ["public/face_rec.py", imagePath]);
    
    let recognizedUser = "";
    let errorOutput = "";
    
    pythonProcess.stdout.on("data", (data) => {
        recognizedUser += data.toString().trim();
    });
    
    pythonProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
        console.error(`Python recognition error: ${data}`);
    });
    
    pythonProcess.on("close", (code) => {
        // Delete the temporary image after processing
        try {
            if (fs.existsSync(imagePath)) {
                fs.unlinkSync(imagePath);
                console.log(`Deleted temporary image: ${imagePath}`);
            }
        } catch (err) {
            console.error(`Error deleting temporary image: ${err}`);
        }
        
        if (code === 0 && recognizedUser && recognizedUser !== "Unknown") {
            console.log(`Face recognized as: ${recognizedUser}`);
            
            // Check if user already has attendance for today before responding
            checkTodayAttendance(recognizedUser)
                .then(hasAttendance => {
                    if (hasAttendance) {
                        console.log(`User ${recognizedUser} already has attendance for today`);
                        res.json({ 
                            success: true, 
                            recognizedUser: recognizedUser,
                            attendanceRecorded: false,
                            message: "Attendance already recorded for today"
                        });
                    } else {
                        res.json({ 
                            success: true, 
                            recognizedUser: recognizedUser,
                            attendanceRecorded: true
                        });
                    }
                })
                .catch(err => {
                    console.error(`Error checking attendance: ${err}`);
                    res.json({ 
                        success: true, 
                        recognizedUser: recognizedUser 
                    });
                });
        } else {
            console.log("Face recognition failed or unknown face detected");
            if (errorOutput) {
                console.error(`Error details: ${errorOutput}`);
            }
            res.json({ 
                success: false, 
                message: "Face not recognized", 
                error: errorOutput 
            });
        }
    });
});

// Helper function to check if a user already has attendance for today
async function checkTodayAttendance(userId) {
    try {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        
        const tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);
        
        const existingAttendance = await Attendance.findOne({ 
            UserID: userId, 
            Date: { 
                $gte: today, 
                $lt: tomorrow 
            } 
        });
        
        return existingAttendance !== null;
    } catch (error) {
        console.error(`Error checking today's attendance: ${error}`);
        return false;
    }
}

// Original upload endpoint for user registration Python recognition error
app.post("/upload", upload.single("face_image"), async (req, res) => {
    // Check if the file is received
    if (!req.file) {
        console.error("No file received");
        return res.status(400).json({ error: "No file uploaded" });
    }
    
    const { user_id } = req.body;
    const imagePath = req.file.path;
    
    console.log(`Received image for user ${user_id}: ${imagePath}`);
    
    if (!user_id) {
        // Delete the file if no user_id
        try {
            if (fs.existsSync(imagePath)) {
                fs.unlinkSync(imagePath);
            }
        } catch (err) {
            console.error(`Error deleting file: ${err}`);
        }
        return res.status(400).json({ error: "Missing user ID" });
    }
    
    // Ensure the folder structure is correct
    const userFolder = path.join("user_images", user_id);
    console.log("Checking user folder:", userFolder);
    
    // Check the contents of the folder to confirm the images are uploaded
    const uploadedFiles = fs.readdirSync(userFolder).filter(file => file.endsWith(".jpg"));
    console.log(`Uploaded files count: ${uploadedFiles.length}`);
    
    if (uploadedFiles.length >= 5) {
        // Call the Python script to process the 5 images
        const pythonProcess = spawn("python", ["process_image.py", userFolder, user_id]);
        
        pythonProcess.stdout.on("data", (data) => {
            console.log(`Python Output: ${data}`);
        });
        
        pythonProcess.stderr.on("data", (data) => {
            console.error(`Python Error: ${data}`);
        });
        
        pythonProcess.on("close", async (code) => {
            if (code === 0) {
                // Create the new user in the database after processing the images
                try {
                    // Check if user already exists
                    let newUser = await Users.findOne({ name: user_id });
                    if (!newUser) {
                        newUser = await Users.create({ name: user_id });
                        await newUser.save();
                        console.log("User added to the database:", newUser);
                    } else {
                        console.log("User already exists:", newUser);
                    }
                    
                    // Respond with success
                    res.json({ success: true, message: "Face processed, registration completed, and user added to the database" });
                } catch (err) {
                    console.error("Error adding user to database:", err);
                    res.status(500).json({ error: "Error adding user to database" });
                }
            } else {
                res.status(500).json({ error: "Face processing failed" });
            }
        });
    } else {
        res.json({ success: true, message: `Image uploaded. ${uploadedFiles.length}/5 images captured.` });
    }
});

app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server is running on http://127.0.0.1:${PORT}`);
});