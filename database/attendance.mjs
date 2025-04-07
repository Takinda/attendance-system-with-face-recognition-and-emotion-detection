import mongoose from "mongoose";

// Modify the attendance schema to reference Users by name (instead of ObjectId)
const attendanceSchema = new mongoose.Schema({
    UserID: {
        type: String, // Store the name as a string
        ref: "Users", // Reference the "Users" model
        required: true
    },
    Date: {
        type: Date,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

export const Attendance = mongoose.model("Attendance", attendanceSchema);
