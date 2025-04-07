import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
        trim: true,
        unique: true // Ensure the name is unique at the schema level
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

export const Users = mongoose.model("Users", userSchema);
