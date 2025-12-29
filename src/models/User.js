const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true },
    tenrec_uid: { type: String }, // Original Tenrec user ID
    model_index: { type: Number }, // Index used in LightGCN model (0, 1, 2...)
    preferences: {
        gaming: { type: Number, default: 0.1 },
        music: { type: Number, default: 0.1 },
        technology: { type: Number, default: 0.1 },
        lifestyle: { type: Number, default: 0.1 },
        others: { type: Number, default: 0.1 },
    },
    gender: { type: Number }, 
    age: { type: Number },
    embedding: [Number], // Stored vector from LightGCN
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('User', UserSchema);
