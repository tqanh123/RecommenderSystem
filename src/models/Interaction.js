const mongoose = require('mongoose');

const InteractionSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    itemId: { type: mongoose.Schema.Types.ObjectId, ref: 'Item' },
    type: { type: String, enum: ['click', 'like', 'share', 'follow'], required: true },
    value: { type: Number, default: 1}, // Implicit feedback weight
    timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Interaction', InteractionSchema);
