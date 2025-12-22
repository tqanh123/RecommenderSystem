const mongoose = require('mongoose');

const ItemSchema = new mongoose.Schema({
    tenrec_id: { type: String, required: true, unique: true, index: true },
    model_index: { type: Number, required: true, index: true },
    title: { type: String, required: true },
    category: { type: String, index: true }, 
    category_id: { type: Number, default: 4 },
    type: { type: String, enum: ['video', 'article'], default: 'video' },
    embedding: { type: [Number], select: false },
    
    imageUrl: String,
    click_count: { type: Number, default: 0 },
    like_count: { type: Number, default: 0 },
    share_count: { type: Number, default: 0 },
});

// Tạo index text cho Title để làm chức năng Search
ItemSchema.index({ title: 'text' });

module.exports = mongoose.model('Item', ItemSchema);