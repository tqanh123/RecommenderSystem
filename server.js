require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const apiRoutes = require('./src/routes/api');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Database Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/tenrecrec', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('✅ MongoDB Connected'))
.catch(err => console.error('❌ DB Connection Error:', err));

// Routes
app.use('/api', apiRoutes);

// Serve login page
app.get('/login.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'html/login.html'));
});

// Fallback for SPA (main app)
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'html/index.html'));
});

// Start Server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`🧠 LightGCN Model Interface Ready`);
});
