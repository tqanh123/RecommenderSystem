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

// // User Interaction
// app.post('/api/interact', requireLogin, async (req, res) => {
//     const { item_id, action_type } = req.body;
//     const user = req.session.user;

//     try {
//         // BƯỚC 1: Lưu Log chi tiết (Để dành cho AI train sau này)
//         // Việc này có thể chạy ngầm (không cần await nếu muốn response nhanh)
//         const interactionLog = new Interaction({
//             user_id: user._id,
//             tenrec_user_id: user.tenrec_user_id,
//             item_id: item_id,
//             action_type: action_type,
//             timestamp: new Date()
//         });
//         await interactionLog.save();

//         // BƯỚC 2: Tăng biến đếm ngay lập tức (Real-time Counter)
//         // Xác định trường nào cần tăng dựa vào action_type
//         let updateQuery = {};
        
//         switch (action_type) {
//             case 'click':
//                 updateQuery = { $inc: { "metrics.click_count": 1 } };
//                 break;
//             case 'like':
//                 updateQuery = { $inc: { "metrics.like_count": 1 } };
//                 break;
//             case 'share':
//                 updateQuery = { $inc: { "metrics.share_count": 1 } };
//                 break;
//             case 'follow':
//                 updateQuery = { $inc: { "metrics.follow_count": 1 } };
//                 break;
//             default:
//                 return res.status(400).json({ error: "Invalid action type" });
//         }

//         // Thực hiện update trực tiếp vào DB
//         await Item.updateOne(
//             { item_id: item_id }, // Điều kiện tìm
//             updateQuery           // Lệnh update ($inc)
//         );

//         res.json({ status: 'success', message: 'Interaction recorded' });

//     } catch (err) {
//         console.error("Interaction Error:", err);
//         res.status(500).json({ status: 'error' });
//     }
// });


// // Trend list item
// app.get('/api/popular-items', async (req, res) => {
//     try {
//         // Lấy 15 sản phẩm có view cao nhất (click_count hoặc like_count)
//         // Lưu ý: Đảm bảo trong DB bạn đã có field metrics.click_count như bài trước
//         const popularItems = await Item.find({})
//             .sort({ "metrics.click_count": -1 }) // Sắp xếp giảm dần
//             .limit(7);
            
//         res.json(popularItems);
//     } catch (err) {
//         res.status(500).json({ error: err.message });
//     }
// });

// Start Server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`🧠 LightGCN Model Interface Ready`);
});
