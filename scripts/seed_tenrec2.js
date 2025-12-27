// scripts/seed_tenrec.js
const mongoose = require('mongoose');
const fs = require('fs').promises;
const path = require('path');
const User = require('../src/models/User');
const Item = require('../src/models/Item');
const { title } = require('process');

const MONGO_URI = 'mongodb://localhost:27017/tenrecrec';

const seedData = async () => {
    try {
        await mongoose.connect(MONGO_URI);
        console.log('🔗 Connected to MongoDB');

        // 1. Đọc file JSON từ Python export
        console.log('📂 Reading JSON files...');
        const usersData = JSON.parse(
            await fs.readFile(path.join(__dirname, '../data/user_data_full.json'), 'utf8')
        );
        const itemsData = JSON.parse(
            await fs.readFile(path.join(__dirname, '../data/item_data_full.json'), 'utf8')
        );

        // 2. Xóa dữ liệu cũ (Reset)
        await User.deleteMany({});
        await Item.deleteMany({});
        console.log('🧹 Cleared old data');
        

        // 3. Import Items
        // Map cat to get img
        const catgoryMap = {
            'Gaming':     'videogames,esports', 
            'Music':      'concert,music,band', 
            'Technology': 'tech,computer,coding',
            'Lifestyle':  'vlog,travel,people', 
            'Others':     'abstract,variety'    
        }
        const itemsToInsert = itemsData.map(item => {
            const catId = item.category_id;
            const displayCategory = Object.keys(catgoryMap)[catId] || 'Others';
            const imgKeywords = catgoryMap[displayCategory];
            
            return {
                tenrec_id: item.tenrec_id,
                model_index: item.model_index,
                category: displayCategory,
                category_id: catId,
                embedding: item.embedding,
                type: 'video',
                imageUrl: `https://source.unsplash.com/400x225/?${imgKeywords}&lock=${item.model_index}`,
                click_count: item.click_count || 0,
                like_count: item.like_count || 0,
                share_count: item.share_count || 0,
                title: 'video '+item.model_index
            };
        });

        // Insert theo lô (Batch) để tránh quá tải nếu file quá lớn
        await Item.insertMany(itemsToInsert);
        console.log(`✅ Seeded ${itemsToInsert.length} items`);

        // 4. Import Users
        const usersToInsert = usersData.map(user => ({
            username: `user_${user.tenrec_uid}`, // Tạo username từ ID gốc
            tenrec_uid: user.tenrec_uid,
            model_index: user.model_index, // Số 0, 1, 2... lưu vào DB
            embedding: user.embedding,
            preferences: { music: 0.1, Technology: 0.1 } // Default
        }));

        await User.insertMany(usersToInsert);
        console.log(`✅ Seeded ${usersToInsert.length} users`);

        // --- 8. INSERT INTERACTIONS (Phần bạn tìm kiếm) ---
        console.log('🔗 Linking & Importing Interactions...');
        
        // Tạo Map để tra cứu: model_index -> MongoDB ObjectId
        // Interaction file sử dụng model_index (user_idx, item_idx)
        const userMap = {};
        createdUsers.forEach(u => { userMap[u.model_index] = u._id; });

        const itemMap = {};
        createdItems.forEach(i => { itemMap[i.model_index] = i._id; });

        console.log(`   -> UserMap size: ${Object.keys(userMap).length}`);
        console.log(`   -> ItemMap size: ${Object.keys(itemMap).length}`);

        // Debug: Check first few interactions
        const firstInter = demoInteractions[0];
        console.log(`   -> Sample interaction: user_idx=${firstInter.user_idx}, item_idx=${firstInter.item_idx}`);
        console.log(`   -> User exists in map: ${userMap[firstInter.user_idx] !== undefined}`);
        console.log(`   -> Item exists in map: ${itemMap[firstInter.item_idx] !== undefined}`);
        
        // Debug: Show sample keys
        const userKeys = Object.keys(userMap).slice(0, 5);
        const itemKeys = Object.keys(itemMap).slice(0, 5);
        console.log(`   -> Sample user keys: ${userKeys.join(', ')}`);
        console.log(`   -> Sample item keys: ${itemKeys.join(', ')}`);

        // Tạo danh sách Interaction để insert
        const interactionsToInsert = demoInteractions.map(inter => {
            const userObjectId = userMap[inter.user_idx];
            const itemObjectId = itemMap[inter.item_idx];

            // Chỉ insert nếu cả User và Item đều đã được tạo thành công trong DB
            if (userObjectId && itemObjectId) {
                return {
                    userId: userObjectId, // <--- Link tới User
                    itemId: itemObjectId, // <--- Link tới Item
                    type: inter.type || 'click',
                    value: inter.type === 'like' ? 5 : (inter.type === 'share' ? 3 : 1), // Like=5, Share=3, Click=1
                    timestamp: new Date()
                };
            }
            return null;
        }).filter(Boolean); // Loại bỏ null

        if (interactionsToInsert.length > 0) {
            const createdInteractions = await Interaction.insertMany(interactionsToInsert);
            console.log(`✅ Successfully inserted ${createdInteractions.length} interactions.`);
        } else {
            console.log('⚠️ No interactions matched with inserted users/items.');
        }

        // 6. Thống kê
        console.log('\n📊 Database Statistics:');
        console.log(`   Users: ${createdUsers.length}`);
        console.log(`   Items: ${createdItems.length}`);
        console.log(`   Interactions: ${interactionsToInsert.length}`);
        
        console.log('🎉 DONE! Database is synchronized with LightGCN Model.');
        process.exit();

    } catch (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }
};

seedData();