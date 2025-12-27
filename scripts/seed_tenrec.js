// scripts/seed_tenrec.js
const mongoose = require('mongoose');
const fs = require('fs').promises;
const path = require('path');
const User = require('../src/models/User');
const Item = require('../src/models/Item');
const Interaction = require('../src/models/Interaction');

const MONGO_URI = 'mongodb://localhost:27017/tenrecrec';
const DEMO_LIMIT = 300; // Số interaction để demo

const seedData = async () => {
    try {
        await mongoose.connect(MONGO_URI);
        console.log('🔗 Connected to MongoDB');

        // 1. Đọc file JSON từ Python export
        console.log('📂 Reading JSON files...');
        const allUsers = JSON.parse(
            await fs.readFile(path.join(__dirname, '../framework/data/user_data_full.json'), 'utf8')
        );
        const allItems = JSON.parse(
            await fs.readFile(path.join(__dirname, '../framework/data/item_data_full.json'), 'utf8')
        );

        // 2. Xóa dữ liệu cũ (Reset)
        await User.deleteMany({});
        await Item.deleteMany({});
        await Interaction.deleteMany({});
        console.log('🧹 Cleared old data');

        // Load Interaction (Nếu file không tồn tại hoặc lỗi thì mảng rỗng)
        let allInteractions = [];
        try {
            allInteractions = JSON.parse(await fs.readFile(path.join(__dirname, '../framework/data/interaction_data.json'), 'utf8'));
        } catch (e) {
            console.log('⚠️ Warning: No interaction data found.');
        }

        console.log(`⚡ Filtering subset for Demo (${DEMO_LIMIT} interactions)...`);
        console.log(`   -> Total interactions available: ${allInteractions.length}`);

        // Lấy 300 dòng đầu tiên (KHÔNG shuffle để đảm bảo consistency)
        const demoInteractions = allInteractions.slice(0, DEMO_LIMIT);

        // Tìm danh sách User Index và Item Index xuất hiện trong 300 dòng này
        const activeUserIndices = new Set(demoInteractions.map(i => i.user_idx));
        const activeItemIndices = new Set(demoInteractions.map(i => i.item_idx));

        console.log(`   -> Have ${activeUserIndices.size} unique users and ${activeItemIndices.size} unique items`);

        // Lọc ra các User và Item tương ứng
        // Interaction file sử dụng model_index (user_idx, item_idx)
        const demoUsers = allUsers.filter(u => activeUserIndices.has(u.model_index));
        const demoItems = allItems.filter(i => activeItemIndices.has(i.model_index));

        console.log(`   -> Found ${demoUsers.length} Users and ${demoItems.length} Items involved.`);
        // Print first 5 demo users for debugging
        console.log('\n📋 Sample of first 5 demo users:');
        demoUsers.slice(0, 5).forEach((user, idx) => {
            console.log(`   ${idx + 1}. tenrec_uid: ${user.tenrec_uid}, model_index: ${user.model_index}`);
        });
        
        // Debug: Check if we found all required users and items
        console.log(`   -> Expected users: ${activeUserIndices.size}, Found: ${demoUsers.length}`);
        console.log(`   -> Expected items: ${activeItemIndices.size}, Found: ${demoItems.length}`);
        
        if (demoUsers.length === 0 || demoItems.length === 0) {
            console.error('❌ ERROR: No users or items found! Check data files.');
            process.exit(1);
        }

        // 3. Import Items
        // Map cat to get img
        const categoryMap = {
            'Gaming':     'videogames,esports', 
            'Music':      'concert,music,band', 
            'Technology': 'tech,computer,coding',
            'Lifestyle':  'vlog,travel,people', 
            'Others':     'abstract,variety'    
        };
        const itemsToInsert = demoItems.map(item => {
            const catId = item.category_id;
            const displayCategory = Object.keys(categoryMap)[catId] || 'Others';
            const imgKeywords = categoryMap[displayCategory];
            
            return {
                tenrec_id: item.tenrec_id,
                model_index: item.model_index,
                category: displayCategory,
                category_id: catId,
                embedding: item.embedding,
                type: 'video',
                imageUrl: `https://picsum.photos/seed/${item.model_index}/400/225`,
                click_count: item.click_count || 0,
                like_count: item.like_count || 0,
                share_count: item.share_count || 0,
                title: `Video ${item.model_index}`
            };
        });

        // Insert theo lô (Batch) để tránh quá tải nếu file quá lớn
        const createdItems = await Item.insertMany(itemsToInsert);
        console.log(`✅ Seeded ${createdItems.length} items`);

        // 4. Import Users
        const usersToInsert = demoUsers.map(user => ({
            username: `user_${user.tenrec_uid}`, // Tạo username từ ID gốc
            tenrec_uid: user.tenrec_uid,
            model_index: user.model_index, // Số 0, 1, 2... lưu vào DB
            embedding: user.embedding,
            preferences: { music: 0.1, Technology: 0.1 } // Default
        }));

        const createdUsers = await User.insertMany(usersToInsert);
        console.log(`✅ Seeded ${createdUsers.length} users`);

        // Debug: Check for duplicate model_index
        const userModelIndices = createdUsers.map(u => u.model_index);
        const uniqueUserIndices = new Set(userModelIndices);
        console.log(`   -> Unique user model_index count: ${uniqueUserIndices.size} (should be ${createdUsers.length})`);
        
        if (uniqueUserIndices.size !== createdUsers.length) {
            console.log('   ⚠️ WARNING: Duplicate model_index found in users!');
        }

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

        console.log('\n🎉 DONE! Database is synchronized with LightGCN Model.');
        
        await mongoose.connection.close();
        process.exit(0);

    } catch (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }
};

seedData();