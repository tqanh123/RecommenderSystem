// scripts/seed_demo.js - Seed database with demo data
const mongoose = require('mongoose');
const User = require('../src/models/User');
const Item = require('../src/models/Item');
const Interaction = require('../src/models/Interaction');

const MONGO_URI = 'mongodb://localhost:27017/tenrecrec';
const NUM_USERS = 50;
const NUM_ITEMS = 100;
const NUM_INTERACTIONS = 300;

const seedDemoData = async () => {
    try {
        await mongoose.connect(MONGO_URI);
        console.log('🔗 Connected to MongoDB');

        // Xóa dữ liệu cũ
        await User.deleteMany({});
        await Item.deleteMany({});
        await Interaction.deleteMany({});
        console.log('🧹 Cleared old data');

        // Tạo Users
        console.log(`👥 Creating ${NUM_USERS} demo users...`);
        const users = [];
        for (let i = 0; i < NUM_USERS; i++) {
            users.push({
                username: `demo_user_${i}`,
                tenrec_uid: `uid_${i}`,
                model_index: i,
                embedding: Array(64).fill(0).map(() => Math.random()),
                preferences: { music: Math.random(), technology: Math.random() }
            });
        }
        const createdUsers = await User.insertMany(users);
        console.log(`✅ Created ${createdUsers.length} users`);

        // Tạo Items
        console.log(`📦 Creating ${NUM_ITEMS} demo items...`);
        const categoryMap = {
            0: 'Gaming',
            1: 'Music',
            2: 'Technology',
            3: 'Lifestyle',
            4: 'Others'
        };
        
        const categoryImages = {
            'Gaming': 'videogames,esports',
            'Music': 'concert,music,band',
            'Technology': 'tech,computer,coding',
            'Lifestyle': 'vlog,travel,people',
            'Others': 'abstract,variety'
        };

        const items = [];
        for (let i = 0; i < NUM_ITEMS; i++) {
            const catId = i % 5;
            const category = categoryMap[catId];
            const imgKeywords = categoryImages[category];
            
            items.push({
                tenrec_id: `tid_${i}`,
                model_index: i,
                category: category,
                category_id: catId,
                embedding: Array(64).fill(0).map(() => Math.random()),
                type: 'video',
                imageUrl: `https://picsum.photos/seed/${i}/400/225`,
                click_count: Math.floor(Math.random() * 1000),
                like_count: Math.floor(Math.random() * 200),
                share_count: Math.floor(Math.random() * 50),
                title: `Demo Video ${i} - ${category}`
            });
        }
        const createdItems = await Item.insertMany(items);
        console.log(`✅ Created ${createdItems.length} items`);

        // Tạo Interactions
        console.log(`🔗 Creating ${NUM_INTERACTIONS} demo interactions...`);
        const interactions = [];
        const interactionTypes = ['click', 'like', 'share'];
        
        for (let i = 0; i < NUM_INTERACTIONS; i++) {
            const randomUser = createdUsers[Math.floor(Math.random() * createdUsers.length)];
            const randomItem = createdItems[Math.floor(Math.random() * createdItems.length)];
            const type = interactionTypes[Math.floor(Math.random() * interactionTypes.length)];
            
            interactions.push({
                userId: randomUser._id,
                itemId: randomItem._id,
                type: type,
                value: type === 'like' ? 5 : (type === 'share' ? 3 : 1),
                timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000) // Random trong 30 ngày
            });
        }
        
        const createdInteractions = await Interaction.insertMany(interactions);
        console.log(`✅ Created ${createdInteractions.length} interactions`);

        // Thống kê
        console.log('\n📊 Database Statistics:');
        console.log(`   Users: ${createdUsers.length}`);
        console.log(`   Items: ${createdItems.length}`);
        console.log(`   Interactions: ${createdInteractions.length}`);
        
        // Thống kê theo loại interaction
        const clickCount = createdInteractions.filter(i => i.type === 'click').length;
        const likeCount = createdInteractions.filter(i => i.type === 'like').length;
        const shareCount = createdInteractions.filter(i => i.type === 'share').length;
        
        console.log('\n📈 Interaction Breakdown:');
        console.log(`   Clicks: ${clickCount}`);
        console.log(`   Likes: ${likeCount}`);
        console.log(`   Shares: ${shareCount}`);

        console.log('\n🎉 DONE! Demo database is ready.');
        console.log('   You can now run: node server.js');
        console.log('   And visit: http://localhost:3000');
        
        await mongoose.connection.close();
        process.exit(0);

    } catch (err) {
        console.error('❌ Error:', err);
        process.exit(1);
    }
};

seedDemoData();
