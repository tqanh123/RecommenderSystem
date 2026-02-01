const express = require('express');
const router = express.Router();
const User = require('../models/User');
const Item = require('../models/Item');
const Interaction = require('../models/Interaction');
const interactions = await Interaction.find().lean();
const recommender = require('../services/recommender');

/**
 * GET /api/items - Get all items
 */
router.get('/items', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        const category = req.query.category;
        const includeEmbedding = req.query.embedding === 'true';
        
        let query = {};
        if (category) {
            query.category = category;
        }

        let itemsQuery = Item.find(query)
            .limit(limit)
            .sort({ createdAt: -1 });
        
        // Include embedding if requested
        if (includeEmbedding) {
            itemsQuery = itemsQuery.select('+embedding');
        }
        
        const items = await itemsQuery;

        res.json({
            success: true,
            count: items.length,
            items: items
        });
    } catch (error) {
        console.error('Error fetching items:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * GET /api/users - Get all users
 */
router.get('/users', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 100;
        const includeEmbedding = req.query.embedding === 'true';
        
        let query = User.find()
            .limit(limit)
            .sort({ createdAt: -1 });
        
        // Include embedding if requested
        if (includeEmbedding) {
            query = query.select('username tenrec_uid model_index embedding');
        } else {
            query = query.select('username tenrec_uid model_index');
        }
        
        const users = await query;

        res.json({
            success: true,
            count: users.length,
            users: users
        });
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * GET /api/items/:id - Get single item
 */
router.get('/items/:id', async (req, res) => {
    try {
        const item = await Item.findById(req.params.id);
        
        if (!item) {
            return res.status(404).json({
                success: false,
                error: 'Item not found'
            });
        }

        res.json({
            success: true,
            item: item
        });
    } catch (error) {
        console.error('Error fetching item:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get Recommendations Endpoint
router.get('/recommend/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const limit = parseInt(req.query.limit) || 10;

        console.log(`Getting recommendations for user: ${userId}`);
        
        const recommendations = await recommender.getRecommendations(userId, limit);

        res.json({
            success: true,
            userId,
            count: recommendations.length,
            recommendations
        });
    } catch (error) {
        console.error('Recommendation error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Login / Init Session
router.post('/login', async (req, res) => {
    const { username } = req.body;
    let user = await User.findOne({ username });
    if (!user) {
        user = await User.create({ username });
    }
    res.json({ success: true, user });
});

// Log Interaction
router.post('/interact', async (req, res) => {
    try {
        const { userId, itemId, type } = req.body;
        
        // Save to DB
        await Interaction.create({ userId, itemId, type, timestamp: new Date() });
        
        // Trigger real-time embedding update (Simplified)
        await recommender.updateUserEmbedding(userId, itemId, type);
        
        res.json({ success: true, message: 'Interaction logged' });
    } catch (error) {
        console.error('Error logging interaction:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get user's interactions
router.get('/interactions/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        
        const interactions = await Interaction.find({ userId })
            .sort({ timestamp: -1 })
            .lean();
        
        // Convert ObjectIds to strings for frontend
        const formattedInteractions = interactions.map(interaction => ({
            ...interaction,
            itemId: interaction.itemId.toString(),
            userId: interaction.userId.toString()
        }));
        
        res.json({
            success: true,
            count: formattedInteractions.length,
            interactions: formattedInteractions
        });
    } catch (error) {
        console.error('Error fetching interactions:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get interactions for a specific item (all users who interacted with it)
router.get('/interactions/item/:itemId', async (req, res) => {
    try {
        const { itemId } = req.params;
        
        const interactions = await Interaction.find({ itemId })
            .sort({ timestamp: -1 })
            .lean();
        
        // Convert ObjectIds to strings for frontend
        const formattedInteractions = interactions.map(interaction => ({
            ...interaction,
            itemId: interaction.itemId.toString(),
            userId: interaction.userId.toString()
        }));
        
        res.json({
            success: true,
            count: formattedInteractions.length,
            interactions: formattedInteractions
        });
    } catch (error) {
        console.error('Error fetching item interactions:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Export Data for Retraining
router.get('/export-data', async (req, res) => {
    const data = await Interaction.find().sort({ timestamp: -1 }).limit(10000);
    res.json(data);
});

module.exports = router;
