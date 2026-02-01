const User = require('../models/User');
const Item = require('../models/Item');
const Interaction = require('../models/Interaction');
const axios = require('axios');

class RecommenderService {
    constructor() {
        this.isReady = false;
        this.pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:5001';
        this.usePythonModel = process.env.USE_PYTHON_MODEL === 'true';
        this.checkPythonService();
    }

    /**
     * Check if Python inference service is available
     */
    async checkPythonService() {
        try {
            const response = await axios.get(`${this.pythonServiceUrl}/health`, { timeout: 2000 });
            if (response.data.model_loaded) {
                console.log('✅ Python LightGCN inference service is available');
                this.usePythonModel = true;
            }
        } catch (error) {
            console.log('⚠️  Python inference service not available, using fallback recommendations');
            this.usePythonModel = false;
        }
    }

    /**
     * Tính dot product giữa 2 vectors (user embedding · item embedding)
     */
    dotProduct(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) {
            return 0;
        }
        return vecA.reduce((sum, val, i) => sum + (val * vecB[i]), 0);
    }

    /**
     * Normalize vector về magnitude = 1
     */
    normalize(vec) {
        const magnitude = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
        return magnitude === 0 ? vec : vec.map(v => v / magnitude);
    }

    /**
     * Get recommendations for user (exclude interacted items)
     */
    async getRecommendations(userId, limit = 20) {
        try {
            // 1. Lấy thông tin user và embedding
            const user = await User.findById(userId).lean();
            if (!user) {
                console.log(`User ${userId} not found`);
                return this.getPopularItems(limit);
            }

            // 2. Try using Python LightGCN model if available
            if (this.usePythonModel) {
                try {
                    const recommendations = await this.getPythonRecommendations(userId, limit * 2); // Get more for filtering
                    if (recommendations && recommendations.length > 0) {
                        // Filter out interacted items
                        const interactions = await Interaction.find({ userId }).select('itemId').lean();
                        const interactedItemIds = interactions.map(i => i.itemId.toString());
                        
                        // Map Python item IDs to MongoDB items
                        const filteredRecs = recommendations.filter(rec => 
                            !interactedItemIds.includes(rec._id?.toString())
                        );
                        
                        console.log(`✅ Python model returned ${filteredRecs.length} recommendations for user ${userId}`);
                        return filteredRecs.slice(0, limit);
                    }
                } catch (error) {
                    console.log('⚠️  Python model failed, falling back to baseline:', error.message);
                }
            }

            // 3. Fallback: Use baseline recommendations
            // Lấy danh sách items user đã tương tác
            const interactions = await Interaction.find({ userId })
                .select('itemId')
                .lean();
            
            const interactedItemIds = interactions.map(i => i.itemId.toString());
            
            console.log(`User ${userId} has interacted with ${interactedItemIds.length} items`);

            // 3. Lấy ALL items chưa tương tác (bao gồm embedding)
            const candidateItems = await Item.find({
                _id: { $nin: interactedItemIds }
            })
            .select('+embedding') // Include embedding field (select: false by default)
            .lean();

            if (candidateItems.length === 0) {
                return [];
            }

            // 4. Kiểm tra xem user có embedding không
            const hasEmbedding = user.embedding && user.embedding.length > 0;
            
            if (!hasEmbedding) {
                console.log(`User ${userId} has no embedding, using popularity-based recommendation`);
                return this.getPopularityBasedRecommendations(candidateItems, limit);
            }

            // 5. Tính dot product cho mỗi item với user embedding
            const userEmbedding = this.normalize(user.embedding);
            
            const itemsWithScore = candidateItems.map(item => {
                let score;
                
                if (item.embedding && item.embedding.length > 0) {
                    // Tính dot product (cosine similarity vì đã normalize)
                    const itemEmbedding = this.normalize(item.embedding);
                    score = this.dotProduct(userEmbedding, itemEmbedding);
                } 
                // else {
                //     // Fallback: popularity score cho items không có embedding
                //     score = (item.like_count * 0.5 + item.click_count * 0.1 + (item.share_count || 0) * 0.3) / 100;
                // }
                
                return {
                    ...item,
                    embeddingScore: score
                };
            });

            // 6. Tìm min/max score để normalize về 60-95%
            const scores = itemsWithScore.map(i => i.embeddingScore);
            const maxScore = Math.max(...scores);
            const minScore = Math.min(...scores);
            const scoreRange = maxScore - minScore || 1;

            // 7. Tính % dự đoán dựa trên embedding similarity
            const itemsWithPrediction = itemsWithScore.map(item => {
                // Normalize score to 0-1 range, then map to 60-95%
                const normalized = (item.embeddingScore - minScore) / scoreRange;
                const prediction = 60 + (normalized * 35); // 60% + (0-35%)
                
                return {
                    ...item,
                    predictionScore: Math.round(prediction)
                };
            });

            // 8. Sắp xếp theo % dự đoán GIẢM DẦN và lấy top items
            itemsWithPrediction.sort((a, b) => b.predictionScore - a.predictionScore);
            const topRecommendations = itemsWithPrediction.slice(0, limit);

            console.log(`Found ${topRecommendations.length} recommended items using embedding similarity`);

            return topRecommendations;
        } catch (error) {
            console.error('Error getting recommendations:', error);
            // Fallback: return popular items
            return this.getPopularItems(limit);
        }
    }

    /**
     * Popularity-based recommendations (fallback when no embeddings)
     */
    getPopularityBasedRecommendations(candidateItems, limit) {
        // Score = (like_count * 5) + (click_count * 1) + (share_count * 3)
        const itemsWithScore = candidateItems.map(item => {
            const score = (item.like_count * 5) + (item.click_count * 1) + ((item.share_count || 0) * 3);
            return {
                ...item,
                popularityScore: score
            };
        });

        const maxScore = Math.max(...itemsWithScore.map(i => i.popularityScore));
        const minScore = Math.min(...itemsWithScore.map(i => i.popularityScore));
        const scoreRange = maxScore - minScore || 1;

        const itemsWithPrediction = itemsWithScore.map(item => {
            const normalized = (item.popularityScore - minScore) / scoreRange;
            const prediction = 60 + (normalized * 35);
            
            return {
                ...item,
                predictionScore: Math.round(prediction)
            };
        });

        itemsWithPrediction.sort((a, b) => b.predictionScore - a.predictionScore);
        return itemsWithPrediction.slice(0, limit);
    }

    /**
     * Get popular items (fallback)
     */
    async getPopularItems(limit = 20) {
        try {
            const items = await Item.find()
                .sort({ like_count: -1, click_count: -1 })
                .limit(limit)
                .lean();
            
            return items;
        } catch (error) {
            console.error('Error getting popular items:', error);
            return [];
        }
    }

    /**
     * Get recommendations from Python LightGCN model
     */
    async getPythonRecommendations(userId, limit = 20) {
        try {
            const response = await axios.get(`${this.pythonServiceUrl}/recommend/${userId}`, {
                params: { k: limit },
                timeout: 5000
            });

            if (!response.data.success) {
                throw new Error(response.data.error || 'Python service error');
            }

            // Map Python recommendations to MongoDB items
            const itemIds = response.data.recommendations.map(r => r.item_id);
            const items = await Item.find({ item_id: { $in: itemIds } })
                .select('+embedding')
                .lean();

            // Attach prediction scores from Python model
            const itemsWithScores = items.map(item => {
                const rec = response.data.recommendations.find(r => r.item_id === item.item_id);
                const score = rec ? rec.score : 0;
                
                // Map score to prediction percentage (60-95%)
                const normalized = (score + 1) / 2; // Assume scores are roughly in [-1, 1]
                const prediction = Math.round(60 + (normalized * 35));
                
                return {
                    ...item,
                    embeddingScore: score,
                    predictionScore: Math.min(95, Math.max(60, prediction))
                };
            });

            // Sort by score (highest first)
            itemsWithScores.sort((a, b) => b.embeddingScore - a.embeddingScore);

            return itemsWithScores;
        } catch (error) {
            console.error('Error getting Python recommendations:', error.message);
            throw error;
        }
    }

    /**
     * Update user embedding (placeholder for future ML integration)
     */
    async updateUserEmbedding(userId, itemId, interactionType) {
        console.log(`[Interaction] User ${userId} -> Item ${itemId} (${interactionType})`);
        
        try {
            // Fetch user and item with embeddings
            const user = await User.findById(userId).lean();
            const item = await Item.findById(itemId).select('+embedding').lean();
            
            if (!user || !item || !user.embedding || !item.embedding) {
                console.log('Missing embeddings, skipping update');
                return false;
            }
            
            // Simulate embedding update: move user embedding closer to item embedding
            // Learning rate based on interaction type
            const learningRate = {
                'click': 0.01,
                'like': 0.05,
                'share': 0.03
            }[interactionType] || 0.01;
            
            const updatedEmbedding = user.embedding.map((userVal, idx) => {
                const itemVal = item.embedding[idx] || 0;
                // Gradient descent: move user embedding towards item embedding
                return userVal + learningRate * (itemVal - userVal);
            });
            
            // Update user embedding in database
            await User.findByIdAndUpdate(userId, {
                embedding: updatedEmbedding
            });
            
            console.log(`✅ User embedding updated (lr=${learningRate}, interaction=${interactionType})`);
            return true;
        } catch (error) {
            console.error('Error updating user embedding:', error);
            return false;
        }
    }
}

// Singleton export
module.exports = new RecommenderService();

exports.updateUserEmbedding = async (userId, itemId, interactionType) => {
    console.log(`[LightGCN] Updating gradient for user ${userId} based on ${interactionType}`);
    // Logic to invoke Python training step or update vector cache
    return true;
};
