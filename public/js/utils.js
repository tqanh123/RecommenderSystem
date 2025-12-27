function redrawCanvas() {
    if (!ctx || !canvas) return;
    
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    
    // Clear canvas
    ctx.clearRect(0, 0, w, h);
    
    // Background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, w, h);
    
    if (!embeddingData.user) return;
    
    // Combine all items: interacted + recommended
    const allItems = [
        ...embeddingData.interactedItems,
        ...embeddingData.items
    ];
    
    // Remove duplicates by _id
    const uniqueItems = [];
    const seenIds = new Set();
    for (const item of allItems) {
        if (!seenIds.has(item._id)) {
            uniqueItems.push(item);
            seenIds.add(item._id);
        }
    }
    
    if (uniqueItems.length === 0) return;
    
    // Combine all users: current user + other users
    const allUsers = [embeddingData.user, ...embeddingData.otherUsers];
    
    // Prepare all embeddings for PCA (users first, then items)
    const allEmbeddings = [
        ...allUsers.map(user => user.embedding),
        ...uniqueItems.map(item => item.embedding)
    ];
    
    // Reduce to 2D
    const points2D = pcaReduce(allEmbeddings);
    const normalized = normalizePoints(points2D);
    
    const userPositions = normalized.slice(0, allUsers.length);
    const itemPositions = normalized.slice(allUsers.length);
    
    const currentUserPos = userPositions[0];
    
    // Create map of user positions
    const userPosMap = new Map();
    allUsers.forEach((user, i) => {
        userPosMap.set(user._id.toString(), userPositions[i]);
    });
    
    // Create map of item positions
    const itemPosMap = new Map();
    uniqueItems.forEach((item, i) => {
        itemPosMap.set(item._id.toString(), itemPositions[i]);
    });
    
    console.log(`🗺️ Item position map has ${itemPosMap.size} items`);
    console.log(`🗺️ User position map has ${userPosMap.size} users`);
    console.log(`🔗 Drawing ${embeddingData.interactions.length} current user interactions`);
    console.log(`🔗 Drawing ${embeddingData.otherUsersInteractions.length} other user interactions`);
    
    // Draw other users' interaction lines first (lighter color, behind)
    embeddingData.otherUsersInteractions.forEach(interaction => {
        const userPos = userPosMap.get(interaction.userId.toString());
        const itemPos = itemPosMap.get(interaction.itemId.toString());
        
        if (!userPos || !itemPos) return;
        
        ctx.beginPath();
        ctx.moveTo(userPos.x, userPos.y);
        ctx.lineTo(itemPos.x, itemPos.y);
        
        // Lighter colors for other users
        if (interaction.type === 'like') {
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
            ctx.lineWidth = 1.5;
        } else if (interaction.type === 'share') {
            ctx.strokeStyle = 'rgba(168, 85, 247, 0.3)';
            ctx.lineWidth = 1.5;
        } else {
            ctx.strokeStyle = 'rgba(244, 114, 182, 0.3)';
            ctx.lineWidth = 1;
        }
        
        ctx.stroke();
    });
    
    // Draw current user's interaction lines (on top, bright colors)
    embeddingData.interactions.forEach(interaction => {
        const itemPos = itemPosMap.get(interaction.itemId.toString());
        if (!itemPos) {
            console.warn(`⚠️ Item position not found for interaction: ${interaction.itemId}`);
            return;
        }
        
        ctx.beginPath();
        ctx.moveTo(currentUserPos.x, currentUserPos.y);
        ctx.lineTo(itemPos.x, itemPos.y);
        
        // Color based on interaction type
        if (interaction.type === 'like') {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2.5;
        } else if (interaction.type === 'share') {
            ctx.strokeStyle = '#a855f7';
            ctx.lineWidth = 2;
        } else {
            ctx.strokeStyle = '#f472b6';
            ctx.lineWidth = 1.5;
        }
        
        ctx.stroke();
    });
    
    // Create set of interacted item IDs
    const interactedIds = new Set(embeddingData.interactedItems.map(i => i._id.toString()));
    
    // Draw item points
    uniqueItems.forEach((item, i) => {
        const pos = itemPositions[i];
        const isInteracted = interactedIds.has(item._id.toString());
        
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isInteracted ? 5 : 4, 0, Math.PI * 2);
        
        // Different colors for interacted vs recommended
        if (isInteracted) {
            ctx.fillStyle = '#f59e0b'; // Orange for interacted
            ctx.fill();
            ctx.strokeStyle = '#d97706';
            ctx.lineWidth = 1.5;
        } else {
            ctx.fillStyle = '#10b981'; // Green for recommended
            ctx.fill();
            ctx.strokeStyle = '#059669';
            ctx.lineWidth = 1;
        }
        ctx.stroke();
    });
    
    // Draw other user points (smaller, gray)
    embeddingData.otherUsers.forEach((user, i) => {
        const pos = userPositions[i + 1]; // +1 because current user is at index 0
        
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#94a3b8'; // Gray for other users
        ctx.fill();
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    
    // Draw current user point (larger, on top)
    ctx.beginPath();
    ctx.arc(currentUserPos.x, currentUserPos.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#1d4ed8';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Label
    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 11px Inter';
    ctx.fillText('You', currentUserPos.x + 12, currentUserPos.y + 4);
}

/**
 * Calculate dot product between two vectors
 */
function calculateDotProduct(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    return vecA.reduce((sum, val, i) => sum + (val * vecB[i]), 0);
}

/**
 * Update stats display
 */
function updateStats() {
    document.getElementById('interaction-count').textContent = interactionCount;
    document.getElementById('other-users-count').textContent = embeddingData.otherUsers.length;
    const totalItems = embeddingData.interactedItems.length + embeddingData.items.length;
    // Remove duplicates for accurate count
    const allItemIds = new Set([
        ...embeddingData.interactedItems.map(i => i._id),
        ...embeddingData.items.map(i => i._id)
    ]);
    document.getElementById('items-count').textContent = allItemIds.size;
}

/**
 * Clear visualization
 */
function clearVisualization() {
    embeddingData.interactions = [];
    embeddingData.interactedItems = [];
    embeddingData.otherUsers = [];
    embeddingData.otherUsersInteractions = [];
    interactionCount = 0;
    redrawCanvas();
    updateStats();
}