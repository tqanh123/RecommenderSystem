// 1. Hàm Sigmoid: Chuyển raw score thành xác suất (0 -> 1)
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// 2. Dot Product: Tích vô hướng giữa 2 vector
function dotProduct(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0;
    let sum = 0;
    for (let i = 0; i < vecA.length; i++) {
        sum += vecA[i] * vecB[i];
    }
    return sum;
}

// 3. Cập nhật Vector (Moving Average)
// alpha: Tốc độ học (0.1 -> 0.2 là an toàn cho demo)
function updateVector(oldVec, targetVec, alpha = 0.2) {
    return oldVec.map((val, i) => {
        return (1 - alpha) * val + alpha * targetVec[i];
    });
}

// 4. Tính Cosine Similarity (Tùy chọn, dùng để tìm item tương đồng)
function cosineSimilarity(vecA, vecB) {
    const dot = dotProduct(vecA, vecB);
    const magA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    if (magA === 0 || magB === 0) return 0;
    return dot / (magA * magB);
}

module.exports = { sigmoid, dotProduct, updateVector, cosineSimilarity };