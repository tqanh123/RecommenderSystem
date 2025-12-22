import sys
import json
import torch
import numpy as np
from model import LightGCN

def load_model(model_path, n_users, n_items, embedding_dim=64, n_layers=3):
    """Load trained model"""
    model = LightGCN(n_users, n_items, embedding_dim, n_layers)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_recommendations(model, user_id, k=10):
    """Get top-k recommendations for user"""
    with torch.no_grad():
        # Get user embedding
        user_emb = model.user_emb(torch.LongTensor([user_id]))
        
        # Get all item embeddings
        all_items = torch.arange(model.n_items)
        item_embs = model.item_emb(all_items)
        
        # Calculate scores
        scores = torch.matmul(user_emb, item_embs.T).squeeze()
        
        # Get top-k items
        top_k_scores, top_k_items = torch.topk(scores, k)
        
        return top_k_items.tolist(), top_k_scores.tolist()

if __name__ == "__main__":
    # Parse arguments
    user_id = int(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Load model config
    with open('python/config.json', 'r') as f:
        config = json.load(f)
    
    # Load model
    model = load_model(
        config['model_path'],
        config['n_users'],
        config['n_items'],
        config['embedding_dim'],
        config['n_layers']
    )
    
    # Get recommendations
    item_ids, scores = get_recommendations(model, user_id, k)
    
    # Output JSON
    result = {
        'user_id': user_id,
        'recommendations': [
            {'item_id': int(item_id), 'score': float(score)}
            for item_id, score in zip(item_ids, scores)
        ]
    }
    
    print(json.dumps(result))