"""
Flask Inference Service for LightGCN Recommender System
Serves real-time recommendations using the trained LightGCN model
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import json
import sys
from pathlib import Path

# Add framework directory to path
sys.path.append(str(Path(__file__).parent))

from model_inference import LightGCN

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global variables
model = None
config = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained LightGCN model"""
    global model, config
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"📋 Config loaded: {config}")
        
        # Initialize model architecture
        model_args = {
            'user_num': config['n_users'],
            'item_num': config['n_items'],
            'embedding_dim': config['embedding_dim'],
            'num_layers': config['n_layers'],
            'device': str(device),
            'interaction_matrix': None  # Not needed for inference
        }
        
        model = LightGCN(model_args)
        
        # Load trained weights
        model_path = Path(__file__).parent / config['model_path'].replace('framework/', '')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Format 1: {'model_state_dict': {...}, ...}
                state_dict = checkpoint['model_state_dict']
            else:
                # Format 2: Direct state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove norm_adj_matrix if present (not needed for inference)
        if 'norm_adj_matrix' in state_dict:
            del state_dict['norm_adj_matrix']
        
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Users: {config['n_users']}, Items: {config['n_items']}")
        print(f"   Embedding dim: {config['embedding_dim']}, Layers: {config['n_layers']}")
        print(f"   Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'device': str(device),
        'config': config
    })

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    """
    Get recommendations for a specific user
    
    Args:
        user_id (int): User ID (0-indexed)
        k (int): Number of recommendations (default: 20)
    
    Returns:
        JSON with recommendations: {item_id, score}[]
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        k = int(request.args.get('k', 20))
        
        # Validate user_id
        if user_id < 0 or user_id >= config['n_users']:
            return jsonify({
                'success': False,
                'error': f'Invalid user_id: {user_id}. Must be 0-{config["n_users"]-1}'
            }), 400
        
        with torch.no_grad():
            # Get user embedding from model
            user_emb = model.embed_user(torch.LongTensor([user_id]).to(device))
            
            # Get all item embeddings
            all_items = torch.arange(config['n_items']).to(device)
            item_embs = model.embed_item(all_items)
            
            # Calculate scores (dot product)
            scores = torch.matmul(user_emb, item_embs.T).squeeze()
            
            # Get top-k items
            top_k_scores, top_k_items = torch.topk(scores, min(k, len(scores)))
            
            recommendations = [
                {
                    'item_id': int(item_id),
                    'score': float(score)
                }
                for item_id, score in zip(top_k_items.cpu(), top_k_scores.cpu())
            ]
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'recommendations': recommendations,
            'model': 'LightGCN'
        })
        
    except Exception as e:
        print(f"❌ Error in recommendation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/user-embedding/<int:user_id>', methods=['GET'])
def get_user_embedding(user_id):
    """Get the embedding vector for a specific user"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        if user_id < 0 or user_id >= config['n_users']:
            return jsonify({
                'success': False,
                'error': f'Invalid user_id: {user_id}'
            }), 400
        
        with torch.no_grad():
            # Get user embedding
            user_emb = model.embed_user(torch.LongTensor([user_id]).to(device))
            embedding = user_emb.cpu().numpy()[0].tolist()
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'embedding': embedding,
            'embedding_dim': len(embedding)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/item-embedding/<int:item_id>', methods=['GET'])
def get_item_embedding(item_id):
    """Get the embedding vector for a specific item"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        if item_id < 0 or item_id >= config['n_items']:
            return jsonify({
                'success': False,
                'error': f'Invalid item_id: {item_id}'
            }), 400
        
        with torch.no_grad():
            # Get item embedding
            item_emb = model.embed_item(torch.LongTensor([item_id]).to(device))
            embedding = item_emb.cpu().numpy()[0].tolist()
        
        return jsonify({
            'success': True,
            'item_id': item_id,
            'embedding': embedding,
            'embedding_dim': len(embedding)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting LightGCN Inference Service")
    print("="*60)
    
    # Load model before starting server
    if not load_model():
        print("❌ Failed to load model. Exiting...")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Server ready!")
    print("   Endpoints:")
    print("   - GET  /health")
    print("   - GET  /recommend/<user_id>?k=20")
    print("   - GET  /user-embedding/<user_id>")
    print("   - GET  /item-embedding/<item_id>")
    print("="*60 + "\n")
    
    # Start Flask server
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
