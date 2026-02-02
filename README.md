# TenrecRec - LightGCN Recommender System

## Overview
A dynamic web application demonstrating a Graph Neural Network (LightGCN) recommender system using the Tenrec dataset (Tencent). This system features a Node.js backend, MongoDB storage, and a Python-based model integration.

---

## Yêu Cầu Hệ Thống / Prerequisites

Trước khi cài đặt, đảm bảo máy tính của bạn đã cài đặt:
- **Node.js** (phiên bản 14 trở lên) - [Tải tại đây](https://nodejs.org/)
- **MongoDB** (chạy local hoặc sử dụng MongoDB Atlas) - [Tải tại đây](https://www.mongodb.com/try/download/community)
- **Python** (phiên bản 3.8 trở lên) - [Tải tại đây](https://www.python.org/downloads/)
- **PyTorch** (cho mô hình GNN)

---

## 🚀 Installation guide for demo

### Step 1: Clone or download source
```bash
# Nếu dùng Git
git clone <https://github.com/tqanh123/RecommenderSystem.git>
cd RecommenderSystem

# Hoặc giải nén file zip vào thư mục RecommenderSystem
```

### Step 2: Install Dependencies

#### 2.1. Install Node.js Dependencies
```powershell
npm install
```

#### 2.2. Install Python Dependencies

**Option A: Using Conda (gnn)**
```powershell
# Create conda environment from file
conda env create -f framework/environment.yml

# Activate environment
conda activate gnn
```

**Option B: Using pip + venv**
```powershell
# Create virtual environment
python -m venv .venv

# Activate venv (Windows)
.venv\Scripts\activate

# Activate venv (Linux/Mac)
# source .venv/bin/activate

# Install PyTorch with CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### Step 3: Config environment
Create file `.env` in project folder:

**Option 1: Create thủ công**
- Create new file `.env`
- Add contents:
```
PORT=3000
MONGODB_URI=mongodb://localhost:27017/tenrecrec
PYTHON_PATH=python
```

**Option 2: run in terminal (Windows PowerShell):**
```powershell
@"
PORT=3000
MONGODB_URI=mongodb://localhost:27017/tenrecrec
PYTHON_PATH=python
"@ | Out-File -FilePath .env -Encoding utf8
```

**Option 2: run in terminal (Linux/Mac):**
```bash
cat > .env << EOF
PORT=3000
MONGODB_URI=mongodb://localhost:27017/tenrecrec
PYTHON_PATH=python3
EOF
```

**Lưu ý:**
- If MongoDB run on another path, fix in `MONGODB_URI`
- If using MongoDB Atlas (cloud), replace with connection string of Atlas

### Step 4: Start MongoDB
Confirm MongoDB is running:

**Windows:**
```powershell
mongod
```

**Linux/Mac:**
```bash
# Start MongoDB service
sudo systemctl start mongod
# or
mongod
```

---

## ▶️ Running the Application

### Before running
1. **Load data into database**
```powershell
node scripts/seed_tenrec.js
```

2. **Start the Python LightGCN Inference Service (REQUIRED for real-time updates)**

Open a **separate Terminal/PowerShell** window and run:

**Option A: Using Conda**
```powershell
# Activate conda environment
conda activate recommender

# Start the inference service
python framework/inference_service.py
```

**Option B: Using venv**
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Start the inference service
python framework/inference_service.py
```

You should see output like:
```
============================================================
🚀 Starting LightGCN Inference Service
============================================================
📋 Config loaded: {'n_users': 100, 'n_items': 500, ...}
✅ Model loaded successfully from framework/checkpoint/best_model.pth
   Users: 100, Items: 500
   Embedding dim: 64, Layers: 3
   Device: cpu
============================================================
✅ Server ready!
   Endpoints:
   - GET  /health
   - GET  /recommend/<user_id>?k=20
   - GET  /user-embedding/<user_id>
   - GET  /item-embedding/<item_id>
============================================================
 * Running on http://0.0.0.0:5001
```

**⚠️ IMPORTANT:** Keep this terminal window open. The Python service must be running for real-time recommendations to work!

### Option 1: Running the Node.js Server
Open Terminal/PowerShell in project folder and run:

```powershell
node server.js
```

Or using npm script:
```powershell
npm start
```

### Option 2: Run in Development (Auto-reload)
```powershell
npm run dev
```
(Requirement nodemon: `npm install -g nodemon`)

### Access website
If server starting, you will see the notification:
```
Server running on port 3000
MongoDB Connected!
✅ Python LightGCN inference service is available
```

Click to access the demo website:
```
http://localhost:3000
```

---

### What Happens When You Interact?

1. **User clicks/likes/shares an item:**
   - Interaction is logged to MongoDB
   - User embedding is updated (moves closer to item embedding)
   - Frontend immediately re-fetches the updated user embedding
   - Canvas redraws showing new user position

2. **Real-time visualization:**
   - User node moves in embedding space
   - Lines connect user to interacted items
   - Other users who liked the same items appear
   - Top 10 closest items are highlighted

3. **Recommendation refresh:**
   - Python LightGCN model computes new recommendations
   - Items are ranked by predicted affinity
   - Prediction scores (60-95%) reflect confidence

### Testing Real-Time Updates

1. Open the demo at `http://localhost:3000`
2. Watch the **Embedding Space** panel on the right
3. Click "Like ❤️" on any item
4. **Observe:**
   - User node moves immediately
   - New line appears connecting to the liked item
   - Recommendations update after 1.5 seconds
   - Prediction scores change based on new preferences

### Performance Notes

- **Embedding updates:** ~50ms (simple gradient descent)
- **Python inference:** ~100-200ms (depends on model size)
- **Frontend redraw:** ~16ms (60 FPS canvas rendering)
- **Total latency:** <500ms for complete update cycle

---

### Option 3: Connect MongoDB in VSCode

**1. Install Extension:**
- **MongoDB for VS Code** (mongodb.mongodb-vscode)

**2. Connect Database:**
- Click MongoDB symbol at sidebar
- Click "Add Connection"
- Input connection string:
  ```
  mongodb://localhost:27017/tenrecrec
  ```
  Or MongoDB Atlas:
  ```
  mongodb+srv://username:password@cluster.mongodb.net/tenrecrec
  ```

**3. View Database:**
- Browse collections: `users`, `items`, `interactions`
- Run queries in VSCode
- Export/Import data

---

## Project Structure
```
RecommenderSystem/
├── server.js                    # Node.js server chính
├── package.json                 # Node dependencies
├── .env                         # Environment variables
├── framework/
│   ├── inference_service.py    # Flask API serving LightGCN model
│   ├── model.py                # LightGCN implementation (PyTorch)
│   ├── config.json             # Model configuration
│   └── checkpoint/
│       └── best_model.pth      # Trained model weights
├── public/
│   └── html/
│       └── index.html          # Frontend with real-time visualization
└── src/
    ├── models/                 # MongoDB Models
    │   ├── User.js
    │   ├── Item.js
    │   └── Interaction.js
    ├── routes/
    │   └── api.js              # API routes
    └── services/
        └── recommender.js      # Recommendation logic + Python integration
```

---

## Troubleshooting

### Error: "Cannot find module"
```powershell
npm install
```

### Error: MongoDB connection failed
- Check MongoDB is running
- Check `MONGO_URI` in file `.env`

### Error: Port 3000 is used
Change `PORT` in file `.env` into other ports (ex: 3001)

### Error: Python is not found
Check `PYTHON_PATH` in `.env`:
- Windows: usually `python`
- Linux/Mac: usually `python3`

### Error: Python inference service failed to start
```powershell
# Install dependencies
pip install -r framework/requirements_inference.txt

# Check if model file exists
dir framework\checkpoint\best_model.pth

# Test Python imports
python -c "import torch; import flask; print('OK')"
```

### Error: Graph not updating in real-time
1. **Check Python service is running:**
   - Open http://localhost:5001/health in browser
   - Should return: `{"status": "healthy", "model_loaded": true}`

2. **Check browser console (F12):**
   - Look for "✅ User embedding re-fetched" messages
   - Look for "✅ Python model returned X recommendations"

3. **Check backend logs:**
   - Should see: "✅ Python LightGCN inference service is available"
   - If not, check `.env` file has: `PYTHON_SERVICE_URL=http://localhost:5001`

4. **Force refresh:**
   - Clear browser cache (Ctrl+Shift+Delete)
   - Restart both Node.js server and Python service
   - Hard reload page (Ctrl+F5)

### Performance issues
- **Python service slow?** 
  - Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
  - Reduce batch size in config.json
  
- **Frontend lag?**
  - Reduce number of items displayed (limit=20 instead of 50)
  - Disable canvas animations if too slow
