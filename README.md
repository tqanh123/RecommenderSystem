# TenrecRec - LightGCN Recommender System

## Overview
A dynamic web application demonstrating a Graph Neural Network (LightGCN) recommender system using the Tenrec dataset (Tencent). This system features a Node.js backend, MongoDB storage, and a Python-based model integration.

---

## 📋 Yêu Cầu Hệ Thống / Prerequisites

Trước khi cài đặt, đảm bảo máy tính của bạn đã cài đặt:
- **Node.js** (phiên bản 14 trở lên) - [Tải tại đây](https://nodejs.org/)
- **MongoDB** (chạy local hoặc sử dụng MongoDB Atlas) - [Tải tại đây](https://www.mongodb.com/try/download/community)
- **Python** (phiên bản 3.8 trở lên) - [Tải tại đây](https://www.python.org/downloads/)
- **PyTorch** (cho mô hình GNN)

### ✅ Kiểm Tra Phiên Bản / Check Versions

Mở Terminal/PowerShell và chạy các lệnh sau để kiểm tra phiên bản:

**1. Kiểm tra Node.js:**
```powershell
node --version
# Hoặc
node -v
```
Kết quả mong đợi: `v14.x.x` hoặc cao hơn (ví dụ: `v18.17.0`)

**2. Kiểm tra npm (Node Package Manager):**
```powershell
npm --version
# Hoặc
npm -v
```
Kết quả mong đợi: `6.x.x` hoặc cao hơn (ví dụ: `9.6.7`)

**3. Kiểm tra Python:**
```powershell
python --version
# Hoặc thử
python3 --version
```
Kết quả mong đợi: `Python 3.8.x` hoặc cao hơn (ví dụ: `Python 3.11.4`)

**4. Kiểm tra MongoDB:**
```powershell
mongod --version
# Hoặc
mongo --version
```
Kết quả mong đợi: `db version v4.x.x` hoặc cao hơn

**5. Kiểm tra PyTorch (sau khi cài):**
```powershell
python -c "import torch; print(torch.__version__)"
```
Kết quả mong đợi: Phiên bản PyTorch (ví dụ: `2.0.1`)

**Lưu ý:**
- Nếu lệnh không được nhận diện, có nghĩa là chưa cài đặt hoặc chưa thêm vào PATH
- Trên Windows, có thể cần khởi động lại Terminal sau khi cài đặt
- Nếu `python` không hoạt động, thử `python3`
- Nếu `pip` không hoạt động, thử `pip3`

---

## 🚀 Installation guide for experiment 
1. Create environment
```bash
conda create -n gnn python=3.8 -y
conda activate gnn
```
2. Install pytorch with CUDA
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

3. Install pytorch geometric
```bash
pip install torch-geometric 
```
4. Install other requirements
```bash
pip install -r requirements.txt
```


## 🚀 Installation guide for demo

### Step 1: Clone or download soure
```bash
# Nếu dùng Git
git clone <repository-url>
cd RecommenderSystem

# Hoặc giải nén file zip vào thư mục RecommenderSystem
```

### Step 2: Install Node.js Dependencies
Open Terminal/Command Prompt at RecommenderSystem folder and run:

**Windows PowerShell:**
```powershell
npm install
```

**Linux/Mac:**
```bash
npm install
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
Load data into database
```powershell
node seed_tenrec.js
```

### Option 1: Running
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
```

Click to access the demo website:
```
http://localhost:3000
```

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

## 📁 Project Structure
```
RecommenderSystem/
├── server.js              # Server chính
├── package.json           # Node dependencies
├── .env                   # Biến môi trường (tự tạo)
├── public/
│   └── index.html        # Giao diện người dùng
├── python/
│   └── model.py          # Mô hình LightGCN (PyTorch)
└── src/
    ├── models/           # MongoDB Models
    │   ├── User.js
    │   ├── Item.js
    │   └── Interaction.js
    ├── routes/
    │   └── api.js        # API routes
    └── services/
        └── recommender.js # Logic recommender
```

---

## 🔧  Troubleshooting

### Error: "Cannot find module"
```powershell
npm install
```

### Error: MongoDB connection failed
- Check MongoDB is running
- Check `MONGODB_URI` in file `.env`

### Error: Port 3000 is used
Change `PORT` in file `.env` into other ports (ex: 3001)

### Error: Python is not found
Check `PYTHON_PATH` in `.env`:
- Windows: usually `python`
- Linux/Mac: usually `python3`
