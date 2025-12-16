# FindForYou - 日常物品搜尋助手

使用 **YOLO12 + DINOv2** 個人化物件偵測技術，幫助你快速找到家中日常生活用品。

## 🎯 功能特色

- **個人化偵測** - 拍攝你的物品照片註冊，系統透過 AI 特徵比對識別你的特定物品
- **YOLO12 物件偵測** - 使用最新 YOLO12m 模型進行快速準確的物件偵測
- **DINOv2 特徵匹配** - 使用 Meta DINOv2 提取視覺特徵，比對用戶註冊的物品
- **多攝影機支援** - 支援多個攝影機，可設定名稱和所在位置
- **自動定時偵測** - 可開關的背景定時偵測，透過 WebSocket 即時推送
- **設定頁面** - 完整的設定介面管理攝影機、物品註冊和自動偵測
- **常用物品快捷** - 可自訂首頁的常用物品快捷搜尋按鈕
- **即時預覽** - 設定頁面可預覽各攝影機畫面
- **語音輸入** - 支援語音搜尋（需瀏覽器支援）
- **離線使用** - 資料存在瀏覽器 IndexedDB，離線也能查詢

## 🔧 系統架構

```
註冊流程: 用戶拍照 → YOLO12 裁切物件 → DINOv2 提取特徵 → 儲存到物品資料庫
偵測流程: 攝影機畫面 → YOLO12 偵測 → DINOv2 提取特徵 → 比對用戶物品 → 識別結果
```

## 📁 專案結構

```
FindForYou/
├── backend/                      # Python 後端服務
│   ├── main.py                  # FastAPI 入口
│   ├── detector.py              # YOLO12 + DINOv2 偵測器
│   ├── feature_extractor.py     # DINOv2 特徵提取器
│   ├── object_registry.py       # 物品註冊資料庫
│   ├── scheduler.py             # 定時排程器
│   ├── registered_objects.json  # 已註冊物品資料
│   ├── camera_config.json       # 攝影機配置
│   └── requirements.txt         # Python 依賴
│
├── frontend/                     # Web 前端
│   ├── index.html               # 主頁面
│   ├── settings.html            # 設定頁面
│   ├── css/style.css            # 現代化樣式
│   └── js/
│       ├── app.js               # 主程式
│       ├── db.js                # IndexedDB 操作
│       ├── api.js               # API 通訊
│       └── ui.js                # UI 互動
│
└── train/                        # YOLOv12 訓練 🆕
    ├── train_yolov12m.py        # YOLOv12m 高解析度訓練腳本
    └── README.md                # 訓練說明文件
```

## 🚀 快速開始

### 1. 啟動後端服務

```bash
cd backend

# 安裝依賴
pip install -r requirements.txt

# 啟動服務
uvicorn main:app --host 0.0.0.0 --port 8000
```

伺服器會在 `http://localhost:8000` 啟動。

### 2. 遠端/跨裝置連線

#### 同網路連線（區域網路）

在同一 WiFi/路由器下的其他裝置，使用本機 IP 連線：

```bash
# 查詢本機 IP
ipconfig  # Windows
ifconfig  # macOS/Linux

# 其他裝置訪問
http://<本機IP>:8000
```

> ⚠️ 手機瀏覽器使用攝影機需要 HTTPS，請參考下方 ngrok 方案。

#### 跨網路連線（使用 ngrok）

透過 [ngrok](https://ngrok.com/) 將本機服務暴露到公網：

```bash
# 1. 安裝 ngrok
winget install ngrok  # 或從官網下載

# 2. 設定 authtoken（註冊免費帳號取得）
ngrok config add-authtoken <your-token>

# 3. 啟動隧道
ngrok http 8000
```

ngrok 會提供一個 HTTPS 公開網址（如 `https://xxxx.ngrok-free.app`），任何裝置都可透過此網址連線。

### 3. 註冊你的物品

1. 開啟 `http://localhost:8000/settings`
2. 在「我的物品」區塊點擊「新增物品」
3. 上傳物品照片，輸入名稱
4. 可為同一物品新增多張不同角度的照片，提高識別準確度

### 4. 開始偵測

1. 開啟 `http://localhost:8000`
2. 點擊偵測按鈕，系統會識別畫面中你的物品
3. 偵測結果會顯示相似度分數

## 🔧 物品管理 API

```bash
# 列出已註冊物品
GET /api/objects

# 註冊新物品 (multipart/form-data)
POST /api/objects/register
  - image: 照片檔案
  - name: 英文名稱
  - name_zh: 中文名稱

# 為物品新增照片
POST /api/objects/{id}/images
  - image: 照片檔案

# 刪除物品
DELETE /api/objects/{id}
```

## 🛠️ 技術棧

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Detection**: YOLO12m (Ultralytics)
- **Feature Extraction**: DINOv2 ViT-S/14 (Meta)
- **Training**: YOLOv12m with 1024px high-resolution training
- **Storage**: IndexedDB (瀏覽器端), JSON (伺服器端)

## 🎓 模型訓練 (新增)

本專案支援自訂訓練 YOLOv12 模型以提升特定物品的偵測精度。

### 訓練配置
- **模型**: YOLOv12m (medium) - Transfer Learning 優化版
- **解析度**: 640x640 (針對小物件優化)
- **Batch Size**: 16
- **資料集**: 7 個日常物品類別 (手機、錢包、遙控器、手錶、耳機、杯子、瓶子) - *已移除鑰匙*
- **策略**: 凍結 Backbone (前 10 層) 以防止 Overfitting

### 訓練步驟

```bash
# 1. 準備資料集
# 確保資料集位於 ../../datasets/findyou_yolo_clean_cleaned_no_key/

# 2. 激活 conda 環境
conda activate d2_final

# 3. 執行訓練
cd train
python train_yolov12m.py
```

### 訓練輸出
- 訓練紀錄: `../../runs/train/findyou_yolov12m_1024/`
- 最佳模型: `../../runs/train/findyou_yolov12m_1024/weights/best.pt`

詳細訓練說明請參考 [train/README.md](train/README.md)

## 📄 License

MIT License
