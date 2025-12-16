"""
YOLOv12m Custom Training Script - Optimized for RTX 4090 & Small Objects
針對 lvis_custom_yolo 資料集 (13類) 進行優化

硬體目標: NVIDIA RTX 4090 (24GB VRAM)
策略重點:
1. High Resolution (1024px): 提升小物件 (Key, Earphone, Glasses) 偵測率
2. Freeze Backbone (10 layers): 防止 Overfitting，保留預訓練特徵
3. RAM Caching: 利用大量 RAM 加速訓練
4. Optimized Hyperparameters: 針對 Transfer Learning 調整
"""
from ultralytics import YOLO
from pathlib import Path
import torch
import sys

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ 錯誤: 未檢測到 GPU！此腳本專為 RTX 4090 設計。")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ 檢測到 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    
    if "4090" not in gpu_name and gpu_mem < 20:
        print("⚠️ 警告: VRAM 可能不足以執行 1024px batch=24，請考慮降低 batch size。")

def main():
    # ===== 1. 路徑設定 =====
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent # 回到專案根目錄
    
    # 資料集路徑
    data_yaml = repo_root / "datasets" / "lvis_custom_yolo" / "data.yaml"
    
    # 輸出路徑
    runs_root = repo_root / "FindForYou" / "runs" / "train"
    
    if not data_yaml.exists():
        print(f"❌ 找不到資料集: {data_yaml}")
        sys.exit(1)
        
    print(f"📂 資料集: {data_yaml}")
    print(f"📂 輸出目錄: {runs_root}")
    
    check_gpu()
    
    # ===== 2. 載入模型 =====
    # 使用 YOLOv12m (Medium) - 兼顧速度與精度
    model_name = "yolov12m.pt" 
    try:
        model = YOLO(model_name)
        print(f"✅ 成功載入預訓練模型: {model_name}")
    except Exception as e:
        # Fallback for older ultralytics versions
        print(f"⚠️ 載入 {model_name} 失敗，嘗試 yolo12m.pt...")
        model = YOLO("yolo12m.pt")

    # ===== 3. 訓練參數 (RTX 4090 Optimized) =====
    training_args = {
        "data": str(data_yaml),
        "project": str(runs_root),
        "name": "lvis_custom_yolov12m_1024", # 專案名稱
        
        # --- 核心參數 ---
        "epochs": 150,          # 訓練輪數
        "patience": 40,         # Early stopping
        "batch": 24,            # 4090 24GB VRAM 建議值 (1024px)
        "imgsz": 1024,          # 🔥 關鍵：高解析度以偵測小物件
        
        # --- 優化與硬體 ---
        "device": 0,
        "workers": 16,          # 4090 處理快，需要更多 DataLoader workers
        "cache": True,          # 🔥 關鍵：將圖片快取到 RAM (加速 epoch 迭代)
        "amp": True,            # 混合精度 (4090 Tensor Cores 必開)
        
        # --- Transfer Learning 策略 ---
        "pretrained": True,
        "freeze": 10,           # 🔥 關鍵：凍結 Backbone 防止 Overfitting
        "lr0": 0.001,           # 初始學習率 (Transfer Learning 建議較低)
        "lrf": 0.01,            # 最終學習率
        "optimizer": "AdamW",   # 推薦優化器
        "warmup_epochs": 5.0,   # 較長的 Warmup
        
        # --- 資料增強 (針對小物件微調) ---
        "mosaic": 0.8,          # 稍微降低 Mosaic (避免小物件過度縮小)
        "mixup": 0.1,           # 輕微 Mixup
        "copy_paste": 0.1,      # Copy-Paste 有助於實例分割/偵測
        "degrees": 5.0,         # 輕微旋轉
        "scale": 0.4,           # 縮放範圍
        
        # --- Loss 權重 ---
        "box": 7.5,             # 提高 Box Loss 權重 (重視定位準確度)
        "cls": 0.5,             # 降低 Class Loss (類別較少且單純)
        
        "exist_ok": True,
        "save": True,
        "val": True,
    }
    
    print("\n" + "="*60)
    print("🚀 開始訓練 (RTX 4090 Mode)")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Image Size: {training_args['imgsz']} (High Res)")
    print(f"Batch Size: {training_args['batch']}")
    print(f"Freeze Layers: {training_args['freeze']}")
    print("="*60 + "\n")
    
    # ===== 4. 開始訓練 =====
    model.train(**training_args)
    
    print("\n✅ 訓練完成！")
    print(f"最佳權重位置: {runs_root}/{training_args['name']}/weights/best.pt")

if __name__ == "__main__":
    main()
