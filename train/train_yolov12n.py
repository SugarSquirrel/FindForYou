"""
YOLOv12n Training Script for Fine-tuning on findyou_yolo_clean dataset
Lightweight 640px configuration for faster experiments
"""
from ultralytics import YOLO
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    findforyou_root = script_dir.parent
    repo_root = findforyou_root.parent
    runs_root = findforyou_root / "runs" / "train"
    runs_root.mkdir(parents=True, exist_ok=True)

    # 資料集路徑 (相對於專案根目錄)
    data_yaml = repo_root / "datasets" / "findyou_yolo_clean" / "data.yaml"
    
    # 檢查資料集檔案是否存在
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
    
    print(f"Loading dataset from: {data_yaml}")
    
    # 初始化 YOLOv12 模型
    # 使用預訓練權重進行 fine-tuning
    try:
        model = YOLO("yolo12n.pt")  # nano 版本，較輕量
        print("Successfully loaded YOLOv12 nano model")
    except Exception as e:
        print(f"Error loading YOLOv12 model: {e}")
        print("Trying alternative model initialization...")
        # 如果 yolo12 不可用，嘗試使用最新的 yolo11
        try:
            model = YOLO("yolo11n.pt")
            print("Using YOLOv11 nano model instead")
        except:
            # 最後備選使用 yolov8
            model = YOLO("yolov8n.pt")
            print("Using YOLOv8 nano model instead")
    
    # 訓練參數
    training_args = {
        "data": str(data_yaml),
        "epochs": 100,               # 訓練輪數
        "imgsz": 640,                # 輸入影像大小
        "batch": 64,                 # batch size (根據 GPU 記憶體調整)
        "patience": 50,              # early stopping patience
        "save": True,                # 儲存檢查點
        "device": 0,                 # GPU 設備 (0 表示第一個 GPU，若無 GPU 會自動使用 CPU)
        "workers": 8,                # 資料載入的工作執行緒數
        "project": str(runs_root),  # 專案目錄 (寫到 FindForYou/runs)
        "name": "findyou_yolov12n",   # 實驗名稱
        "exist_ok": True,            # 允許覆蓋現有實驗
        "pretrained": True,          # 使用預訓練權重
        "optimizer": "auto",         # 優化器 (auto, SGD, Adam, AdamW, etc.)
        "verbose": True,             # 詳細輸出
        "seed": 42,                  # 隨機種子
        "deterministic": True,       # 確定性訓練
        "single_cls": False,         # 是否單類別檢測
        "rect": False,               # 矩形訓練
        "cos_lr": False,             # cosine learning rate scheduler
        "close_mosaic": 10,          # 最後 N 輪關閉 mosaic 增強
        "resume": False,             # 是否從上次中斷處繼續
        "amp": True,                 # 自動混合精度訓練
        "fraction": 1.0,             # 使用的資料集比例
        "profile": False,            # 效能分析
        "freeze": None,              # 凍結層數
        "lr0": 0.01,                 # 初始學習率
        "lrf": 0.01,                 # 最終學習率
        "momentum": 0.937,           # SGD momentum/Adam beta1
        "weight_decay": 0.0005,      # 權重衰減
        "warmup_epochs": 3.0,        # warmup 輪數
        "warmup_momentum": 0.8,      # warmup 初始 momentum
        "warmup_bias_lr": 0.1,       # warmup 初始 bias 學習率
        "box": 7.5,                  # box loss gain
        "cls": 0.5,                  # cls loss gain
        "dfl": 1.5,                  # dfl loss gain
        "pose": 12.0,                # pose loss gain (僅用於 pose 模型)
        "kobj": 1.0,                 # keypoint obj loss gain (僅用於 pose 模型)
        "label_smoothing": 0.0,      # 標籤平滑
        "nbs": 64,                   # nominal batch size
        "overlap_mask": True,        # masks 是否應該重疊 (segment train)
        "mask_ratio": 4,             # mask downsample ratio (segment train)
        "dropout": 0.0,              # 使用 dropout regularization (僅分類訓練)
        "val": True,                 # 訓練時進行驗證
    }
    
    print("\n" + "="*50)
    print("Starting YOLOv12 Training")
    print("="*50)
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {training_args['epochs']}")
    print(f"Image Size: {training_args['imgsz']}")
    print(f"Batch Size: {training_args['batch']}")
    print(f"Device: {training_args['device']}")
    print("="*50 + "\n")
    
    # 開始訓練
    try:
        results = model.train(**training_args)
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
        # 驗證模型
        print("\nValidating trained model...")
        metrics = model.val()
        
        print("\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # 儲存最終模型
        save_path = runs_root / "findyou_yolov12n" / "weights" / "best.pt"
        print(f"\nBest model saved to: {save_path}")
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
