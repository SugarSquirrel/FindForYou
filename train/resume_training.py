from ultralytics import YOLO
from pathlib import Path
import sys

def resume_training():
    # ===== 設定路徑 =====
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    
    # 這是 train_lvis_custom.py 設定的輸出位置
    runs_dir = repo_root / "FindForYou" / "runs" / "train"
    project_name = "lvis_custom_yolov12m_1024"
    
    # 尋找 last.pt (最後一次的權重檔)
    last_weight_path = runs_dir / project_name / "weights" / "last.pt"
    
    if not last_weight_path.exists():
        print(f"❌ 找不到中斷點權重檔: {last_weight_path}")
        print("請確認：")
        print("1. 您是否已經執行過 train_lvis_custom.py？")
        print("2. 訓練是否至少進行了一個 epoch 並儲存了 checkpoint？")
        sys.exit(1)
        
    print("\n" + "="*60)
    print(f"🚀 準備從中斷點恢復訓練")
    print("="*60)
    print(f"讀取權重: {last_weight_path}")
    
    try:
        # 1. 載入 last.pt
        model = YOLO(last_weight_path)
        
        # 2. 恢復訓練 (resume=True)
        # YOLO 會自動讀取 last.pt 裡面儲存的參數、優化器狀態和 Epoch 進度
        model.train(resume=True)
        
        print("\n✅ 訓練已完成！")
        
    except Exception as e:
        print(f"\n❌ 恢復訓練失敗: {e}")
        print("提示: 如果是 CUDA Out of Memory，請嘗試減少 batch size (雖然 resume 通常會沿用舊設定)")

if __name__ == "__main__":
    resume_training()
