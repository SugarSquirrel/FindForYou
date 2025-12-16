"""
續訓腳本 - 從上次中斷處繼續訓練
"""
from ultralytics import YOLO
from pathlib import Path

def main():
    # 設定路徑
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    
    # 指定要續訓的 run 名稱
    run_name = "lvis_yolov12n_freeze20_anti_overfit"
    run_dir = repo_root / "FindForYou" / "runs" / "train" / run_name
    last_weights = run_dir / "weights" / "last.pt"
    
    # 檢查檔案是否存在
    if not last_weights.exists():
        print(f"❌ 找不到權重檔案: {last_weights}")
        print(f"請確認 run 名稱是否正確: {run_name}")
        return
    
    print("="*70)
    print("🔄 續訓模式")
    print("="*70)
    print(f"📂 Run: {run_name}")
    print(f"💾 從權重繼續: {last_weights.name}")
    print("="*70 + "\n")
    
    # 載入模型
    model = YOLO(str(last_weights))
    
    # 續訓 - Ultralytics 會自動讀取上次的 args.yaml 設定
    # 如果要調整參數，可以在這裡覆寫
    model.train(
        resume=True,  # 關鍵：啟用續訓模式
        
        # 以下參數可選：若要調整則取消註解
        # epochs=200,        # 延長總 epochs
        # patience=60,       # 調整 patience
        # lr0=0.003,         # 降低學習率（續訓常見策略）
        # mosaic=0.0,        # 關閉 mosaic（後期微調）
        # close_mosaic=0,    # 立即關閉 mosaic
    )
    
    print("\n✅ 續訓完成！")

if __name__ == "__main__":
    main()
