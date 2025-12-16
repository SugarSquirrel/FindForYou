"""
找東西助手 - YOLO12 + DINOv2 物件偵測器
純推論模式：接收圖片 → YOLO12 偵測 → DINOv2 特徵比對 → 返回結果
"""

import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ultralytics import YOLO

from feature_extractor import FeatureExtractor
from object_registry import ObjectRegistry


@dataclass
class DetectionResult:
    """偵測結果資料結構"""
    object_class: str
    object_class_zh: str
    confidence: float
    bbox: List[float]
    matched_object_id: Optional[str] = None
    matched_object_name: Optional[str] = None
    matched_object_name_zh: Optional[str] = None
    similarity: Optional[float] = None
    surface: Optional[str] = None
    region: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    
    def to_dict(self):
        return {
            "object_class": self.object_class,
            "object_class_zh": self.object_class_zh,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "matched_object_id": self.matched_object_id,
            "matched_object_name": self.matched_object_name,
            "matched_object_name_zh": self.matched_object_name_zh,
            "similarity": self.similarity,
            "surface": self.surface,
            "region": self.region,
            "timestamp": self.timestamp
        }


# COCO 類別對照表
CLASS_NAMES_ZH = {
    "person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "機車",
    "airplane": "飛機", "bus": "公車", "train": "火車", "truck": "卡車",
    "boat": "船", "traffic light": "交通燈", "fire hydrant": "消防栓",
    "stop sign": "停止標誌", "parking meter": "停車收費表", "bench": "長凳",
    "bird": "鳥", "cat": "貓", "dog": "狗", "horse": "馬", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑馬",
    "giraffe": "長頸鹿", "backpack": "背包", "umbrella": "雨傘",
    "handbag": "手提包", "tie": "領帶", "suitcase": "行李箱",
    "frisbee": "飛盤", "skis": "滑雪板", "snowboard": "滑雪板",
    "sports ball": "運動球", "kite": "風箏", "baseball bat": "棒球棒",
    "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "衝浪板",
    "tennis racket": "網球拍", "bottle": "水瓶", "wine glass": "酒杯",
    "cup": "杯子", "fork": "叉子", "knife": "刀子", "spoon": "湯匙",
    "bowl": "碗", "banana": "香蕉", "apple": "蘋果", "sandwich": "三明治",
    "orange": "橘子", "broccoli": "花椰菜", "carrot": "胡蘿蔔",
    "hot dog": "熱狗", "pizza": "披薩", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子", "couch": "沙發", "potted plant": "盆栽", "bed": "床",
    "dining table": "餐桌", "toilet": "馬桶", "tv": "電視",
    "laptop": "筆電", "mouse": "滑鼠", "remote": "遙控器",
    "keyboard": "鍵盤", "cell phone": "手機", "microwave": "微波爐",
    "oven": "烤箱", "toaster": "烤麵包機", "sink": "水槽",
    "refrigerator": "冰箱", "book": "書", "clock": "時鐘",
    "vase": "花瓶", "scissors": "剪刀", "teddy bear": "泰迪熊",
    "hair drier": "吹風機", "toothbrush": "牙刷",
    # Finetune 模型自訂類別
    "cell_phone": "手機", "wallet": "錢包", "key": "鑰匙",
    "remote_control": "遙控器", "watch": "手錶", "earphone": "耳機",
    "cup": "杯子", "bottle": "水瓶"
}


class ObjectDetector:
    """YOLO12 + DINOv2 物件偵測器 (純推論模式)"""
    
    # 允許偵測的 COCO 類別 ID (對應 LVIS 需求)
    ALLOWED_CLASS_IDS = [24, 26, 39, 41, 65, 67, 73]  # backpack, handbag, bottle, cup, remote, cell phone, book
    
    # Tune 模型類別名稱對照表
    TUNE_CLASS_NAMES = {
        0: "cellular phone",
        1: "remote control",
        2: "backpack",
        3: "handbag",
        4: "book",
        5: "bottle",
        6: "cup",
        7: "key",
        8: "watch",
        9: "earphone",
        10: "glasses",
        11: "notebook",
        12: "mask"
    }
    
    # Tune 模型類別中文對照表
    TUNE_CLASS_NAMES_ZH = {
        "cellular phone": "手機",
        "remote control": "遙控器",
        "backpack": "背包",
        "handbag": "手提包",
        "book": "書",
        "bottle": "水瓶",
        "cup": "杯子",
        "key": "鑰匙",
        "watch": "手錶",
        "earphone": "耳機",
        "glasses": "眼鏡",
        "notebook": "筆記本",
        "mask": "口罩"
    }
    
    # 與基本 YOLO 重疊的 tune 類別 (這些類別如果和基本 YOLO 偵測到同一物件則不繪製)
    # 格式: tune_class_name -> 對應的基本 YOLO class_name
    OVERLAPPING_CLASSES = {
        "cellular phone": "cell phone",
        "remote control": "remote",
        "backpack": "backpack",
        "handbag": "handbag",
        "book": "book",
        "bottle": "bottle",
        "cup": "cup"
    }
    
    def __init__(
        self, 
        model_path: str = "yolo12l.pt",
        tune_model_path: str = "yolo12l_tune.pt",
        similarity_threshold: float = 0.7
    ):
        self.model_path = model_path
        self.tune_model_path = tune_model_path
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.tune_model = None  # 微調模型
        self.feature_extractor = None
        self.object_registry = None
        self.is_ready = False
        
        # 初始化
        self._init_model()
        self._init_tune_model()
        self._init_feature_extractor()
        self._init_registry()
    
    def _init_model(self):
        """初始化 YOLO12 模型"""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                print(f"✅ YOLO12 模型已載入到 GPU: {torch.cuda.get_device_name(0)}")
            
            self.model = YOLO(self.model_path)
            self.model.to(device)
            print(f"✅ YOLO12 模型已載入: {self.model_path}")
            
        except Exception as e:
            print(f"❌ YOLO12 模型載入失敗: {e}")
            raise
    
    def _init_tune_model(self):
        """初始化微調 YOLO12 模型"""
        try:
            import torch
            import os
            
            # 檢查 tune 模型檔案是否存在
            tune_path = os.path.join(os.path.dirname(__file__), self.tune_model_path)
            if not os.path.exists(tune_path):
                print(f"⚠️ 微調模型不存在: {tune_path}，跳過載入")
                return
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tune_model = YOLO(tune_path)
            self.tune_model.to(device)
            print(f"✅ 微調 YOLO12 模型已載入: {self.tune_model_path}")
            
        except Exception as e:
            print(f"⚠️ 微調 YOLO12 模型載入失敗: {e}")
            # 不拋出例外，允許系統在沒有 tune 模型的情況下運行
    
    def _init_feature_extractor(self):
        """初始化 DINOv2 特徵提取器"""
        try:
            self.feature_extractor = FeatureExtractor()
            print("✅ DINOv2 特徵提取器已初始化")
            
        except Exception as e:
            print(f"❌ DINOv2 特徵提取器初始化失敗: {e}")
            raise
    
    def _init_registry(self):
        """初始化物品註冊資料庫"""
        try:
            self.object_registry = ObjectRegistry()
            print("✅ 物品註冊資料庫已載入")
            self.is_ready = True
            
        except Exception as e:
            print(f"❌ 物品註冊資料庫載入失敗: {e}")
            raise
    
    def detect_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        偵測圖片中的物品 (主要方法)
        同時使用基本 YOLO 和微調模型進行偵測
        
        Args:
            frame: BGR 格式的 numpy 陣列
            
        Returns:
            List[DetectionResult]: 偵測結果列表
        """
        if not self.is_ready or self.model is None:
            return []
        
        results = []
        
        try:
            # ===== 1. 基本 YOLO12 偵測 =====
            yolo_results = self.model(frame, verbose=False)[0]
            
            for box in yolo_results.boxes:
                # 取得基本資訊
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # 🔥 只保留允許的類別
                if cls_id not in self.ALLOWED_CLASS_IDS:
                    continue
                
                # 類別名稱
                class_name = self.model.names[cls_id]
                class_name_zh = CLASS_NAMES_ZH.get(class_name, class_name)
                
                # 裁切物品區域
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # 計算物品區域
                frame_height, frame_width = frame.shape[:2]
                center_x = (x1 + x2) / 2 / frame_width
                region = "左側" if center_x < 0.33 else "右側" if center_x > 0.67 else "中間"
                
                # 預設結果
                result = DetectionResult(
                    object_class=class_name,
                    object_class_zh=class_name_zh,
                    confidence=conf,
                    bbox=bbox,
                    region=region,
                    surface="偵測區域"
                )
                
                # DINOv2 特徵比對
                if len(self.object_registry.objects) > 0:
                    crop_embedding = self.feature_extractor.extract(crop)
                    
                    if crop_embedding is not None:
                        # 與已註冊物品比對
                        for obj in self.object_registry.objects.values():
                            if not obj.embeddings:
                                continue
                            
                            max_sim = 0
                            for emb in obj.embeddings:
                                sim = self.feature_extractor.cosine_similarity(crop_embedding, emb)
                                max_sim = max(max_sim, sim)
                            
                            if max_sim >= self.similarity_threshold:
                                result.matched_object_id = obj.id
                                result.matched_object_name = obj.name
                                result.matched_object_name_zh = obj.name_zh
                                result.similarity = max_sim
                                result.object_class_zh = obj.name_zh
                                break
                
                results.append(result)
            
            # ===== 2. 微調模型偵測 (如果有載入) =====
            if self.tune_model is not None:
                tune_results = self.tune_model(frame, verbose=False)[0]
                
                for box in tune_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    
                    # 取得 tune 模型的類別名稱
                    tune_class_name = self.TUNE_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    tune_class_name_zh = self.TUNE_CLASS_NAMES_ZH.get(tune_class_name, tune_class_name)
                    
                    # 🔥 完全跳過重疊類別（不管位置是否重疊）
                    if tune_class_name in self.OVERLAPPING_CLASSES:
                        continue
                    
                    # 裁切物品區域
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # 計算物品區域
                    frame_height, frame_width = frame.shape[:2]
                    center_x = (x1 + x2) / 2 / frame_width
                    region = "左側" if center_x < 0.33 else "右側" if center_x > 0.67 else "中間"
                    
                    # 預設結果
                    result = DetectionResult(
                        object_class=tune_class_name,
                        object_class_zh=tune_class_name_zh,
                        confidence=conf,
                        bbox=bbox,
                        region=region,
                        surface="偵測區域"
                    )
                    
                    # DINOv2 特徵比對
                    if len(self.object_registry.objects) > 0:
                        crop_embedding = self.feature_extractor.extract(crop)
                        
                        if crop_embedding is not None:
                            # 與已註冊物品比對
                            for obj in self.object_registry.objects.values():
                                if not obj.embeddings:
                                    continue
                                
                                max_sim = 0
                                for emb in obj.embeddings:
                                    sim = self.feature_extractor.cosine_similarity(crop_embedding, emb)
                                    max_sim = max(max_sim, sim)
                                
                                if max_sim >= self.similarity_threshold:
                                    result.matched_object_id = obj.id
                                    result.matched_object_name = obj.name
                                    result.matched_object_name_zh = obj.name_zh
                                    result.similarity = max_sim
                                    result.object_class_zh = obj.name_zh
                                    break
                    
                    results.append(result)
                
        except Exception as e:
            print(f"❌ 偵測錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def annotate_frame(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """在圖片上畫出偵測框和標籤"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # 決定顏色（有匹配到用綠色，否則用藍色）
            if det.matched_object_id:
                color = (0, 255, 0)  # 綠色
                # 使用英文名稱避免中文渲染問題
                label = f"{det.matched_object_name} ({det.similarity:.0%})"
            else:
                color = (255, 128, 0)  # 藍色
                label = f"{det.object_class} ({det.confidence:.0%})"
            
            # 畫框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 畫標籤背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    # ========================================
    # 物品註冊方法
    # ========================================
    
    def register_object(self, name: str, name_zh: str, image: np.ndarray) -> Optional[dict]:
        """註冊新物品"""
        if self.object_registry is None or self.feature_extractor is None:
            return None
        
        # 使用 YOLO 裁切主要物件
        crops = self._extract_main_object(image)
        target_image = crops[0] if crops else image
        
        # 提取特徵
        embedding = self.feature_extractor.extract(target_image)
        if embedding is None:
            return None
        
        # 儲存圖片
        import os
        from datetime import datetime
        
        img_dir = os.path.join(os.path.dirname(__file__), "object_images")
        os.makedirs(img_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{name}_{timestamp}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, target_image)
        
        # 註冊到資料庫
        obj = self.object_registry.register(
            name=name,
            name_zh=name_zh,
            embedding=embedding,
            image_path=img_path
        )
        
        if obj:
            return {
                "id": obj.id,
                "name": obj.name,
                "name_zh": obj.name_zh,
                "embedding_count": len(obj.embeddings),
                "thumbnail": f"/object_images/{img_name}"
            }
        return None
    
    def register_object_direct(self, name: str, name_zh: str, image: np.ndarray) -> Optional[dict]:
        """直接註冊物品（不使用 YOLO 裁切，適用於已裁切的偵測結果）"""
        if self.object_registry is None or self.feature_extractor is None:
            return None
        
        # 直接使用傳入的圖片，不進行 YOLO 裁切
        embedding = self.feature_extractor.extract(image)
        if embedding is None:
            return None
        
        # 儲存圖片
        import os
        from datetime import datetime
        
        img_dir = os.path.join(os.path.dirname(__file__), "object_images")
        os.makedirs(img_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{name}_{timestamp}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, image)
        
        # 註冊到資料庫
        obj = self.object_registry.register(
            name=name,
            name_zh=name_zh,
            embedding=embedding,
            image_path=img_path
        )
        
        if obj:
            return {
                "id": obj.id,
                "name": obj.name,
                "name_zh": obj.name_zh,
                "embedding_count": len(obj.embeddings),
                "thumbnail": f"/object_images/{img_name}"
            }
        return None
    
    def add_object_image(self, obj_id: str, image: np.ndarray) -> Optional[dict]:
        """為已註冊物品新增照片"""
        if self.object_registry is None or self.feature_extractor is None:
            return None
        
        obj = self.object_registry.get(obj_id)
        if not obj:
            return None
        
        # 裁切主要物件
        crops = self._extract_main_object(image)
        target_image = crops[0] if crops else image
        
        # 提取特徵
        embedding = self.feature_extractor.extract(target_image)
        if embedding is None:
            return None
        
        # 儲存圖片
        import os
        from datetime import datetime
        
        img_dir = os.path.join(os.path.dirname(__file__), "object_images")
        os.makedirs(img_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{obj.name}_{timestamp}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, target_image)
        
        # 更新物品
        updated = self.object_registry.add_embedding(obj_id, embedding, image_path=img_path)
        
        if updated:
            return {
                "id": updated.id,
                "name": updated.name,
                "name_zh": updated.name_zh,
                "embedding_count": len(updated.embeddings)
            }
        return None
    
    def delete_object(self, obj_id: str) -> bool:
        """刪除物品"""
        if self.object_registry is None:
            return False
        return self.object_registry.delete(obj_id)
    
    def get_registered_objects(self) -> List[dict]:
        """取得所有已註冊物品"""
        if self.object_registry is None:
            return []
        
        objects = []
        for obj in self.object_registry.objects.values():
            thumbnail = None
            if obj.images:
                import os
                thumbnail = f"/object_images/{os.path.basename(obj.images[0])}"
            
            objects.append({
                "id": obj.id,
                "name": obj.name,
                "name_zh": obj.name_zh,
                "embedding_count": len(obj.embeddings),
                "thumbnail": thumbnail
            })
        return objects
    
    def _extract_main_object(self, image: np.ndarray) -> List[np.ndarray]:
        """使用 YOLO 裁切圖片中的主要物件"""
        if self.model is None:
            return []
        
        try:
            results = self.model(image, verbose=False)[0]
            crops = []
            
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                
                # 擴展邊界
                h, w = image.shape[:2]
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
            
            # 按面積排序，返回最大的
            crops.sort(key=lambda c: c.shape[0] * c.shape[1], reverse=True)
            return crops[:1] if crops else []
            
        except Exception as e:
            print(f"⚠️ 物件裁切失敗: {e}")
            return []
