"""
物件偵測器模組
使用 YOLO12 + DINOv2 進行個人化物件偵測
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 嘗試導入 ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics 未安裝，使用模擬模式")

# 導入特徵提取器和物品註冊資料庫
from feature_extractor import FeatureExtractor
from object_registry import ObjectRegistry


# ========================================
# 設定
# ========================================

# COCO 80 類別的中文對照 (常見物品)
COCO_CLASSES_ZH = {
    "person": "人",
    "bicycle": "腳踏車",
    "car": "汽車",
    "motorcycle": "機車",
    "airplane": "飛機",
    "bus": "公車",
    "train": "火車",
    "truck": "卡車",
    "boat": "船",
    "traffic light": "紅綠燈",
    "fire hydrant": "消防栓",
    "stop sign": "停止標誌",
    "parking meter": "停車計費器",
    "bench": "長椅",
    "bird": "鳥",
    "cat": "貓",
    "dog": "狗",
    "horse": "馬",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑馬",
    "giraffe": "長頸鹿",
    "backpack": "背包",
    "umbrella": "雨傘",
    "handbag": "手提包",
    "tie": "領帶",
    "suitcase": "行李箱",
    "frisbee": "飛盤",
    "skis": "滑雪板",
    "snowboard": "滑雪板",
    "sports ball": "球",
    "kite": "風箏",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "衝浪板",
    "tennis racket": "網球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀子",
    "spoon": "湯匙",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "蘋果",
    "sandwich": "三明治",
    "orange": "橘子",
    "broccoli": "花椰菜",
    "carrot": "胡蘿蔔",
    "hot dog": "熱狗",
    "pizza": "披薩",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙發",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "馬桶",
    "tv": "電視",
    "laptop": "筆電",
    "mouse": "滑鼠",
    "remote": "遙控器",
    "keyboard": "鍵盤",
    "cell phone": "手機",
    "microwave": "微波爐",
    "oven": "烤箱",
    "toaster": "烤麵包機",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "書",
    "clock": "時鐘",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹風機",
    "toothbrush": "牙刷",
}

# 常見居家物品類別 (優先偵測這些)
HOME_OBJECT_CLASSES = [
    "cell phone", "remote", "book", "cup", "bottle", 
    "laptop", "mouse", "keyboard", "scissors", "clock",
    "backpack", "handbag", "umbrella", "suitcase",
    "teddy bear", "vase", "toothbrush", "hair drier"
]


@dataclass
class Detection:
    """偵測結果資料類別"""
    object_class: str           # COCO 類別名稱
    object_class_zh: str        # 中文類別名稱
    confidence: float           # YOLO 偵測信心度
    bbox: List[float]           # 邊界框 [x1, y1, x2, y2]
    matched_object_id: Optional[str] = None  # 匹配的用戶物品 ID
    matched_object_name: Optional[str] = None  # 匹配的用戶物品名稱
    matched_object_name_zh: Optional[str] = None  # 匹配的用戶物品中文名稱
    similarity: Optional[float] = None  # 特徵相似度
    surface: Optional[str] = None  # 所在表面/位置
    region: Optional[str] = None   # 區域
    timestamp: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class ObjectDetector:
    """YOLO12 + DINOv2 物件偵測器類別"""
    
    def __init__(
        self, 
        model_path: str = "yolo12m.pt",  # YOLO12 Medium
        camera_source: int = 0,
        similarity_threshold: float = 0.7
    ):
        self.model_path = model_path
        self.camera_source = camera_source
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.feature_extractor = None
        self.object_registry = None
        self.is_ready = False
        
        # 初始化
        self._init_model()
        self._init_feature_extractor()
        self._init_registry()
    
    def _init_model(self):
        """初始化 YOLO12 模型"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO 不可用，使用模擬模式")
            self.is_ready = True
            return
        
        try:
            # 載入 YOLO12 模型
            self.model = YOLO(self.model_path)
            
            # 設定使用 GPU
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                print(f"✅ YOLO12 模型已載入到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA 不可用，YOLO12 使用 CPU")
            
            print(f"✅ YOLO12 模型已載入: {self.model_path}")
            
        except Exception as e:
            print(f"❌ YOLO12 模型載入失敗: {e}")
            self.model = None
    
    def _init_feature_extractor(self):
        """初始化 DINOv2 特徵提取器"""
        try:
            self.feature_extractor = FeatureExtractor(model_name="dinov2_vits14")
            print("✅ DINOv2 特徵提取器已初始化")
        except Exception as e:
            print(f"❌ DINOv2 初始化失敗: {e}")
            self.feature_extractor = None
    
    def _init_registry(self):
        """初始化物品註冊資料庫"""
        try:
            self.object_registry = ObjectRegistry()
            self.is_ready = True
            print("✅ 物品註冊資料庫已載入")
        except Exception as e:
            print(f"❌ 物品註冊資料庫初始化失敗: {e}")
            self.object_registry = None
    
    # ========================================
    # 物品註冊功能
    # ========================================
    
    def register_object(
        self,
        name: str,
        name_zh: str,
        image: np.ndarray
    ) -> Optional[Dict]:
        """
        註冊新物品
        
        Args:
            name: 物品英文名稱
            name_zh: 物品中文名稱
            image: 物品圖片 (已裁切的物品區域)
            
        Returns:
            註冊結果或 None
        """
        if not self.feature_extractor or not self.object_registry:
            return None
        
        try:
            # 提取特徵
            embedding = self.feature_extractor.extract_features(image)
            
            # 將圖片編碼為 bytes
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            
            # 註冊到資料庫
            obj = self.object_registry.register(
                name=name,
                name_zh=name_zh,
                embedding=embedding,
                image_data=image_bytes
            )
            
            return {
                "id": obj.id,
                "name": obj.name,
                "name_zh": obj.name_zh,
                "embedding_count": len(obj.embeddings)
            }
        except Exception as e:
            print(f"❌ 註冊物品失敗: {e}")
            return None
    
    def add_object_image(
        self,
        obj_id: str,
        image: np.ndarray
    ) -> Optional[Dict]:
        """為已註冊物品新增照片"""
        if not self.feature_extractor or not self.object_registry:
            return None
        
        try:
            embedding = self.feature_extractor.extract_features(image)
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            
            obj = self.object_registry.add_embedding(
                obj_id=obj_id,
                embedding=embedding,
                image_data=image_bytes
            )
            
            if obj:
                return {
                    "id": obj.id,
                    "name": obj.name,
                    "embedding_count": len(obj.embeddings)
                }
            return None
        except Exception as e:
            print(f"❌ 新增物品照片失敗: {e}")
            return None
    
    def get_registered_objects(self) -> List[Dict]:
        """取得所有已註冊物品"""
        if not self.object_registry:
            return []
        return self.object_registry.to_api_response()
    
    def delete_object(self, obj_id: str) -> bool:
        """刪除已註冊物品"""
        if not self.object_registry:
            return False
        return self.object_registry.delete(obj_id)
    
    # ========================================
    # 偵測功能
    # ========================================
    
    async def detect_snapshot(self, save_image: bool = True) -> Tuple[List[Detection], Optional[str]]:
        """
        從攝影機擷取快照並進行偵測
        
        Returns:
            tuple: (detections, image_path)
        """
        if not YOLO_AVAILABLE or self.model is None:
            return self._get_mock_detections(), None
        
        try:
            # 開啟攝影機
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                print("⚠️ 無法開啟攝影機，使用模擬資料")
                return self._get_mock_detections(), None
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return self._get_mock_detections(), None
            
            # 執行偵測
            detections = self._detect_frame(frame)
            
            # 儲存截圖
            image_path = None
            if save_image:
                image_path = self._save_snapshot(frame, detections)
            
            return detections, image_path
            
        except Exception as e:
            print(f"❌ 偵測失敗: {e}")
            import traceback
            traceback.print_exc()
            return self._get_mock_detections(), None
    
    def _detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """對單幀影像進行偵測並匹配用戶物品"""
        if self.model is None:
            return []
        
        # YOLO12 偵測
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            names = r.names if hasattr(r, 'names') else {}
            
            for box, conf, cls in zip(boxes, confs, clss):
                cls_id = int(cls)
                class_name = names.get(cls_id, f"class_{cls_id}")
                class_name_zh = COCO_CLASSES_ZH.get(class_name, class_name)
                
                bbox = [float(x) for x in box]
                
                # 建立基礎偵測結果
                detection = Detection(
                    object_class=class_name,
                    object_class_zh=class_name_zh,
                    confidence=float(conf),
                    bbox=bbox,
                    timestamp=int(datetime.now().timestamp() * 1000)
                )
                
                # 嘗試匹配用戶註冊的物品
                if self.feature_extractor and self.object_registry:
                    match_result = self._match_object(frame, bbox)
                    if match_result:
                        detection.matched_object_id = match_result["id"]
                        detection.matched_object_name = match_result["name"]
                        detection.matched_object_name_zh = match_result["name_zh"]
                        detection.similarity = match_result["similarity"]
                
                detections.append(detection)
        
        return detections
    
    def _match_object(
        self, 
        frame: np.ndarray, 
        bbox: List[float]
    ) -> Optional[Dict]:
        """匹配偵測到的物件與用戶註冊的物品"""
        if not self.object_registry.objects:
            return None
        
        try:
            # 裁切物件區域
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            # 確保邊界在圖片範圍內
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            # 提取特徵
            embedding = self.feature_extractor.extract_features(cropped)
            
            # 匹配
            match = self.object_registry.find_match(
                embedding, 
                threshold=self.similarity_threshold
            )
            
            if match:
                obj, similarity = match
                return {
                    "id": obj.id,
                    "name": obj.name,
                    "name_zh": obj.name_zh,
                    "similarity": similarity
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ 物件匹配失敗: {e}")
            return None
    
    def _save_snapshot(self, frame: np.ndarray, detections: List[Detection]) -> str:
        """儲存截圖並在圖片上畫出偵測框"""
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
        
        frame_with_boxes = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det.bbox]
            
            # 根據是否匹配到用戶物品選擇顏色
            if det.matched_object_id:
                color = (0, 255, 0)  # 綠色：匹配到用戶物品
                label = f"{det.matched_object_name_zh or det.matched_object_name} {det.similarity:.0%}"
            else:
                color = (128, 128, 128)  # 灰色：未匹配
                label = f"{det.object_class} {det.confidence:.0%}"
            
            # 畫框
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 畫標籤背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_with_boxes, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame_with_boxes, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 儲存圖片
        filename = f"snapshot_{int(datetime.now().timestamp() * 1000)}.jpg"
        filepath = os.path.join(static_dir, filename)
        cv2.imwrite(filepath, frame_with_boxes)
        
        print(f"📸 截圖已儲存: {filename}")
        return f"/static/{filename}"
    
    def _get_mock_detections(self) -> List[Detection]:
        """產生模擬偵測資料（用於測試）"""
        import random
        
        mock_items = [
            ("cell phone", "手機", 0.95),
            ("remote", "遙控器", 0.88),
            ("book", "書", 0.92),
            ("cup", "杯子", 0.85),
            ("bottle", "瓶子", 0.90),
        ]
        
        selected = random.sample(mock_items, k=min(random.randint(1, 3), len(mock_items)))
        
        detections = []
        for item in selected:
            det = Detection(
                object_class=item[0],
                object_class_zh=item[1],
                confidence=item[2] + random.uniform(-0.05, 0.05),
                bbox=[100.0, 100.0, 200.0, 200.0],
                timestamp=int(datetime.now().timestamp() * 1000)
            )
            
            # 模擬匹配
            if self.object_registry and random.random() > 0.5:
                objects = self.object_registry.get_all()
                if objects:
                    obj = random.choice(objects)
                    det.matched_object_id = obj.id
                    det.matched_object_name = obj.name
                    det.matched_object_name_zh = obj.name_zh
                    det.similarity = random.uniform(0.7, 0.95)
            
            detections.append(det)
        
        return detections
    
    # ========================================
    # 相容性 API (供 main.py 使用)
    # ========================================
    
    def get_class_name_zh(self, class_name: str) -> str:
        """取得類別的中文名稱"""
        return COCO_CLASSES_ZH.get(class_name, class_name)
