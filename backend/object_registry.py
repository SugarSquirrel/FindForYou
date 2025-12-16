"""
物品註冊資料庫模組
管理用戶註冊的個人物品及其特徵嵌入
"""

import os
import json
import uuid
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np


# 配置檔路徑
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registered_objects.json")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "object_images")


@dataclass
class RegisteredObject:
    """已註冊物品資料類別"""
    id: str
    name: str
    name_zh: str
    embeddings: List[List[float]] = field(default_factory=list)  # 多個特徵向量
    images: List[str] = field(default_factory=list)  # 圖片路徑列表
    created_at: int = 0
    updated_at: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(data: dict) -> 'RegisteredObject':
        return RegisteredObject(**data)
    



class ObjectRegistry:
    """物品註冊資料庫"""
    
    def __init__(self, registry_path: str = REGISTRY_PATH):
        self.registry_path = registry_path
        self.objects: Dict[str, RegisteredObject] = {}
        
        # 確保圖片目錄存在
        os.makedirs(IMAGES_DIR, exist_ok=True)
        
        # 載入現有資料
        self._load()
    
    def _load(self):
        """載入註冊資料"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.objects = {
                    k: RegisteredObject.from_dict(v) 
                    for k, v in data.get("objects", {}).items()
                }
                print(f"✅ 已載入 {len(self.objects)} 個註冊物品")
            else:
                print("📝 建立新的物品註冊資料庫")
                self.objects = {}
                self._save()
        except Exception as e:
            print(f"⚠️ 載入物品註冊資料失敗: {e}")
            self.objects = {}
    
    def _save(self):
        """儲存註冊資料"""
        try:
            data = {
                "objects": {k: v.to_dict() for k, v in self.objects.items()},
                "version": "2.0",
                "updated_at": int(datetime.now().timestamp() * 1000)
            }
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 儲存物品註冊資料失敗: {e}")
    
    def register(
        self,
        name: str,
        name_zh: str,
        embedding: np.ndarray,
        image_data: bytes = None,
        image_path: str = None
    ) -> RegisteredObject:
        """
        註冊新物品
        
        Args:
            name: 物品英文名稱
            name_zh: 物品中文名稱
            embedding: 特徵向量
            image_data: 圖片二進位資料 (可選)
            image_path: 已存在的圖片路徑 (可選)
            
        Returns:
            註冊的物品物件
        """
        now = int(datetime.now().timestamp() * 1000)
        
        # 使用 UUID 生成唯一 ID，確保每個物品都有獨立的 ID
        obj_id = str(uuid.uuid4())
        
        # 儲存圖片
        saved_image_path = None
        if image_data:
            saved_image_path = self._save_image(obj_id, image_data)
        elif image_path and os.path.exists(image_path):
            saved_image_path = image_path
        
        # 建立新物品（每次註冊都是新物品）
        obj = RegisteredObject(
            id=obj_id,
            name=name,
            name_zh=name_zh,
            embeddings=[embedding.tolist()],
            images=[saved_image_path] if saved_image_path else [],
            created_at=now,
            updated_at=now
        )
        self.objects[obj_id] = obj
        print(f"✅ 註冊新物品: {name} ({name_zh}) [ID: {obj_id[:8]}...]")
        
        self._save()
        return obj
    
    def _save_image(self, obj_id: str, image_data: bytes) -> str:
        """儲存物品圖片"""
        timestamp = int(datetime.now().timestamp() * 1000)
        filename = f"{obj_id}_{timestamp}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filepath
    
    def get(self, obj_id: str) -> Optional[RegisteredObject]:
        """取得單一物品"""
        return self.objects.get(obj_id)
    
    def get_all(self) -> List[RegisteredObject]:
        """取得所有已註冊物品"""
        return list(self.objects.values())
    
    def update(
        self,
        obj_id: str,
        name: str = None,
        name_zh: str = None
    ) -> Optional[RegisteredObject]:
        """更新物品資訊"""
        if obj_id not in self.objects:
            return None
        
        obj = self.objects[obj_id]
        if name:
            obj.name = name
        if name_zh:
            obj.name_zh = name_zh
        obj.updated_at = int(datetime.now().timestamp() * 1000)
        
        self._save()
        return obj
    
    def add_embedding(
        self,
        obj_id: str,
        embedding: np.ndarray,
        image_data: bytes = None,
        image_path: str = None
    ) -> Optional[RegisteredObject]:
        """
        為物品新增特徵 (多張照片)
        
        Args:
            obj_id: 物品 ID
            embedding: 特徵向量
            image_data: 圖片二進位資料 (可選)
            image_path: 已存在的圖片路徑 (可選，傳入此參數時不會重新儲存圖片)
        """
        if obj_id not in self.objects:
            return None
        
        obj = self.objects[obj_id]
        obj.embeddings.append(embedding.tolist())
        
        # 處理圖片：優先使用已存在的路徑，否則儲存 image_data
        if image_path and os.path.exists(image_path):
            obj.images.append(image_path)
        elif image_data:
            saved_path = self._save_image(obj_id, image_data)
            obj.images.append(saved_path)
        
        obj.updated_at = int(datetime.now().timestamp() * 1000)
        self._save()
        
        print(f"📝 物品 {obj.name} 新增特徵 (共 {len(obj.embeddings)} 個)")
        return obj
    
    def delete(self, obj_id: str) -> bool:
        """刪除物品"""
        if obj_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        
        # 刪除關聯的圖片
        for img_path in obj.images:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass
        
        del self.objects[obj_id]
        self._save()
        
        print(f"🗑️ 已刪除物品: {obj.name}")
        return True
    
    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.7
    ) -> Optional[tuple]:
        """
        在已註冊物品中找到最佳匹配
        對每個物件的所有特徵向量分別計算相似度，取最大值
        
        Args:
            query_embedding: 查詢特徵向量
            threshold: 相似度閾值
            
        Returns:
            (物品, 相似度) 或 None
        """
        if not self.objects:
            return None
        
        best_obj = None
        best_sim = 0.0
        
        for obj in self.objects.values():
            if not obj.embeddings:
                continue
            
            # 對該物件的所有特徵向量計算相似度，取最大值
            max_sim = 0.0
            for emb in obj.embeddings:
                emb_array = np.array(emb).flatten()
                sim = float(np.dot(query_embedding.flatten(), emb_array))
                max_sim = max(max_sim, sim)
            
            if max_sim > best_sim:
                best_sim = max_sim
                best_obj = obj
        
        if best_sim >= threshold and best_obj:
            return (best_obj, best_sim)
        
        return None
    
    def find_all_matches(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5
    ) -> List[tuple]:
        """
        找到所有超過閾值的匹配物品
        對每個物件的所有特徵向量分別計算相似度，取最大值
        
        Returns:
            [(物品, 相似度), ...] 按相似度降序排序
        """
        matches = []
        
        for obj in self.objects.values():
            if not obj.embeddings:
                continue
            
            # 對該物件的所有特徵向量計算相似度，取最大值
            max_sim = 0.0
            for emb in obj.embeddings:
                emb_array = np.array(emb).flatten()
                sim = float(np.dot(query_embedding.flatten(), emb_array))
                max_sim = max(max_sim, sim)
            
            if max_sim >= threshold:
                matches.append((obj, max_sim))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """取得所有物品的所有特徵向量（不計算平均值）"""
        result = {}
        for obj_id, obj in self.objects.items():
            if obj.embeddings:
                result[obj_id] = [np.array(emb) for emb in obj.embeddings]
        return result
    
    def to_api_response(self) -> List[Dict[str, Any]]:
        """轉換為 API 回應格式"""
        result = []
        for obj in self.objects.values():
            # 取得第一張圖片作為縮圖
            thumbnail = None
            if obj.images and os.path.exists(obj.images[0]):
                # 返回相對路徑
                thumbnail = f"/object_images/{os.path.basename(obj.images[0])}"
            
            result.append({
                "id": obj.id,
                "name": obj.name,
                "name_zh": obj.name_zh,
                "thumbnail": thumbnail,
                "image_count": len(obj.images),
                "embedding_count": len(obj.embeddings),
                "created_at": obj.created_at,
                "updated_at": obj.updated_at
            })
        
        return result
