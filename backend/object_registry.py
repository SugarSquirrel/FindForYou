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
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """取得平均特徵向量"""
        if not self.embeddings:
            return None
        
        embeddings_array = np.array(self.embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)
        # 正規化
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding


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
            name: 物品英文名稱 (作為 ID)
            name_zh: 物品中文名稱
            embedding: 特徵向量
            image_data: 圖片二進位資料 (可選)
            image_path: 已存在的圖片路徑 (可選)
            
        Returns:
            註冊的物品物件
        """
        now = int(datetime.now().timestamp() * 1000)
        obj_id = name.lower().replace(" ", "_")
        
        # 儲存圖片
        saved_image_path = None
        if image_data:
            saved_image_path = self._save_image(obj_id, image_data)
        elif image_path and os.path.exists(image_path):
            saved_image_path = image_path
        
        # 檢查是否已存在
        if obj_id in self.objects:
            # 更新現有物品
            obj = self.objects[obj_id]
            obj.embeddings.append(embedding.tolist())
            if saved_image_path:
                obj.images.append(saved_image_path)
            obj.updated_at = now
            print(f"📝 更新物品: {name} (共 {len(obj.embeddings)} 個特徵)")
        else:
            # 建立新物品
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
            print(f"✅ 註冊新物品: {name} ({name_zh})")
        
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
        image_data: bytes = None
    ) -> Optional[RegisteredObject]:
        """為物品新增特徵 (多張照片)"""
        if obj_id not in self.objects:
            return None
        
        obj = self.objects[obj_id]
        obj.embeddings.append(embedding.tolist())
        
        if image_data:
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
            avg_emb = obj.get_average_embedding()
            if avg_emb is None:
                continue
            
            # 計算餘弦相似度
            sim = float(np.dot(query_embedding.flatten(), avg_emb.flatten()))
            
            if sim > best_sim:
                best_sim = sim
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
        
        Returns:
            [(物品, 相似度), ...] 按相似度降序排序
        """
        matches = []
        
        for obj in self.objects.values():
            avg_emb = obj.get_average_embedding()
            if avg_emb is None:
                continue
            
            sim = float(np.dot(query_embedding.flatten(), avg_emb.flatten()))
            
            if sim >= threshold:
                matches.append((obj, sim))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """取得所有物品的平均特徵向量"""
        result = {}
        for obj_id, obj in self.objects.items():
            avg_emb = obj.get_average_embedding()
            if avg_emb is not None:
                result[obj_id] = avg_emb
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
