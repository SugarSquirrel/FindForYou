/**
 * IndexedDB 資料庫操作模組
 * 用於儲存和查詢物品偵測記錄
 */

const DB_NAME = 'ObjectFinderDB';
const DB_VERSION = 1;

// 資料表名稱
const STORES = {
    DETECTIONS: 'detections',  // 所有偵測記錄
    OBJECTS: 'objects'         // 物品最後位置
};

// 物品類別中英對照
const OBJECT_CLASS_MAP = {
    'cell phone': '手機',
    'phone': '手機',
    'remote': '遙控器',
    'bottle': '水瓶',
    'cup': '杯子',
    'book': '書',
    'clock': '時鐘',
    'scissors': '剪刀',
    'glasses': '眼鏡',
    'keys': '鑰匙',
    'wallet': '錢包',
    // YOLO-World 新增類別
    'medicine bottle': '藥罐',
    'hearing aid': '助聽器',
    'denture case': '假牙盒',
    'pen': '筆',
    'notebook': '筆記本',
    'tissue box': '面紙盒'
};

// 區域中英對照
const REGION_MAP = {
    'left': '左側',
    'center': '中間',
    'right': '右側'
};

// 表面中英對照
const SURFACE_MAP = {
    'sofa': '沙發',
    'table': '桌子',
    'cabinet': '櫃子',
    'desk': '書桌',
    'bed': '床',
    'chair': '椅子'
};

/**
 * ObjectFinderDB 類別
 * 封裝 IndexedDB 操作
 */
class ObjectFinderDB {
    constructor() {
        this.db = null;
        this.isReady = false;
    }

    /**
     * 初始化資料庫
     */
    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => {
                console.error('IndexedDB 開啟失敗:', request.error);
                reject(request.error);
            };

            request.onsuccess = async () => {
                this.db = request.result;
                this.isReady = true;
                
                // 不再每次啟動清空資料庫，保留偵測記錄
                console.log('IndexedDB 初始化成功');
                
                resolve(this);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // 版本升級時刪除舊的 store 並重新建立
                if (db.objectStoreNames.contains(STORES.DETECTIONS)) {
                    db.deleteObjectStore(STORES.DETECTIONS);
                }
                if (db.objectStoreNames.contains(STORES.OBJECTS)) {
                    db.deleteObjectStore(STORES.OBJECTS);
                }

                // 建立 detections store
                const detectionsStore = db.createObjectStore(STORES.DETECTIONS, { 
                    keyPath: 'id',
                    autoIncrement: true 
                });
                detectionsStore.createIndex('timestamp', 'timestamp', { unique: false });
                detectionsStore.createIndex('objectClass', 'objectClass', { unique: false });
                detectionsStore.createIndex('objectClassZh', 'objectClassZh', { unique: false });

                // 建立 objects store (物品最後位置)
                const objectsStore = db.createObjectStore(STORES.OBJECTS, { 
                    keyPath: 'objectClass' 
                });
                objectsStore.createIndex('lastSeen', 'lastSeen', { unique: false });
                objectsStore.createIndex('objectClassZh', 'objectClassZh', { unique: false });

                console.log('IndexedDB 結構已建立');
            };
        });
    }

    /**
     * 清空所有資料
     */
    async clearAll() {
        if (!this.db) return;
        
        return new Promise((resolve) => {
            try {
                const transaction = this.db.transaction([STORES.DETECTIONS, STORES.OBJECTS], 'readwrite');
                
                transaction.objectStore(STORES.DETECTIONS).clear();
                transaction.objectStore(STORES.OBJECTS).clear();
                
                transaction.oncomplete = () => {
                    console.log('📭 資料庫已清空');
                    resolve();
                };
                
                transaction.onerror = () => {
                    console.error('清空資料庫失敗');
                    resolve();
                };
            } catch (e) {
                console.error('clearAll error:', e);
                resolve();
            }
        });
    }

    /**
     * 儲存偵測結果
     * @param {Object} detection 偵測資料
     */
    async saveDetection(detection) {
        if (!this.isReady) await this.init();

        // 記憶體層去重鎖：防止 API 和 WebSocket 同時寫入
        if (!this.savingLocks) this.savingLocks = new Set();
        const lockKey = `${detection.timestamp}-${detection.objectClass}`;
        if (this.savingLocks.has(lockKey)) {
            // console.log(`🔒 忽略並發儲存: ${lockKey}`);
            return Promise.resolve(detection);
        }
        this.savingLocks.add(lockKey);
        // 5秒後釋放鎖（足夠覆蓋並發時間差）
        setTimeout(() => this.savingLocks.delete(lockKey), 5000);

        const record = {
            timestamp: detection.timestamp || Date.now(),
            objectClass: detection.objectClass,
            // 優先使用傳入的中文名稱（匹配物品時會有自定義名稱）
            objectClassZh: detection.objectClassZh || OBJECT_CLASS_MAP[detection.objectClass] || detection.objectClass,
            confidence: detection.confidence,
            bbox: detection.bbox,
            surface: detection.surface,
            // 如果 surface 已經是中文（不在英文映射表中），就直接使用
            surfaceZh: SURFACE_MAP[detection.surface] || detection.surface || '未知位置',
            region: detection.region || '',
            regionZh: REGION_MAP[detection.region] || detection.region || '',
            imagePath: detection.imagePath || null,  // 帶標註的圖片 (顯示用)
            imageOriginal: detection.imageOriginal || null,  // 原始圖片 (註冊用)
            matchedObjectId: detection.matchedObjectId || null,
            matchedObjectName: detection.matchedObjectName || null
        };

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORES.DETECTIONS, STORES.OBJECTS], 'readwrite');
            const detectionsStore = transaction.objectStore(STORES.DETECTIONS);
            
            // 1. 先透過 timestamp 查詢是否已有相同記錄（去重關鍵）
            const index = detectionsStore.index('timestamp');
            const checkRequest = index.getAll(record.timestamp);
            
            checkRequest.onsuccess = () => {
                const existing = checkRequest.result;
                const isDuplicate = existing && existing.some(d => d.objectClass === record.objectClass);
                
                if (isDuplicate) {
                    console.log(`♻️ 忽略重複記錄: ${record.objectClass} (${record.timestamp})`);
                    return; // 直接返回，不執行 add，交易會自動結束
                }
                
                // 2. 無重複則新增
                const addRequest = detectionsStore.add(record);

                addRequest.onsuccess = () => {
                    // 3. 更新 objects store 的最後位置
                    const objectsStore = transaction.objectStore(STORES.OBJECTS);
                    const objectRecord = {
                        objectClass: record.objectClass,
                        objectClassZh: record.objectClassZh,
                        lastSeen: record.timestamp,
                        surface: record.surface,
                        surfaceZh: record.surfaceZh,
                        region: record.region,
                        regionZh: record.regionZh,
                        confidence: record.confidence,
                        description: `${record.objectClassZh}在${record.surfaceZh}${record.regionZh}`,
                        imagePath: record.imagePath
                    };
                    objectsStore.put(objectRecord);
                };
            };

            transaction.oncomplete = () => {
                resolve(record);
            };

            transaction.onerror = () => {
                console.error('儲存失敗:', transaction.error);
                reject(transaction.error);
            };
        });
    }

    /**
     * 查詢物品最後位置
     * @param {string} query 查詢字串 (中文或英文)
     */
    async getLastLocation(query) {
        if (!this.isReady) await this.init();

        const normalizedQuery = query.toLowerCase().trim();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.OBJECTS, 'readonly');
            const store = transaction.objectStore(STORES.OBJECTS);
            const request = store.openCursor();
            
            let result = null;

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    const record = cursor.value;
                    // 支援中英文搜尋
                    if (record.objectClass.toLowerCase().includes(normalizedQuery) ||
                        record.objectClassZh.includes(query)) {
                        result = record;
                    }
                    cursor.continue();
                } else {
                    resolve(result);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    /**
     * 取得物品歷史記錄
     * @param {string} objectClass 物品類別
     * @param {number} limit 筆數限制
     */
    async getHistory(objectClass, limit = 10) {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.DETECTIONS, 'readonly');
            const store = transaction.objectStore(STORES.DETECTIONS);
            const index = store.index('objectClass');
            const request = index.openCursor(IDBKeyRange.only(objectClass), 'prev');
            
            const results = [];

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor && results.length < limit) {
                    results.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    /**
     * 取得所有已知物品
     */
    async getAllObjects() {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.OBJECTS, 'readonly');
            const store = transaction.objectStore(STORES.OBJECTS);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    /**
     * 取得最近偵測記錄
     * @param {number} limit 筆數限制
     */
    async getRecentDetections(limit = 10) {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.DETECTIONS, 'readonly');
            const store = transaction.objectStore(STORES.DETECTIONS);
            const index = store.index('timestamp');
            const request = index.openCursor(null, 'prev');
            
            const results = [];

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor && results.length < limit) {
                    results.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    /**
     * 清除所有資料
     */
    async clearAll() {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORES.DETECTIONS, STORES.OBJECTS], 'readwrite');
            
            transaction.objectStore(STORES.DETECTIONS).clear();
            transaction.objectStore(STORES.OBJECTS).clear();

            transaction.oncomplete = () => {
                console.log('所有資料已清除');
                resolve();
            };

            transaction.onerror = () => reject(transaction.error);
        });
    }

    /**
     * 取得所有偵測記錄（依時間倒序）
     * @param {number} limit 最多回傳筆數
     */
    async getAllDetections(limit = 100) {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.DETECTIONS, 'readonly');
            const store = transaction.objectStore(STORES.DETECTIONS);
            const index = store.index('timestamp');
            const request = index.openCursor(null, 'prev');
            
            const results = [];

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor && results.length < limit) {
                    results.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    /**
     * 取得特定物品的歷史記錄
     * @param {string} objectClass 物品類別（英文）
     * @param {number} limit 最多回傳筆數
     */
    async getObjectHistory(objectClass, limit = 50) {
        if (!this.isReady) await this.init();

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(STORES.DETECTIONS, 'readonly');
            const store = transaction.objectStore(STORES.DETECTIONS);
            const index = store.index('timestamp');
            const request = index.openCursor(null, 'prev');
            
            const results = [];
            const normalizedClass = objectClass.toLowerCase();

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    if (cursor.value.objectClass.toLowerCase() === normalizedClass && results.length < limit) {
                        results.push(cursor.value);
                    }
                    cursor.continue();
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }
}

// 匯出全域實例
window.objectFinderDB = new ObjectFinderDB();
