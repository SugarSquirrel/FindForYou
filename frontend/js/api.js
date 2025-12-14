/**
 * API 通訊模組
 * 前端攝影機架構：傳送圖片到後端進行推論
 */

const API_CONFIG = {
    baseUrl: '',  // Same origin
    endpoints: {
        detectImage: '/api/detect/image',
        health: '/api/health',
        objects: '/api/objects',
        registerObject: '/api/objects/register'
    },
    timeout: 30000  // 偵測可能需要較長時間
};

/**
 * ObjectFinderAPI 類別
 */
class ObjectFinderAPI {
    constructor(baseUrl = API_CONFIG.baseUrl) {
        this.baseUrl = baseUrl;
        this.isConnected = false;
    }

    /**
     * 發送 HTTP 請求 (JSON)
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), API_CONFIG.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeout);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeout);
            if (error.name === 'AbortError') {
                throw new Error('請求逾時');
            }
            throw error;
        }
    }

    /**
     * 檢查後端服務狀態
     */
    async checkHealth() {
        try {
            const result = await this.request(API_CONFIG.endpoints.health);
            this.isConnected = true;
            return result;
        } catch (error) {
            this.isConnected = false;
            console.warn('後端服務未連線:', error.message);
            return null;
        }
    }

    /**
     * 偵測圖片中的物品（主要 API）
     * @param {Blob} imageBlob 圖片 Blob
     * @returns {Promise<Object>} 偵測結果
     */
    async detectImage(imageBlob) {
        const url = `${this.baseUrl}${API_CONFIG.endpoints.detectImage}`;
        const formData = new FormData();
        formData.append('file', imageBlob, 'frame.jpg');

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), API_CONFIG.timeout);

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            clearTimeout(timeout);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeout);
            if (error.name === 'AbortError') {
                throw new Error('偵測逾時');
            }
            throw error;
        }
    }

    /**
     * 取得已註冊物品列表
     */
    async getObjects() {
        try {
            const result = await this.request(API_CONFIG.endpoints.objects);
            return result;
        } catch (error) {
            console.error('取得物品列表失敗:', error);
            throw error;
        }
    }

    /**
     * 註冊新物品
     * @param {Blob} imageBlob 圖片 Blob
     * @param {string} name 物品英文名稱
     * @param {string} nameZh 物品中文名稱
     */
    async registerObject(imageBlob, name, nameZh) {
        const url = `${this.baseUrl}${API_CONFIG.endpoints.registerObject}`;
        const formData = new FormData();
        formData.append('image', imageBlob);
        formData.append('name', name);
        formData.append('name_zh', nameZh);

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('註冊物品失敗:', error);
            throw error;
        }
    }

    /**
     * 取得連線狀態
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            baseUrl: this.baseUrl
        };
    }
}

// 匯出全域實例
window.objectFinderAPI = new ObjectFinderAPI();
