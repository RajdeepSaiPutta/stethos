// model-store.ts - IndexedDB wrapper for offline model caching
// Provides persistent storage for ML models in the browser

interface ModelEntry {
  name: string
  blob: Blob
  timestamp: number
}

interface MetadataEntry {
  key: string
  [key: string]: any
}

interface InferenceResultEntry {
  id?: number
  patient_id: string
  modality: string
  result: any
  timestamp: number
  synced: boolean
}

export class ModelStore {
  private dbName: string
  private version: number
  private db: IDBDatabase | null = null

  constructor(dbName = 'stethos-models', version = 1) {
    this.dbName = dbName
    this.version = version
  }

  async open(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version)
      request.onerror = () => reject(request.error)
      request.onsuccess = () => { this.db = request.result; resolve(); }
      request.onupgradeneeded = (e: IDBVersionChangeEvent) => {
        const db = (e.target as IDBOpenDBRequest).result
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'name' })
        }
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' })
        }
        if (!db.objectStoreNames.contains('inference_results')) {
          const store = db.createObjectStore('inference_results', { keyPath: 'id', autoIncrement: true })
          store.createIndex('patient_id', 'patient_id', { unique: false })
          store.createIndex('modality', 'modality', { unique: false })
          store.createIndex('timestamp', 'timestamp', { unique: false })
          store.createIndex('synced', 'synced', { unique: false })
        }
      }
    })
  }

  async putModel(name: string, modelBlob: Blob, metadata: Record<string, any> = {}): Promise<void> {
    if (!this.db) await this.open()
    const tx = this.db!.transaction(['models', 'metadata'], 'readwrite')
    await Promise.all([
      new Promise<void>((resolve, reject) => {
        const req = tx.objectStore('models').put({ name, blob: modelBlob, timestamp: Date.now() } as ModelEntry)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      }),
      new Promise<void>((resolve, reject) => {
        const req = tx.objectStore('metadata').put({ key: name, ...metadata } as MetadataEntry)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      })
    ])
  }

  async getModel(name: string): Promise<Blob | undefined> {
    if (!this.db) await this.open()
    return new Promise((resolve, reject) => {
      const req = this.db!.transaction('models').objectStore('models').get(name)
      req.onsuccess = () => resolve(req.result?.blob)
      req.onerror = () => reject(req.error)
    })
  }

  async getModelMetadata(name: string): Promise<MetadataEntry | undefined> {
    if (!this.db) await this.open()
    return new Promise((resolve, reject) => {
      const req = this.db!.transaction('metadata').objectStore('metadata').get(name)
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
  }

  async getAllModels(): Promise<ModelEntry[]> {
    if (!this.db) await this.open()
    return new Promise((resolve, reject) => {
      const req = this.db!.transaction('models').objectStore('models').getAll()
      req.onsuccess = () => resolve(req.result as ModelEntry[])
      req.onerror = () => reject(req.error)
    })
  }

  async deleteModel(name: string): Promise<void> {
    if (!this.db) await this.open()
    const tx = this.db!.transaction(['models', 'metadata'], 'readwrite')
    await Promise.all([
      new Promise<void>((resolve, reject) => {
        const req = tx.objectStore('models').delete(name)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      }),
      new Promise<void>((resolve, reject) => {
        const req = tx.objectStore('metadata').delete(name)
        req.onsuccess = () => resolve()
        req.onerror = () => reject(req.error)
      })
    ])
  }

  // Inference results storage for offline sync
  async saveInferenceResult(result: Omit<InferenceResultEntry, 'id' | 'timestamp' | 'synced'>): Promise<number> {
    if (!this.db) await this.open()
    const tx = this.db!.transaction('inference_results', 'readwrite')
    return new Promise((resolve, reject) => {
      const req = tx.objectStore('inference_results').put({
        ...result,
        timestamp: Date.now(),
        synced: false
      } as InferenceResultEntry)
      req.onsuccess = () => resolve(req.result as number)
      req.onerror = () => reject(req.error)
    })
  }

  async getPendingResults(): Promise<InferenceResultEntry[]> {
    if (!this.db) await this.open()
    return new Promise((resolve, reject) => {
      const req = this.db!.transaction('inference_results').objectStore('inference_results').index('synced').getAll()
      req.onsuccess = () => resolve(req.result as InferenceResultEntry[])
      req.onerror = () => reject(req.error)
    })
  }

  async markResultSynced(id: number): Promise<void> {
    if (!this.db) await this.open()
    const tx = this.db!.transaction('inference_results', 'readwrite')
    const store = tx.objectStore('inference_results')
    const req = store.get(id)
    req.onsuccess = () => {
      const result = req.result
      if (result) {
        result.synced = true
        store.put(result)
      }
    }
    req.onerror = () => {}
    return new Promise<void>((resolve) => { tx.oncomplete = () => resolve() })
  }

  async clearSyncedResults(): Promise<void> {
    if (!this.db) await this.open()
    const tx = this.db!.transaction('inference_results', 'readwrite')
    const store = tx.objectStore('inference_results')
    const index = store.index('synced')
    const req = index.getAll()
    req.onsuccess = () => {
      req.result.forEach(r => store.delete(r.id))
    }
    return new Promise<void>((resolve) => { tx.oncomplete = () => resolve() })
  }
}

export const modelStore = new ModelStore()