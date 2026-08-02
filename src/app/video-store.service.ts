import { Injectable } from '@angular/core';
import { Detection, ModelMetadata, ModelTask } from './detection';

export interface VideoSession {
  id?: number;
  name: string;
  createdAt: number;
  fileName: string;
  width: number;
  height: number;
  duration: number;
  fps: number | null;
  totalFrames: number | null;
  salto: number;
  count: number;
  thumbnail?: Blob;
  thumbUrl?: string;
  modelFile?: string;
  modelTask?: ModelTask;
  modelMetadata?: ModelMetadata;
}

export interface VideoFrameRecord {
  id?: number;
  sessionId: number;
  index: number;
  timestamp: number;
  blob: Blob;
  detections?: Detection[];
}

@Injectable({ providedIn: 'root' })
export class VideoStoreService {
  private db: IDBDatabase | null = null;

  private ensureDb(): Promise<IDBDatabase> {
    if (this.db) return Promise.resolve(this.db);
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('vidanalytics', 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('sessions')) {
          const store = db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
          store.createIndex('createdAt', 'createdAt');
        }
        if (!db.objectStoreNames.contains('frames')) {
          const store = db.createObjectStore('frames', { keyPath: 'id', autoIncrement: true });
          store.createIndex('sessionId', 'sessionId');
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve(this.db);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async saveSession(
    session: VideoSession,
    frames: Omit<VideoFrameRecord, 'id' | 'sessionId'>[]
  ): Promise<number> {
    const db = await this.ensureDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['sessions', 'frames'], 'readwrite');
      const sessionStore = tx.objectStore('sessions');
      const frameStore = tx.objectStore('frames');
      const sessionReq = sessionStore.add(session);
      sessionReq.onsuccess = () => {
        const sessionId = sessionReq.result as number;
        for (const f of frames) {
          frameStore.add({ ...f, sessionId });
        }
      };
      tx.oncomplete = () => resolve(sessionReq.result as number);
      tx.onerror = () => reject(tx.error);
    });
  }

  async listSessions(): Promise<VideoSession[]> {
    const db = await this.ensureDb();
    return new Promise((resolve, reject) => {
      const req = db.transaction('sessions', 'readonly').objectStore('sessions').getAll();
      req.onsuccess = () => {
        const sessions = (req.result as VideoSession[]).sort((a, b) => b.createdAt - a.createdAt);
        resolve(sessions);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async getFrames(sessionId: number): Promise<VideoFrameRecord[]> {
    const db = await this.ensureDb();
    return new Promise((resolve, reject) => {
      const index = db.transaction('frames', 'readonly').objectStore('frames').index('sessionId');
      const req = index.getAll(sessionId);
      req.onsuccess = () => {
        const frames = (req.result as VideoFrameRecord[]).sort((a, b) => a.index - b.index);
        resolve(frames);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async deleteSession(sessionId: number): Promise<void> {
    const db = await this.ensureDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['sessions', 'frames'], 'readwrite');
      tx.objectStore('sessions').delete(sessionId);
      const index = tx.objectStore('frames').index('sessionId');
      const req = index.openKeyCursor(sessionId);
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          tx.objectStore('frames').delete(cursor.primaryKey as number);
          cursor.continue();
        }
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async clearAll(): Promise<void> {
    const db = await this.ensureDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['sessions', 'frames'], 'readwrite');
      tx.objectStore('sessions').clear();
      tx.objectStore('frames').clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}
