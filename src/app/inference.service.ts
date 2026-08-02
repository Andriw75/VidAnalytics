import { Injectable } from '@angular/core';
import { Detection, ModelMetadata } from './detection';
import { drawPose } from './pose-renderer';

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
  'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

const CLASS_COLORS = [
  '#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
  '#FF8800', '#88FF00', '#0088FF', '#FF0088', '#8800FF', '#00FF88',
];

type ImageSource = HTMLImageElement | ImageBitmap;

interface Pending {
  resolve: (detections: Detection[]) => void;
  reject: (error: Error) => void;
}

interface WorkerMessage {
  type?: string;
  id?: number;
  shape?: string;
  detections?: Detection[];
  error?: string;
  message?: string;
  metadata?: ModelMetadata;
}

@Injectable({ providedIn: 'root' })
export class InferenceService {
  private worker: Worker | null = null;
  private modelName: string | null = null;
  private modelMetadata: ModelMetadata | null = null;
  private nextId = 1;
  private inferenceGeneration = 0;
  private pending = new Map<number, Pending>();
  private readyResolve: ((metadata: ModelMetadata) => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private shape = '';

  get isReady(): boolean {
    return this.shape !== '';
  }

  get metadata(): ModelMetadata | null {
    return this.modelMetadata;
  }

  async init(): Promise<void> {
    if (this.worker) return;
    this.worker = new Worker('/inference.worker.js');
    this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      this.handleMessage(event.data);
    };
    this.worker.onerror = (event) => {
      this.failAll(new Error(`Error del worker de inferencia: ${event.message || 'desconocido'}`));
    };
  }

  private handleMessage(msg: WorkerMessage): void {
    if (!msg?.type) return;

    if (msg.type === 'ready') {
      this.shape = msg.shape ?? '';
      if (msg.metadata) this.modelMetadata = msg.metadata;
      if (this.modelMetadata) this.readyResolve?.(this.modelMetadata);
      this.readyResolve = null;
      this.readyReject = null;
    } else if (msg.type === 'result') {
      const p = msg.id != null ? this.pending.get(msg.id) : undefined;
      if (p) {
        this.pending.delete(msg.id!);
        if (msg.error) p.reject(new Error(msg.error));
        else p.resolve(msg.detections ?? []);
      }
    } else if (msg.type === 'error') {
      this.failAll(new Error(msg.error || msg.message || 'Error del worker de inferencia'));
    }
  }

  private failAll(error: Error): void {
    const worker = this.worker;
    this.worker = null;
    worker?.terminate();
    for (const [, p] of this.pending) p.reject(error);
    this.pending.clear();
    this.readyReject?.(error);
    this.readyResolve = null;
    this.readyReject = null;
    this.shape = '';
  }

  cancelPending(): void {
    this.inferenceGeneration++;
    if (this.pending.size === 0) return;
    const error = new Error('Inferencia cancelada');
    this.failAll(error);
  }

  async loadModel(name: string, metadata: ModelMetadata): Promise<ModelMetadata> {
    this.modelName = name;
    this.modelMetadata = metadata;
    await this.init();
    const metadataPromise = new Promise<ModelMetadata>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.worker!.postMessage({ type: 'init', model: name, metadata });
    return metadataPromise;
  }

  async inferImage(source: ImageSource): Promise<Detection[]> {
    const generation = this.inferenceGeneration;
    if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      return this.inferBitmap(source, generation);
    }
    const bitmap = await createImageBitmap(source as HTMLImageElement);
    if (generation !== this.inferenceGeneration) {
      bitmap.close();
      throw new Error('Inferencia cancelada');
    }
    return this.inferBitmap(bitmap, generation);
  }

  async inferBlob(blob: Blob): Promise<Detection[]> {
    const generation = this.inferenceGeneration;
    const bitmap = await createImageBitmap(blob);
    if (generation !== this.inferenceGeneration) {
      bitmap.close();
      throw new Error('Inferencia cancelada');
    }
    return this.inferBitmap(bitmap, generation);
  }

  private inferBitmap(bitmap: ImageBitmap, generation: number): Promise<Detection[]> {
    return this.inferBitmapAsync(bitmap, generation);
  }

  private async inferBitmapAsync(bitmap: ImageBitmap, generation: number): Promise<Detection[]> {
    try {
      if (generation !== this.inferenceGeneration) throw new Error('Inferencia cancelada');
      await this.ensureReady();
      if (generation !== this.inferenceGeneration) throw new Error('Inferencia cancelada');
      const id = this.nextId++;
      return await new Promise<Detection[]>((resolve, reject) => {
        if (!this.worker) {
          reject(new Error('Worker de inferencia no inicializado'));
          return;
        }
        this.pending.set(id, { resolve, reject });
        this.worker.postMessage({ type: 'infer', id, bitmap }, [bitmap]);
      });
    } catch (error) {
      try {
        bitmap.close();
      } catch {
        // The bitmap may already be detached after transfer to the worker.
      }
      throw error;
    }
  }

  private async ensureReady(): Promise<void> {
    if (this.worker && this.isReady) return;
    if (!this.modelName) throw new Error('Modelo no seleccionado');
    await this.init();
    if (!this.isReady && this.modelMetadata) await this.loadModel(this.modelName, this.modelMetadata);
  }

  cocoClass(classId: number): string {
    return COCO_CLASSES[classId] ?? `class_${classId}`;
  }

  drawFrame(
    canvas: HTMLCanvasElement,
    source: ImageSource,
    detections: Detection[] | undefined
  ): void {
    const imgW = 'naturalWidth' in source ? source.naturalWidth : source.width;
    const imgH = 'naturalHeight' in source ? source.naturalHeight : source.height;
    canvas.width = imgW;
    canvas.height = imgH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, imgW, imgH);
    ctx.drawImage(source, 0, 0);

    if (!detections) return;

    const lineWidth = Math.max(2, Math.round(imgW / 400));
    ctx.lineWidth = lineWidth;

    for (const d of detections) {
      const color = CLASS_COLORS[d.classId % CLASS_COLORS.length];
      ctx.strokeStyle = color;
      ctx.strokeRect(d.x, d.y, d.width, d.height);

      const label = `${COCO_CLASSES[d.classId] ?? d.classId} ${(d.confidence * 100).toFixed(1)}%`;
      ctx.font = `${Math.max(12, Math.round(imgW / 60))}px Arial`;
      const textW = ctx.measureText(label).width;
      const labelH = Math.max(16, Math.round(imgW / 45));
      ctx.fillStyle = color;
      ctx.fillRect(d.x, d.y - labelH, textW + 8, labelH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, d.x + 4, d.y - 4);
      if (d.keypoints?.length) drawPose(ctx, d.keypoints, imgW);
    }
  }
}
