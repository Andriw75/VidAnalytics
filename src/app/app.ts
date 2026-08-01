import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { loadLiteRt, loadAndCompile, Tensor, CompiledModel } from '@litertjs/core';
import { VideoStoreService, VideoSession, VideoFrameRecord } from './video-store.service';

interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  classId: number;
}

interface PadInfo {
  top: number;
  left: number;
  bottom: number;
  right: number;
}

interface VideoMeta {
  fileName: string;
  width: number;
  height: number;
  duration: number;
  fps: number | null;
  totalFrames: number | null;
}

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
  'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewInit, OnDestroy {
  @ViewChild('imageCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('sourceImage') imageRef!: ElementRef<HTMLImageElement>;
  @ViewChild('sourceVideo') videoRef!: ElementRef<HTMLVideoElement>;

  imageUrl: string | null = null;
  selectedModel = 'yolo26n.tflite';
  models = ['yolo26n.tflite'];
  status = 'Inicializando LiteRT...';
  isLoading = true;
  detections: Detection[] = [];
  processing = false;
  inferenceTime: number | null = null;
  inferenceDone = false;

  private model: CompiledModel | null = null;

  videoUrl: string | null = null;
  videoFileName: string | null = null;
  videoMeta: VideoMeta | null = null;
  videoLoading = false;

  salto = 15;
  extractEstimate: { fpsKept: number | null; count: number } | null = null;
  extracting = false;
  extractAbort = false;
  extractProgress: { done: number; total: number } | null = null;
  extractStatus = '';

  sessions: VideoSession[] = [];
  selectedSession: VideoSession | null = null;
  sessionFrames: { index: number; timestamp: number; url: string }[] = [];

  private readonly supportsWebp = this.detectWebp();

  constructor(private cdr: ChangeDetectorRef, private store: VideoStoreService) {}

  async ngAfterViewInit() {
    try {
      await loadLiteRt('/wasm/');
      this.status = 'LiteRT listo. Cargando modelo...';
      this.cdr.detectChanges();
      await this.loadModel();
    } catch (err) {
      this.status = `Error al iniciar LiteRT: ${err}`;
      this.isLoading = false;
      this.cdr.detectChanges();
    }
  }

  async loadModel() {
    try {
      if (this.model) {
        this.model.delete();
        this.model = null;
      }
      this.model = await loadAndCompile(`/models/${this.selectedModel}`, {
        accelerator: 'wasm',
      });
      const details = this.model.getInputDetails();
      const shape = Array.from(details[0].shape);
      this.status = `Modelo listo (entrada: ${shape.slice(1).join('x')}). Selecciona una imagen.`;
      this.isLoading = false;
      this.loadSessions();
      this.cdr.detectChanges();
    } catch (err) {
      this.status = `Error al cargar modelo: ${err}`;
      this.isLoading = false;
      this.cdr.detectChanges();
    }
  }

  async onModelChange(event: Event) {
    const select = event.target as HTMLSelectElement;
    this.selectedModel = select.value;
    this.isLoading = true;
    this.status = 'Cargando modelo...';
    this.detections = [];
    this.inferenceDone = false;
    this.inferenceTime = null;
    this.cdr.detectChanges();
    await this.loadModel();
  }

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
      this.imageUrl = URL.createObjectURL(file);
      this.detections = [];
      this.inferenceDone = false;
      this.inferenceTime = null;
      this.cdr.detectChanges();
    }
  }

  async onVideoSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    input.value = '';
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    if (this.videoRef) this.videoRef.nativeElement.pause();
    this.videoUrl = URL.createObjectURL(file);
    this.videoFileName = file.name;
    this.videoMeta = null;
    this.videoLoading = true;
    this.cdr.detectChanges();
  }

  async onVideoLoaded() {
    const video = this.videoRef.nativeElement;

    let duration = video.duration;
    if (!isFinite(duration) || duration <= 0) {
      duration = await this.getAccurateDuration(video);
    }

    const fps = await this.measureFps(video);
    const totalFrames = fps && duration > 0 ? Math.round(fps * duration) : null;

    this.videoMeta = {
      fileName: this.videoFileName ?? 'video',
      width: video.videoWidth,
      height: video.videoHeight,
      duration,
      fps,
      totalFrames,
    };
    this.salto = fps ? Math.max(1, Math.round(fps / 2)) : 15;
    this.computeExtractEstimate();
    this.videoLoading = false;
    this.cdr.detectChanges();
  }

  private detectWebp(): boolean {
    const canvas = document.createElement('canvas');
    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
  }

  get saltoMax(): number {
    const meta = this.videoMeta;
    return meta?.fps ? Math.max(1, Math.ceil(meta.fps)) : 60;
  }

  computeExtractEstimate() {
    const meta = this.videoMeta;
    if (!meta) {
      this.extractEstimate = null;
      return;
    }
    const salto = Math.max(1, Math.round(this.salto));
    let count: number;
    let fpsKept: number | null = null;
    if (meta.fps && meta.totalFrames) {
      count = Math.ceil(meta.totalFrames / salto);
      fpsKept = meta.fps / salto;
    } else if (meta.fps && meta.duration > 0) {
      const tf = Math.round(meta.fps * meta.duration);
      count = Math.ceil(tf / salto);
      fpsKept = meta.fps / salto;
    } else {
      count = Math.ceil(meta.duration * salto);
    }
    this.extractEstimate = { fpsKept, count };
  }

  onSaltoInput(event: Event) {
    const input = event.target as HTMLInputElement;
    const v = parseInt(input.value, 10);
    this.salto = Number.isFinite(v) && v > 0 ? v : 1;
    this.computeExtractEstimate();
  }

  formatFps(v: number | null): string {
    if (v === null) return 'fps no medible';
    return `${Math.round(v * 100) / 100} fps`;
  }

  sessionDate(ts: number): string {
    return new Date(ts).toLocaleString();
  }

  private getAccurateDuration(video: HTMLVideoElement): Promise<number> {
    return new Promise((resolve) => {
      const onDuration = () => resolve(video.duration);
      const onError = () => resolve(video.duration || 0);
      video.currentTime = 1e7;
      video.addEventListener('durationchange', onDuration, { once: true });
      video.addEventListener('error', onError, { once: true });
      window.setTimeout(() => resolve(video.duration || 0), 3000);
    });
  }

  private measureFps(video: HTMLVideoElement): Promise<number | null> {
    return new Promise((resolve) => {
      const v = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: (now: number) => void) => number;
      };

      if (typeof v.requestVideoFrameCallback !== 'function') {
        resolve(null);
        return;
      }

      let firstNow: number | null = null;
      let frames = 0;
      let done = false;

      const finish = () => {
        if (done) return;
        done = true;
        video.pause();
        const elapsed = firstNow !== null ? (performance.now() - firstNow) / 1000 : 0;
        const fps = firstNow !== null && elapsed > 0 ? frames / elapsed : null;
        resolve(fps ? Math.round(fps * 100) / 100 : null);
      };

      const timer = window.setTimeout(finish, 2500);
      video.addEventListener('ended', finish, { once: true });

      const step = (now: number) => {
        if (firstNow === null) firstNow = now;
        frames++;
        if (firstNow !== null && (performance.now() - firstNow) / 1000 >= 1) {
          window.clearTimeout(timer);
          finish();
          return;
        }
        v.requestVideoFrameCallback!(step);
      };

      v.requestVideoFrameCallback(step);
      video.muted = true;
      video.playbackRate = 1;
      video.currentTime = 0;
      video.play().catch(() => {
        window.clearTimeout(timer);
        finish();
      });
    });
  }

  formatDuration(seconds: number): string {
    const s = Math.max(0, seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const secStr = sec.toFixed(2);
    if (h > 0) {
      return `${h}:${String(m).padStart(2, '0')}:${secStr.padStart(5, '0')}`;
    }
    return `${m}:${secStr.padStart(5, '0')}`;
  }

  async extractFrames() {
    const meta = this.videoMeta;
    if (!meta || !this.videoRef || this.extracting) return;
    if (!this.extractEstimate) this.computeExtractEstimate();
    const estimate = this.extractEstimate;
    if (!estimate || estimate.count <= 0) return;

    const video = this.videoRef.nativeElement;
    const salto = Math.max(1, Math.round(this.salto));
    const fps = meta.fps;

    this.extracting = true;
    this.extractAbort = false;
    this.extractProgress = { done: 0, total: estimate.count };
    this.extractStatus = 'Preparando extracción...';
    this.cdr.detectChanges();

    const frames: Omit<VideoFrameRecord, 'id' | 'sessionId'>[] = [];

    try {
      video.pause();
      for (let k = 0; k < estimate.count; k++) {
        if (this.extractAbort) {
          this.extractStatus = 'Extracción cancelada.';
          break;
        }
        const time = Math.min(
          fps && fps > 0 ? (k * salto) / fps : k / salto,
          Math.max(0, meta.duration - 0.001)
        );
        const blob = await this.captureFrame(video, time);
        frames.push({ index: k, timestamp: time, blob });
        this.extractProgress = { done: k + 1, total: estimate.count };
        this.extractStatus = `Extrayendo ${k + 1}/${estimate.count}`;
        this.cdr.detectChanges();
      }
    } catch (err) {
      this.extractStatus = `Error en extracción: ${err}`;
    }

    video.pause();
    video.currentTime = 0;

    if (!this.extractAbort && frames.length > 0) {
      const thumbnail = await this.makeThumbnail(frames[0].blob);
      const session: VideoSession = {
        name: `${meta.fileName} — salto ${salto}`,
        createdAt: Date.now(),
        fileName: meta.fileName,
        width: meta.width,
        height: meta.height,
        duration: meta.duration,
        fps: meta.fps,
        totalFrames: meta.totalFrames,
        salto,
        count: frames.length,
        thumbnail,
      };
      try {
        await this.store.saveSession(session, frames);
        this.extractStatus = `Sesión guardada: ${frames.length} fotogramas.`;
        await this.loadSessions();
      } catch (err) {
        this.extractStatus = `Error al guardar sesión: ${err}`;
      }
    }

    this.extracting = false;
    this.extractProgress = null;
    this.cdr.detectChanges();
  }

  cancelExtract() {
    this.extractAbort = true;
  }

  private captureFrame(video: HTMLVideoElement, time: number): Promise<Blob> {
    return new Promise((resolve, reject) => {
      let timeout = 0;
      const cleanup = () => {
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
        window.clearTimeout(timeout);
      };
      const onSeeked = () => {
        cleanup();
        const canvas = document.createElement('canvas');
        const maxDim = 640;
        const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error('toBlob falló'))),
          this.supportsWebp ? 'image/webp' : 'image/jpeg',
          this.supportsWebp ? 0.82 : 0.85
        );
      };
      const onError = () => {
        cleanup();
        reject(new Error('error al buscar fotograma'));
      };

      if (Math.abs(video.currentTime - time) < 0.001) {
        onSeeked();
        return;
      }
      video.addEventListener('seeked', onSeeked, { once: true });
      video.addEventListener('error', onError, { once: true });
      timeout = window.setTimeout(onSeeked, 2000);
      video.currentTime = time;
    });
  }

  private makeThumbnail(blob: Blob): Promise<Blob | undefined> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const maxDim = 160;
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        canvas.toBlob(
          (blob) => resolve(blob ?? undefined),
          this.supportsWebp ? 'image/webp' : 'image/jpeg',
          0.7
        );
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(undefined);
      };
      img.src = url;
    });
  }

  async loadSessions() {
    for (const s of this.sessions) {
      if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
    }
    const sessions = await this.store.listSessions();
    for (const s of sessions) {
      if (s.thumbnail) s.thumbUrl = URL.createObjectURL(s.thumbnail);
    }
    this.sessions = sessions;
    this.cdr.detectChanges();
  }

  async openSession(session: VideoSession) {
    if (this.selectedSession) this.closeSession();
    this.selectedSession = session;
    this.sessionFrames = [];
    this.cdr.detectChanges();
    const records = await this.store.getFrames(session.id!);
    this.sessionFrames = records.map((r) => ({
      index: r.index,
      timestamp: r.timestamp,
      url: URL.createObjectURL(r.blob),
    }));
    this.cdr.detectChanges();
  }

  closeSession() {
    for (const f of this.sessionFrames) URL.revokeObjectURL(f.url);
    this.sessionFrames = [];
    this.selectedSession = null;
  }

  async deleteSession(id: number) {
    await this.store.deleteSession(id);
    if (this.selectedSession?.id === id) this.closeSession();
    await this.loadSessions();
  }

  async clearSessions() {
    await this.store.clearAll();
    this.closeSession();
    await this.loadSessions();
  }

  ngOnDestroy() {
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
    this.closeSession();
    for (const s of this.sessions) {
      if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
    }
  }

  async runInference() {
    if (!this.model || !this.imageUrl) return;
    if (this.processing) return;

    this.processing = true;
    this.inferenceDone = false;
    this.inferenceTime = null;
    this.status = 'Ejecutando inferencia...';
    this.cdr.detectChanges();
    await new Promise(r => setTimeout(r, 0));

    const start = performance.now();

    try {
      const img = this.imageRef.nativeElement;
      const inputDetails = this.model.getInputDetails();
      const inputShape = Array.from(inputDetails[0].shape);
      const targetH = inputShape[2];
      const targetW = inputShape[3];

      const { pixels, pad } = this.letterbox(img, targetW, targetH);
      const inputTensor = new Tensor(pixels, [1, 3, targetH, targetW]);

      const outputs = await this.model.run(inputTensor);
      inputTensor.delete();

      const outputData = await outputs[0].data();
      outputs[0].delete();

      const outputShape = Array.from(this.model.getOutputDetails()[0].shape);
      this.detections = this.parseYoloOutput(
        new Float32Array(outputData),
        outputShape,
        img.naturalWidth,
        img.naturalHeight,
        targetW,
        targetH,
        pad
      );

      this.inferenceTime = performance.now() - start;
      this.drawDetections();
      this.inferenceDone = true;
      this.status = `Inferencia completada en ${(this.inferenceTime / 1000).toFixed(2)}s — ${this.detections.length} objeto(s) detectado(s).`;
    } catch (err) {
      this.status = `Error en inferencia: ${err}`;
    }
    this.processing = false;
    this.cdr.detectChanges();
  }

  private letterbox(
    img: HTMLImageElement,
    targetW: number,
    targetH: number
  ): { pixels: Float32Array; pad: PadInfo } {
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetW, targetH);

    const scale = Math.min(targetW / img.naturalWidth, targetH / img.naturalHeight);
    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);
    const x = Math.round((targetW - w) / 2);
    const y = Math.round((targetH - h) / 2);

    ctx.drawImage(img, x, y, w, h);

    const pad: PadInfo = {
      left: x,
      top: y,
      right: targetW - w - x,
      bottom: targetH - h - y,
    };

    const imageData = ctx.getImageData(0, 0, targetW, targetH);
    const rgba = imageData.data;
    const pixels = new Float32Array(3 * targetH * targetW);

    for (let i = 0; i < targetH * targetW; i++) {
      pixels[i] = rgba[i * 4] / 255.0;
      pixels[targetH * targetW + i] = rgba[i * 4 + 1] / 255.0;
      pixels[2 * targetH * targetW + i] = rgba[i * 4 + 2] / 255.0;
    }

    return { pixels, pad };
  }

  private parseYoloOutput(
    data: Float32Array,
    shape: number[],
    imgW: number,
    imgH: number,
    modelW: number,
    modelH: number,
    pad: PadInfo
  ): Detection[] {
    const detections: Detection[] = [];
    const confidenceThreshold = 0.25;

    if (shape.length === 3) {
      const dim1 = shape[1];
      const dim2 = shape[2];

      if (dim2 === 6) {
        for (let i = 0; i < dim1; i++) {
          const offset = i * 6;
          const conf = data[offset + 4];
          if (conf < confidenceThreshold) continue;

          const x1 = data[offset];
          const y1 = data[offset + 1];
          const x2 = data[offset + 2];
          const y2 = data[offset + 3];
          const classId = Math.round(data[offset + 5]);

          detections.push(this.scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
        }
      } else if (dim1 === 6) {
        for (let i = 0; i < dim2; i++) {
          const offset = i * 6;
          const conf = data[offset + 4];
          if (conf < confidenceThreshold) continue;

          const x1 = data[offset];
          const y1 = data[offset + 1];
          const x2 = data[offset + 2];
          const y2 = data[offset + 3];
          const classId = Math.round(data[offset + 5]);

          detections.push(this.scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
        }
      } else {
        const numClasses = dim2 - 4;
        for (let i = 0; i < dim1; i++) {
          const offset = i * dim2;
          const cx = data[offset];
          const cy = data[offset + 1];
          const w = data[offset + 2];
          const h = data[offset + 3];

          let bestClass = 0;
          let bestScore = 0;
          for (let c = 0; c < numClasses; c++) {
            const score = data[offset + 4 + c];
            if (score > bestScore) {
              bestScore = score;
              bestClass = c;
            }
          }

          if (bestScore < confidenceThreshold) continue;

          const x1 = cx - w / 2;
          const y1 = cy - h / 2;
          const x2 = cx + w / 2;
          const y2 = cy + h / 2;

          detections.push(this.scaleBox(x1, y1, x2, y2, bestScore, bestClass, imgW, imgH, modelW, modelH, pad));
        }
      }
    } else if (shape.length === 2) {
      const numDetections = shape[0];
      const cols = shape[1];
      for (let i = 0; i < numDetections; i++) {
        const offset = i * cols;
        if (cols >= 6) {
          const conf = data[offset + 4];
          if (conf < confidenceThreshold) continue;
          const x1 = data[offset];
          const y1 = data[offset + 1];
          const x2 = data[offset + 2];
          const y2 = data[offset + 3];
          const classId = Math.round(data[offset + 5]);
          detections.push(this.scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
        }
      }
    }

    return this.nms(detections, 0.45);
  }

  private scaleBox(
    x1: number, y1: number, x2: number, y2: number,
    confidence: number, classId: number,
    imgW: number, imgH: number,
    modelW: number, modelH: number,
    pad: PadInfo
  ): Detection {
    const scaleX = imgW / (modelW - pad.left - pad.right);
    const scaleY = imgH / (modelH - pad.top - pad.bottom);

    const bx1 = Math.max(0, (x1 - pad.left) * scaleX);
    const by1 = Math.max(0, (y1 - pad.top) * scaleY);
    const bx2 = Math.min(imgW, (x2 - pad.left) * scaleX);
    const by2 = Math.min(imgH, (y2 - pad.top) * scaleY);

    return {
      x: bx1,
      y: by1,
      width: bx2 - bx1,
      height: by2 - by1,
      confidence,
      classId,
    };
  }

  private nms(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) return [];

    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    const keep: Detection[] = [];

    while (sorted.length > 0) {
      const current = sorted.shift()!;
      keep.push(current);

      const remaining: Detection[] = [];
      for (const d of sorted) {
        const iou = this.computeIoU(current, d);
        if (iou <= iouThreshold) {
          remaining.push(d);
        }
      }
      sorted.length = 0;
      sorted.push(...remaining);
    }

    return keep;
  }

  cocoClass(classId: number): string {
    return COCO_CLASSES[classId] ?? `class_${classId}`;
  }

  private computeIoU(a: Detection, b: Detection): number {
    const ax1 = a.x, ay1 = a.y, ax2 = a.x + a.width, ay2 = a.y + a.height;
    const bx1 = b.x, by1 = b.y, bx2 = b.x + b.width, by2 = b.y + b.height;

    const interX1 = Math.max(ax1, bx1);
    const interY1 = Math.max(ay1, by1);
    const interX2 = Math.min(ax2, bx2);
    const interY2 = Math.min(ay2, by2);

    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const interArea = interW * interH;

    const areaA = a.width * a.height;
    const areaB = b.width * b.height;

    return interArea / (areaA + areaB - interArea);
  }

  private drawDetections() {
    const img = this.imageRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const colors = [
      '#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
      '#FF8800', '#88FF00', '#0088FF', '#FF0088', '#8800FF', '#00FF88',
    ];

    for (const d of this.detections) {
      const color = colors[d.classId % colors.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(d.x, d.y, d.width, d.height);

      const label = `${COCO_CLASSES[d.classId] ?? d.classId} ${(d.confidence * 100).toFixed(1)}%`;
      ctx.font = '16px Arial';
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(d.x, d.y - 20, textW + 8, 20);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, d.x + 4, d.y - 5);
    }
  }
}
