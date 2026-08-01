import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { loadLiteRt, loadAndCompile, Tensor, CompiledModel } from '@litertjs/core';

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

  constructor(private cdr: ChangeDetectorRef) {}

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
    this.videoLoading = false;
    this.cdr.detectChanges();
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

  ngOnDestroy() {
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
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
