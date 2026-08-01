import { Injectable } from '@angular/core';
import { loadLiteRt, loadAndCompile, Tensor, CompiledModel } from '@litertjs/core';
import { Detection, PadInfo } from './detection';

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

function sourceWidth(src: ImageSource): number {
  return 'naturalWidth' in src ? src.naturalWidth : src.width;
}

function sourceHeight(src: ImageSource): number {
  return 'naturalHeight' in src ? src.naturalHeight : src.height;
}

@Injectable({ providedIn: 'root' })
export class InferenceService {
  private model: CompiledModel | null = null;
  private modelName: string | null = null;

  get isReady(): boolean {
    return this.model !== null;
  }

  async init(): Promise<void> {
    await loadLiteRt('/wasm/');
  }

  async loadModel(name: string): Promise<string> {
    if (this.model) {
      this.model.delete();
      this.model = null;
    }
    this.modelName = name;
    try {
      this.model = await loadAndCompile(`/models/${name}`, {
        accelerator: 'wasm',
      });
    } catch {
      this.model = await loadAndCompile(`/models/${name}`, {
        accelerator: 'webgpu',
      });
    }
    const details = this.model.getInputDetails();
    const shape = Array.from(details[0].shape);
    return shape.slice(1).join('x');
  }
  cocoClass(classId: number): string {
    return COCO_CLASSES[classId] ?? `class_${classId}`;
  }

  async inferImage(source: ImageSource): Promise<Detection[]> {
    if (!this.model) throw new Error('Modelo no cargado');
    const imgW = sourceWidth(source);
    const imgH = sourceHeight(source);
    const inputDetails = this.model.getInputDetails();
    const inputShape = Array.from(inputDetails[0].shape);
    const targetH = inputShape[2];
    const targetW = inputShape[3];

    const { pixels, pad } = this.letterbox(source, imgW, imgH, targetW, targetH);
    const inputTensor = new Tensor(pixels, [1, 3, targetH, targetW]);

    try {
      const outputs = await this.model.run(inputTensor);
      const outputData = await outputs[0].data();
      outputs[0].delete();

      const outputShape = Array.from(this.model.getOutputDetails()[0].shape);
      return this.parseYoloOutput(
        new Float32Array(outputData),
        outputShape,
        imgW,
        imgH,
        targetW,
        targetH,
        pad
      );
    } finally {
      inputTensor.delete();
    }
  }

  async inferBlob(blob: Blob): Promise<Detection[]> {
    const bmp = await createImageBitmap(blob);
    try {
      return await this.inferImage(bmp);
    } finally {
      bmp.close();
    }
  }

  drawFrame(
    canvas: HTMLCanvasElement,
    source: ImageSource,
    detections: Detection[] | undefined
  ): void {
    const imgW = sourceWidth(source);
    const imgH = sourceHeight(source);
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
    }
  }

  private letterbox(
    img: ImageSource,
    imgW: number,
    imgH: number,
    targetW: number,
    targetH: number
  ): { pixels: Float32Array; pad: PadInfo } {
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, targetW, targetH);

    const scale = Math.min(targetW / imgW, targetH / imgH);
    const w = Math.round(imgW * scale);
    const h = Math.round(imgH * scale);
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
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    confidence: number,
    classId: number,
    imgW: number,
    imgH: number,
    modelW: number,
    modelH: number,
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
}
