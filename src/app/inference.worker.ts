import { loadLiteRt, loadAndCompile, Tensor, CompiledModel } from '@litertjs/core';
import { Detection, PadInfo } from './detection';

interface InitMessage {
  type: 'init';
  model: string;
}

interface InferMessage {
  type: 'infer';
  id: number;
  bitmap: ImageBitmap;
}

type WorkerContext = {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent) => void) | null;
};

const ctx = self as unknown as WorkerContext;

let model: CompiledModel | null = null;

const workerGlobal = self as typeof self & {
  Module?: { locateFile?: (path: string) => string };
};

// Emscripten no tiene document.currentScript dentro de un worker. Fijamos
// explícitamente la ubicación de los binarios WASM servidos por Angular.
workerGlobal.Module = {
  locateFile: (path: string) => `/wasm/${path.split('/').pop() ?? path}`,
};

ctx.onmessage = async (event: MessageEvent) => {
  const msg = event.data as InitMessage | InferMessage;
  if (!msg) return;

  if (msg.type === 'init') {
    try {
      await loadLiteRt('/wasm/');
      if (model) {
        model.delete();
        model = null;
      }
      model = await loadAndCompile(`/models/${msg.model}`, {
        accelerator: 'wasm',
      });
      const shape = Array.from(model.getInputDetails()[0].shape);
      ctx.postMessage({ type: 'ready', shape: shape.slice(1).join('x') });
    } catch (err) {
      ctx.postMessage({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  } else if (msg.type === 'infer') {
    try {
      if (!model) throw new Error('Modelo no cargado');
      const detections = await inferBitmap(model, msg.bitmap);
      ctx.postMessage({ type: 'result', id: msg.id, detections });
    } catch (err) {
      ctx.postMessage({
        type: 'result',
        id: msg.id,
        detections: [],
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      msg.bitmap.close();
    }
  }
};

async function inferBitmap(model: CompiledModel, bitmap: ImageBitmap): Promise<Detection[]> {
  const imgW = bitmap.width;
  const imgH = bitmap.height;
  const inputShape = Array.from(model.getInputDetails()[0].shape);
  const targetH = inputShape[2];
  const targetW = inputShape[3];

  const { pixels, pad } = letterbox(bitmap, imgW, imgH, targetW, targetH);
  const inputTensor = new Tensor(pixels, [1, 3, targetH, targetW]);

  try {
    const outputs = await model.run(inputTensor);
    const outputData = await outputs[0].data();
    outputs[0].delete();

    const outputShape = Array.from(model.getOutputDetails()[0].shape);
    return parseYoloOutput(
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

function letterbox(
  img: ImageBitmap,
  imgW: number,
  imgH: number,
  targetW: number,
  targetH: number
): { pixels: Float32Array; pad: PadInfo } {
  const canvas = new OffscreenCanvas(targetW, targetH);
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

function parseYoloOutput(
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

        detections.push(scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
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

        detections.push(scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
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

        detections.push(scaleBox(x1, y1, x2, y2, bestScore, bestClass, imgW, imgH, modelW, modelH, pad));
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
        detections.push(scaleBox(x1, y1, x2, y2, conf, classId, imgW, imgH, modelW, modelH, pad));
      }
    }
  }

  return nms(detections, 0.45);
}

function scaleBox(
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

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  if (detections.length === 0) return [];

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const keep: Detection[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;
    keep.push(current);

    const remaining: Detection[] = [];
    for (const d of sorted) {
      const iou = computeIoU(current, d);
      if (iou <= iouThreshold) {
        remaining.push(d);
      }
    }
    sorted.length = 0;
    sorted.push(...remaining);
  }

  return keep;
}

function computeIoU(a: Detection, b: Detection): number {
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
