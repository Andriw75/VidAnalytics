import { loadLiteRt, loadAndCompile, Tensor, CompiledModel } from '@litertjs/core';
import { Detection, Keypoint, ModelMetadata, PadInfo } from './detection';

interface InitMessage {
  type: 'init';
  model: string;
  metadata: ModelMetadata;
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
let runtimeReady = false;
let modelMetadata: ModelMetadata | null = null;

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
      if (!runtimeReady) {
        await loadLiteRt('/wasm/');
        runtimeReady = true;
      }
      if (model) {
        model.delete();
        model = null;
      }
      modelMetadata = msg.metadata;
      model = await loadAndCompile(`/models/${msg.model}`, {
        accelerator: 'wasm',
      });
      const shape = Array.from(model.getInputDetails()[0].shape);
      const input = model.getInputDetails()[0];
      const outputs = model.getOutputDetails();
      ctx.postMessage({
        type: 'ready',
        metadata: {
          ...msg.metadata,
          input: { shape: shape.map(Number), dtype: String(input.dtype) },
          outputs: outputs.map((output) => ({ shape: Array.from(output.shape).map(Number), dtype: String(output.dtype) })),
          source: 'manifest+runtime',
        },
        shape: shape.slice(1).join('x'),
      });
    } catch (err) {
      ctx.postMessage({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  } else if (msg.type === 'infer') {
    try {
      if (!model) throw new Error('Modelo no cargado');
      const detections = await inferBitmap(model, msg.bitmap, modelMetadata);
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

async function inferBitmap(model: CompiledModel, bitmap: ImageBitmap, metadata: ModelMetadata | null): Promise<Detection[]> {
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
    return parseModelOutput(
      new Float32Array(outputData),
      outputShape,
      imgW,
      imgH,
      targetW,
      targetH,
      pad,
      metadata
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

function parseModelOutput(
  data: Float32Array,
  shape: number[],
  imgW: number,
  imgH: number,
  modelW: number,
  modelH: number,
  pad: PadInfo,
  metadata: ModelMetadata | null
): Detection[] {
  if (metadata?.task === 'pose') {
    return parsePoseOutput(data, shape, imgW, imgH, modelW, modelH, pad, metadata);
  }
  return parseYoloOutput(data, shape, imgW, imgH, modelW, modelH, pad);
}

function parsePoseOutput(
  data: Float32Array,
  shape: number[],
  imgW: number,
  imgH: number,
  modelW: number,
  modelH: number,
  pad: PadInfo,
  metadata: ModelMetadata
): Detection[] {
  const detections: Detection[] = [];
  const confidenceThreshold = 0.25;
  const keypointCount = metadata.kptShape?.[0] ?? 17;
  const keypointDimensions = metadata.kptShape?.[1] ?? 3;
  const classCount = Math.max(1, metadata.classCount || 1);
  const expectedRawChannels = 4 + classCount + keypointCount * keypointDimensions;
  const expectedEndToEndColumns = 6 + keypointCount * keypointDimensions;

  const addRaw = (candidate: number[], keypointValues: number[]) => {
    const cx = candidate[0];
    const cy = candidate[1];
    const w = candidate[2];
    const h = candidate[3];
    let classId = 0;
    let score = 0;
    for (let c = 0; c < classCount; c++) {
      if (candidate[4 + c] > score) {
        score = candidate[4 + c];
        classId = c;
      }
    }
    if (score < confidenceThreshold) return;
    detections.push({
      ...scaleBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, classId, imgW, imgH, modelW, modelH, pad),
      keypoints: decodeKeypoints(keypointValues, keypointDimensions, keypointCount, imgW, imgH, modelW, modelH, pad, metadata),
    });
  };

  const addEndToEnd = (row: number[]) => {
    const confidence = row[4];
    if (confidence < confidenceThreshold) return;
    const classId = Math.round(row[5]);
    detections.push({
      ...scaleBox(row[0], row[1], row[2], row[3], confidence, classId, imgW, imgH, modelW, modelH, pad),
      keypoints: decodeKeypoints(row.slice(6), keypointDimensions, keypointCount, imgW, imgH, modelW, modelH, pad, metadata),
    });
  };

  if (shape.length === 3) {
    const dim1 = shape[1];
    const dim2 = shape[2];
    if (dim2 === expectedEndToEndColumns || dim2 === 6) {
      for (let i = 0; i < dim1; i++) addEndToEnd(Array.from(data.slice(i * dim2, (i + 1) * dim2)));
    } else if (dim1 === expectedRawChannels || dim1 === 4 + classCount + keypointCount * 2) {
      for (let i = 0; i < dim2; i++) {
        const candidate = [];
        for (let c = 0; c < 4 + classCount; c++) candidate.push(data[c * dim2 + i]);
        const keypoints = [];
        const base = (4 + classCount) * dim2 + i;
        for (let k = 0; k < keypointCount * keypointDimensions; k++) {
          keypoints.push(data[(4 + classCount + k) * dim2 + i]);
        }
        addRaw(candidate, keypoints.length ? keypoints : Array.from(data.slice(base, base + keypointCount * keypointDimensions)));
      }
    }
  } else if (shape.length === 2) {
    const rows = shape[0];
    const cols = shape[1];
    for (let i = 0; i < rows; i++) {
      const row = Array.from(data.slice(i * cols, (i + 1) * cols));
      if (cols === expectedEndToEndColumns || cols === 6) addEndToEnd(row);
      else if (cols === expectedRawChannels) addRaw(row.slice(0, 4 + classCount), row.slice(4 + classCount));
    }
  }

  return nms(detections, 0.45);
}

function decodeKeypoints(
  values: number[],
  dimensions: number,
  count: number,
  imgW: number,
  imgH: number,
  modelW: number,
  modelH: number,
  pad: PadInfo,
  metadata: ModelMetadata
): Keypoint[] {
  const keypoints: Keypoint[] = [];
  for (let i = 0; i < count; i++) {
    const offset = i * dimensions;
    const point = scalePoint(values[offset], values[offset + 1], imgW, imgH, modelW, modelH, pad);
    keypoints.push({
      x: point.x,
      y: point.y,
      confidence: dimensions > 2 ? values[offset + 2] : undefined,
      name: metadata.kptNames?.[i],
    });
  }
  return keypoints;
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

function scalePoint(
  x: number,
  y: number,
  imgW: number,
  imgH: number,
  modelW: number,
  modelH: number,
  pad: PadInfo
): { x: number; y: number } {
  const scaleX = imgW / (modelW - pad.left - pad.right);
  const scaleY = imgH / (modelH - pad.top - pad.bottom);
  return {
    x: Math.max(0, Math.min(imgW, (x - pad.left) * scaleX)),
    y: Math.max(0, Math.min(imgH, (y - pad.top) * scaleY)),
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
