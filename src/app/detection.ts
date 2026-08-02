export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  classId: number;
  keypoints?: Keypoint[];
}

export interface Keypoint {
  x: number;
  y: number;
  confidence?: number;
  name?: string;
}

export type ModelTask = 'detect' | 'pose' | 'segment' | 'classify' | 'obb' | 'unknown';

export interface TensorMetadata {
  name?: string;
  shape: number[];
  dtype?: string;
}

export interface ModelMetadata {
  file: string;
  label: string;
  task: ModelTask;
  head: string | null;
  stride: number | null;
  imageSize: [number, number];
  channels: number;
  batch: number;
  classCount: number;
  classNames?: string[];
  kptShape?: [number, number];
  kptNames?: string[];
  input?: TensorMetadata;
  outputs?: TensorMetadata[];
  source: 'manifest' | 'manifest+runtime' | 'inferred';
}

export interface PadInfo {
  top: number;
  left: number;
  bottom: number;
  right: number;
}

export interface VideoMeta {
  fileName: string;
  width: number;
  height: number;
  duration: number;
  fps: number | null;
  totalFrames: number | null;
}
