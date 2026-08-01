export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  classId: number;
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
