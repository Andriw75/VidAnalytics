import { Detection, Keypoint } from '../detection';

export interface NormalizedRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Zone {
  id: string;
  name: string;
  rect: NormalizedRect;
  enabled: boolean;
  color: string;
}

export type ClassSelector = 'all' | number;

export type ZoneMatchMode = 'none' | 'center' | 'intersect' | 'area';

export interface BboxFilter {
  enabled: boolean;
  classId: ClassSelector;
  minCount: number;
  maxCount: number | null;
  minConfidence: number;
  zoneMode: ZoneMatchMode;
  minZoneArea: number;
}

export type PosePreset = 'none' | 'armsUp' | 'armsDown';

export type KeypointZoneMode = 'none' | 'any' | 'named';

export interface KeypointFilter {
  enabled: boolean;
  minVisible: number;
  minConfidence: number;
  zoneMode: KeypointZoneMode;
  zoneKeypoint: string | null;
  preset: PosePreset;
}

export interface SearchFilters {
  bbox: BboxFilter;
  keypoints: KeypointFilter;
  zones: Zone[];
}

export interface SuggestionReason {
  type: string;
  label: string;
}

export interface FrameSuggestion {
  frameIndex: number;
  timestamp: number;
  score: number;
  reasons: SuggestionReason[];
  detectionIndexes: number[];
}

export interface SuggestionEvent {
  startFrame: number;
  endFrame: number;
  startTime: number;
  endTime: number;
  frames: FrameSuggestion[];
  primaryReason: string;
  color: string;
}

export interface SearchResult {
  suggestions: FrameSuggestion[];
  events: SuggestionEvent[];
}

export interface AnalysisFrame {
  index: number;
  timestamp: number;
  detections?: Detection[];
}

export interface ImageSize {
  width: number;
  height: number;
}

export function defaultFilters(zones: Zone[] = []): SearchFilters {
  return {
    bbox: {
      enabled: true,
      classId: 'all',
      minCount: 1,
      maxCount: null,
      minConfidence: 0.25,
      zoneMode: 'none',
      minZoneArea: 0.5,
    },
    keypoints: {
      enabled: false,
      minVisible: 6,
      minConfidence: 0.25,
      zoneMode: 'none',
      zoneKeypoint: null,
      preset: 'none',
    },
    zones,
  };
}
