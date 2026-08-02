import {
  AnalysisFrame,
  BboxFilter,
  FrameSuggestion,
  ImageSize,
  KeypointFilter,
  SearchFilters,
  SearchResult,
  SuggestionEvent,
  SuggestionReason,
  Zone,
  ZoneMatchMode,
} from './filter-types';
import { Detection } from '../detection';
import { pointInRect, rectIntersects, rectOverlapArea } from './geometry';
import { visibleKeypoints, namedKeypoint, posePresetMatch } from './pose-metrics';

const MAX_EVENT_GAP = 2;

interface NormalizedDetection {
  detection: Detection;
  x: number;
  y: number;
  width: number;
  height: number;
  cx: number;
  cy: number;
}

function normalizeDetection(detection: Detection, size: ImageSize): NormalizedDetection {
  const width = size.width > 0 ? detection.width / size.width : 0;
  const height = size.height > 0 ? detection.height / size.height : 0;
  const x = size.width > 0 ? detection.x / size.width : 0;
  const y = size.height > 0 ? detection.y / size.height : 0;
  return {
    detection,
    x,
    y,
    width,
    height,
    cx: x + width / 2,
    cy: y + height / 2,
  };
}

function matchesClass(classId: number, filter: BboxFilter): boolean {
  return filter.classId === 'all' || filter.classId === classId;
}

function bboxInZone(normalized: NormalizedDetection, mode: ZoneMatchMode, zones: Zone[], minZoneArea: number): { matched: boolean; label: string | null } {
  if (mode === 'none') return { matched: true, label: null };
  const enabled = zones.filter((zone) => zone.enabled);
  if (enabled.length === 0) return { matched: false, label: null };

  for (const zone of enabled) {
    if (mode === 'center') {
      if (pointInRect(normalized.cx, normalized.cy, zone.rect)) {
        return { matched: true, label: `Centro de detección dentro de ${zone.name}` };
      }
    } else if (mode === 'intersect') {
      if (
        rectIntersects(
          { x: normalized.x, y: normalized.y, width: normalized.width, height: normalized.height },
          zone.rect
        )
      ) {
        return { matched: true, label: `Bounding box intersecta ${zone.name}` };
      }
    } else if (mode === 'area') {
      const area = rectOverlapArea(
        { x: normalized.x, y: normalized.y, width: normalized.width, height: normalized.height },
        zone.rect
      );
      if (area >= minZoneArea) {
        return { matched: true, label: `${Math.round(area * 100)}% del box dentro de ${zone.name}` };
      }
    }
  }

  return { matched: false, label: null };
}

function bboxMatches(normalized: NormalizedDetection, filter: BboxFilter, zones: Zone[]): { matched: boolean; reason: SuggestionReason | null } {
  if (normalized.detection.confidence < filter.minConfidence) return { matched: false, reason: null };
  if (!matchesClass(normalized.detection.classId, filter)) return { matched: false, reason: null };

  const zone = bboxInZone(normalized, filter.zoneMode, zones, filter.minZoneArea);
  if (!zone.matched) return { matched: false, reason: null };

  const label = zone.label ?? `Detección ${normalized.detection.confidence >= filter.minConfidence ? 'confiable' : ''}`;
  return {
    matched: true,
    reason: {
      type: filter.zoneMode !== 'none' ? 'bbox-zone' : 'class-count',
      label: label.trim() || `Clase ${normalized.detection.classId} con confianza ${Math.round(normalized.detection.confidence * 100)}%`,
    },
  };
}

function keypointInZone(keypoint: { x: number; y: number } | undefined, zones: Zone[], size: ImageSize): boolean {
  if (!keypoint) return false;
  const enabled = zones.filter((zone) => zone.enabled);
  if (enabled.length === 0) return false;
  const nx = size.width > 0 ? keypoint.x / size.width : 0;
  const ny = size.height > 0 ? keypoint.y / size.height : 0;
  return enabled.some((zone) => pointInRect(nx, ny, zone.rect));
}

function keypointMatches(
  detection: Detection,
  filter: KeypointFilter,
  zones: Zone[],
  size: ImageSize
): { matched: boolean; reason: SuggestionReason | null } {
  const keypoints = detection.keypoints;
  if (!keypoints || keypoints.length === 0) return { matched: false, reason: null };

  const visible = visibleKeypoints(keypoints, filter.minConfidence);
  if (visible < filter.minVisible) return { matched: false, reason: null };

  if (!posePresetMatch(keypoints, filter.preset, filter.minConfidence)) return { matched: false, reason: null };

  if (filter.zoneMode !== 'none') {
    if (filter.zoneMode === 'any') {
      const inZone = keypoints.some((point) => keypointInZone(point, zones, size));
      if (!inZone) return { matched: false, reason: null };
    } else if (filter.zoneMode === 'named' && filter.zoneKeypoint) {
      const point = namedKeypoint(keypoints, filter.zoneKeypoint);
      if (!keypointInZone(point, zones, size)) return { matched: false, reason: null };
    } else {
      return { matched: false, reason: null };
    }
  }

  const labels: string[] = [];
  labels.push(`${visible} keypoints visibles`);
  if (filter.zoneMode !== 'none') labels.push('keypoint en zona restringida');
  if (filter.preset !== 'none') labels.push(`postura: ${filter.preset}`);

  return {
    matched: true,
    reason: {
      type: filter.zoneMode !== 'none' ? 'keypoint-zone' : 'pose',
      label: labels.join(' · '),
    },
  };
}

export function searchFrames(frames: AnalysisFrame[], filters: SearchFilters, size: ImageSize): SearchResult {
  const suggestions: FrameSuggestion[] = [];
  const zones = filters.zones;

  for (const frame of frames) {
    const detections = frame.detections ?? [];
    if (detections.length === 0) continue;

    const reasons: SuggestionReason[] = [];
    const detectionIndexes: number[] = [];
    let score = 0;
    let matched = false;

    if (filters.bbox.enabled) {
      let count = 0;
      for (let i = 0; i < detections.length; i++) {
        const normalized = normalizeDetection(detections[i], size);
        const result = bboxMatches(normalized, filters.bbox, zones);
        if (result.matched) {
          count++;
          detectionIndexes.push(i);
          if (result.reason && !reasons.some((reason) => reason.label === result.reason!.label)) {
            reasons.push(result.reason);
          }
          score = Math.max(score, detections[i].confidence);
        }
      }
      if (filters.bbox.classId === 'all') {
        if (count >= filters.bbox.minCount && (filters.bbox.maxCount === null || count <= filters.bbox.maxCount)) {
          matched = true;
          reasons.unshift({ type: 'count', label: `${count} detección(es) en el frame` });
        }
      } else if (count >= filters.bbox.minCount) {
        matched = true;
        reasons.unshift({ type: 'count', label: `${count} detección(es) de clase ${filters.bbox.classId}` });
      }
    }

    if (filters.keypoints.enabled) {
      for (let i = 0; i < detections.length; i++) {
        const result = keypointMatches(detections[i], filters.keypoints, zones, size);
        if (result.matched) {
          matched = true;
          detectionIndexes.push(i);
          if (result.reason) reasons.push(result.reason);
        }
      }
    }

    if (matched) {
      suggestions.push({
        frameIndex: frame.index,
        timestamp: frame.timestamp,
        score,
        reasons,
        detectionIndexes: Array.from(new Set(detectionIndexes)),
      });
    }
  }

  suggestions.sort((a, b) => a.frameIndex - b.frameIndex);

  const events: SuggestionEvent[] = [];
  let current: FrameSuggestion[] = [];
  for (const suggestion of suggestions) {
    if (current.length === 0) {
      current = [suggestion];
    } else {
      const previous = current[current.length - 1];
      if (suggestion.frameIndex - previous.frameIndex <= MAX_EVENT_GAP) {
        current.push(suggestion);
      } else {
        events.push(buildEvent(current));
        current = [suggestion];
      }
    }
  }
  if (current.length > 0) events.push(buildEvent(current));

  return { suggestions, events };
}

function buildEvent(frames: FrameSuggestion[]): SuggestionEvent {
  const primaryReason = frames[0]?.reasons.find((reason) => reason.type !== 'count')?.label ?? frames[0]?.reasons[0]?.label ?? 'Coincidencia';
  const color = reasonColor(primaryReason, frames[0]);
  return {
    startFrame: frames[0].frameIndex,
    endFrame: frames[frames.length - 1].frameIndex,
    startTime: frames[0].timestamp,
    endTime: frames[frames.length - 1].timestamp,
    frames,
    primaryReason,
    color,
  };
}

function reasonColor(primaryReason: string, suggestion: FrameSuggestion | undefined): string {
  const type = suggestion?.reasons[0]?.type;
  if (type === 'bbox-zone' || type === 'keypoint-zone') return '#ff4444';
  if (type === 'pose') return '#3b82f6';
  return '#f59e0b';
}
