import { Keypoint } from '../detection';
import { PosePreset } from './filter-types';

export function visibleKeypoints(keypoints: Keypoint[] | undefined, minConfidence: number): number {
  if (!keypoints) return 0;
  let count = 0;
  for (const point of keypoints) {
    if (point.x <= 0 && point.y <= 0) continue;
    if (point.confidence !== undefined && point.confidence < minConfidence) continue;
    count++;
  }
  return count;
}

export function namedKeypoint(keypoints: Keypoint[] | undefined, name: string): Keypoint | undefined {
  if (!keypoints) return undefined;
  return keypoints.find((point) => point.name === name);
}

export function keypointPoints(keypoints: Keypoint[] | undefined): { x: number; y: number }[] {
  if (!keypoints) return [];
  return keypoints.map((point) => ({ x: point.x, y: point.y }));
}

function armPoint(keypoints: Keypoint[] | undefined, name: string): Keypoint | undefined {
  return namedKeypoint(keypoints, name);
}

export function posePresetMatch(keypoints: Keypoint[] | undefined, preset: PosePreset, minConfidence: number): boolean {
  if (!keypoints || preset === 'none') return true;

  const leftWrist = armPoint(keypoints, 'left_wrist');
  const rightWrist = armPoint(keypoints, 'right_wrist');
  const leftShoulder = armPoint(keypoints, 'left_shoulder');
  const rightShoulder = armPoint(keypoints, 'right_shoulder');

  const valid = (point: Keypoint | undefined) =>
    !!point && point.x > 0 && point.y > 0 && (point.confidence === undefined || point.confidence >= minConfidence);

  if (preset === 'armsUp') {
    if (!valid(leftWrist) || !valid(rightWrist) || !valid(leftShoulder) || !valid(rightShoulder)) return false;
    return leftWrist!.y < leftShoulder!.y && rightWrist!.y < rightShoulder!.y;
  }

  if (preset === 'armsDown') {
    if (!valid(leftWrist) || !valid(rightWrist) || !valid(leftShoulder) || !valid(rightShoulder)) return false;
    return leftWrist!.y > leftShoulder!.y + 40 && rightWrist!.y > rightShoulder!.y + 40;
  }

  return true;
}
