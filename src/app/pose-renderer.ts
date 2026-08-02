import { Keypoint } from './detection';

type PoseGroup = 'head' | 'torso' | 'leftArm' | 'rightArm' | 'leftLeg' | 'rightLeg';

interface PoseEdge {
  a: number;
  b: number;
  group: PoseGroup;
}

const COLORS: Record<PoseGroup, string> = {
  head: '#00F0FF',
  torso: '#FF3B30',
  leftArm: '#E4FF00',
  rightArm: '#0088FF',
  leftLeg: '#00FF13',
  rightLeg: '#FF00FF',
};

// COCO-17 keypoint order, using zero-based indexes.
const EDGES: PoseEdge[] = [
  { a: 0, b: 1, group: 'head' },
  { a: 0, b: 2, group: 'head' },
  { a: 1, b: 3, group: 'head' },
  { a: 2, b: 4, group: 'head' },
  { a: 5, b: 6, group: 'torso' },
  { a: 11, b: 12, group: 'torso' },
  { a: 5, b: 11, group: 'torso' },
  { a: 6, b: 12, group: 'torso' },
  { a: 5, b: 7, group: 'leftArm' },
  { a: 7, b: 9, group: 'leftArm' },
  { a: 6, b: 8, group: 'rightArm' },
  { a: 8, b: 10, group: 'rightArm' },
  { a: 11, b: 13, group: 'leftLeg' },
  { a: 13, b: 15, group: 'leftLeg' },
  { a: 12, b: 14, group: 'rightLeg' },
  { a: 14, b: 16, group: 'rightLeg' },
];

export function drawPose(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  imageWidth: number,
  threshold = 0.25
): void {
  const radius = Math.max(3, Math.round(imageWidth / 180));
  const lineWidth = Math.max(2, Math.round(imageWidth / 400));
  const valid = (index: number) => {
    const point = keypoints[index];
    return point && (point.confidence === undefined || point.confidence >= threshold) && point.x > 0 && point.y > 0;
  };

  ctx.lineWidth = lineWidth;
  for (const edge of EDGES) {
    if (!valid(edge.a) || !valid(edge.b)) continue;
    const a = keypoints[edge.a];
    const b = keypoints[edge.b];
    ctx.strokeStyle = COLORS[edge.group];
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (let index = 0; index < keypoints.length; index++) {
    if (!valid(index)) continue;
    const point = keypoints[index];
    const group = index <= 4 ? 'head' : index <= 10 ? (index % 2 === 1 ? 'leftArm' : 'rightArm') : index % 2 === 1 ? 'leftLeg' : 'rightLeg';
    ctx.fillStyle = COLORS[group];
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function poseLegend(): Array<{ label: string; color: string }> {
  return [
    { label: 'Cabeza', color: COLORS.head },
    { label: 'Torso', color: COLORS.torso },
    { label: 'Brazo izquierdo', color: COLORS.leftArm },
    { label: 'Brazo derecho', color: COLORS.rightArm },
    { label: 'Pierna izquierda', color: COLORS.leftLeg },
    { label: 'Pierna derecha', color: COLORS.rightLeg },
  ];
}
