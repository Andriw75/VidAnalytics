import { NormalizedRect } from './filter-types';

export function pointInRect(nx: number, ny: number, rect: NormalizedRect): boolean {
  return (
    nx >= rect.x &&
    nx <= rect.x + rect.width &&
    ny >= rect.y &&
    ny <= rect.y + rect.height
  );
}

export function rectIntersects(a: NormalizedRect, b: NormalizedRect): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

export function rectOverlapArea(a: NormalizedRect, b: NormalizedRect): number {
  const iw = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const ih = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const inter = iw * ih;
  const areaA = a.width * a.height;
  if (areaA <= 0) return 0;
  return Math.min(1, inter / areaA);
}

export function rectFromPoints(p1: { x: number; y: number }, p2: { x: number; y: number }): NormalizedRect {
  const x = Math.min(p1.x, p2.x);
  const y = Math.min(p1.y, p2.y);
  return {
    x,
    y,
    width: Math.abs(p2.x - p1.x),
    height: Math.abs(p2.y - p1.y),
  };
}
