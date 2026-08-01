import {
  Component,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
  OnDestroy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Detection } from '../detection';
import { InferenceService } from '../inference.service';

export interface SessionFrame {
  index: number;
  timestamp: number;
  url: string;
  detections?: Detection[];
}

const SPEEDS = [0.5, 1, 2, 4, 8];

@Component({
  selector: 'app-session-player',
  imports: [FormsModule],
  templateUrl: './session-player.html',
  styleUrl: './session-player.css',
})
export class SessionPlayerComponent implements OnDestroy {
  private inference = inject(InferenceService);

  title = input('');
  frames = input<SessionFrame[]>([]);

  close = output<void>();

  currentIdx = signal(0);
  playing = signal(false);
  looping = signal(true);
  speed = signal(1);
  speedInput = signal('1');

  readonly speeds = SPEEDS;

  private canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('canvas');
  private imageCache = new Map<number, HTMLImageElement>();
  private timer: ReturnType<typeof setInterval> | null = null;

  totalFrames = computed(() => this.frames().length);
  currentFrame = computed(() => this.frames()[this.currentIdx()]);

  constructor() {
    effect(() => {
      this.frames();
      this.reset();
    });
  }

  get currentFrameLabel(): string {
    const f = this.currentFrame();
    return f ? this.formatDuration(f.timestamp) : '';
  }

  prevFrame() {
    const total = this.totalFrames();
    if (total === 0) return;
    this.currentIdx.update((i) => (i - 1 + total) % total);
    requestAnimationFrame(() => this.draw());
  }

  nextFrame() {
    const total = this.totalFrames();
    if (total === 0) return;
    this.currentIdx.update((i) => (i + 1) % total);
    requestAnimationFrame(() => this.draw());
  }

  goToFrame(idx: number) {
    this.currentIdx.set(idx);
    requestAnimationFrame(() => this.draw());
  }

  togglePlay() {
    if (this.playing()) {
      this.stop();
    } else {
      this.play();
    }
  }

  play() {
    if (this.totalFrames() === 0) return;
    this.playing.set(true);
    const interval = Math.round(150 / this.speed());
    this.timer = setInterval(() => {
      const total = this.totalFrames();
      if (total === 0) return;
      this.currentIdx.update((i) => {
        const next = i + 1;
        if (next >= total) {
          if (this.looping()) return 0;
          this.stop();
          return i;
        }
        return next;
      });
      requestAnimationFrame(() => this.draw());
    }, interval);
  }

  stop() {
    this.playing.set(false);
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  toggleLoop() {
    this.looping.update((l) => !l);
  }

  changeSpeed(s: number) {
    this.speed.set(s);
    this.speedInput.set(String(s));
    if (this.playing()) {
      this.stop();
      this.play();
    }
  }

  applyCustomSpeed() {
    const val = parseFloat(this.speedInput());
    if (isNaN(val) || val <= 0) {
      this.speedInput.set(String(this.speed()));
      return;
    }
    const clamped = Math.max(0.1, Math.min(val, 20));
    this.changeSpeed(clamped);
  }

  onClose() {
    this.close.emit();
  }

  private reset() {
    this.stop();
    this.currentIdx.set(0);
    for (const img of this.imageCache.values()) {
      img.src = '';
    }
    this.imageCache.clear();
    requestAnimationFrame(() => this.draw());
  }

  private getImage(idx: number): HTMLImageElement {
    let img = this.imageCache.get(idx);
    if (!img) {
      const frame = this.frames()[idx];
      img = new Image();
      img.src = frame ? frame.url : '';
      this.imageCache.set(idx, img);
    }
    return img;
  }

  private draw() {
    const canvasEl = this.canvasRef()?.nativeElement;
    if (!canvasEl) return;

    const frame = this.currentFrame();
    if (!frame) return;

    const img = this.getImage(this.currentIdx());
    if (!img.complete || img.naturalWidth === 0) {
      img.onload = () => requestAnimationFrame(() => this.draw());
      return;
    }

    this.inference.drawFrame(canvasEl, img, frame.detections);
    this.evict(30);
  }

  private evict(window: number) {
    const cur = this.currentIdx();
    for (const k of [...this.imageCache.keys()]) {
      if (Math.abs(k - cur) > window) {
        const img = this.imageCache.get(k);
        if (img) img.src = '';
        this.imageCache.delete(k);
      }
    }
  }

  private formatDuration(seconds: number): string {
    const s = Math.max(0, seconds);
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${String(sec).padStart(2, '0')}`;
  }

  ngOnDestroy() {
    this.stop();
    for (const img of this.imageCache.values()) {
      img.src = '';
    }
    this.imageCache.clear();
  }
}
