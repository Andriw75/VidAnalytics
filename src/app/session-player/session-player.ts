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
import { Detection, ModelMetadata } from '../detection';
import { InferenceService } from '../inference.service';
import { poseLegend } from '../pose-renderer';
import {
  defaultFilters,
  ImageSize,
  NormalizedRect,
  SearchFilters,
  SearchResult,
  SuggestionEvent,
  Zone,
} from '../analysis/filter-types';
import { searchFrames } from '../analysis/frame-filter-engine';
import { rectFromPoints } from '../analysis/geometry';

export interface SessionFrame {
  index: number;
  timestamp: number;
  url: string;
  detections?: Detection[];
}

const SPEEDS = [0.5, 1, 2, 4, 8];
const ZONE_COLOR = '#ff4444';

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
  metadata = input<ModelMetadata | null>(null);
  imageSize = input<ImageSize | null>(null);

  close = output<void>();

  currentIdx = signal(0);
  playing = signal(false);
  looping = signal(true);
  speed = signal(1);
  speedInput = signal('1');

  showFilters = signal(false);
  filters = signal<SearchFilters>(defaultFilters());
  zones = signal<Zone[]>([]);
  editingZone = signal(false);
  result = signal<SearchResult | null>(null);
  selectedEvent = signal<SuggestionEvent | null>(null);

  readonly speeds = SPEEDS;
  readonly poseColors = poseLegend();
  readonly Math = Math;

  private canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('canvas');
  private imageCache = new Map<number, HTMLImageElement>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private draftStart: { x: number; y: number } | null = null;
  private draftRect: NormalizedRect | null = null;

  totalFrames = computed(() => this.frames().length);
  currentFrame = computed(() => this.frames()[this.currentIdx()]);
  isPose = computed(() => this.metadata()?.task === 'pose');
  classOptions = computed(() => {
    const classes = new Set<number>();
    for (const frame of this.frames()) {
      for (const detection of frame.detections ?? []) classes.add(detection.classId);
    }
    return Array.from(classes).sort((a, b) => a - b);
  });
  bboxFilter = computed(() => this.filters().bbox);
  keypointFilter = computed(() => this.filters().keypoints);
  currentEventIdx = computed(() => {
    const events = this.result()?.events ?? [];
    const selected = this.selectedEvent();
    return selected ? events.indexOf(selected) : -1;
  });
  eventCount = computed(() => this.result()?.events.length ?? 0);

  constructor() {
    effect(() => {
      this.frames();
      this.reset();
      this.result.set(null);
      this.selectedEvent.set(null);
    });
  }

  get currentFrameLabel(): string {
    const frame = this.currentFrame();
    return frame ? this.formatDuration(frame.timestamp) : '';
  }

  cocoClass(classId: number): string {
    return this.inference.cocoClass(classId);
  }

  toggleFilters() {
    this.showFilters.update((value) => !value);
  }

  setBbox(patch: Partial<SearchFilters['bbox']>) {
    this.filters.update((filters) => ({ ...filters, bbox: { ...filters.bbox, ...patch } }));
  }

  setKeypoints(patch: Partial<SearchFilters['keypoints']>) {
    this.filters.update((filters) => ({ ...filters, keypoints: { ...filters.keypoints, ...patch } }));
  }

  onBboxEnabledChange(event: Event) {
    this.setBbox({ enabled: (event.target as HTMLInputElement).checked });
  }

  onClassChange(event: Event) {
    const value = (event.target as HTMLSelectElement).value;
    this.setBbox({ classId: value === 'all' ? 'all' : Number(value) });
  }

  onMinCountChange(event: Event) {
    this.setBbox({ minCount: Number((event.target as HTMLInputElement).value) });
  }

  onMaxCountChange(event: Event) {
    const value = (event.target as HTMLInputElement).value;
    this.setBbox({ maxCount: value === '' ? null : Number(value) });
  }

  onMinConfChange(event: Event) {
    this.setBbox({ minConfidence: Number((event.target as HTMLInputElement).value) });
  }

  onZoneModeChange(event: Event) {
    this.setBbox({ zoneMode: (event.target as HTMLSelectElement).value as SearchFilters['bbox']['zoneMode'] });
  }

  onMinZoneAreaChange(event: Event) {
    this.setBbox({ minZoneArea: Number((event.target as HTMLInputElement).value) / 100 });
  }

  onKptEnabledChange(event: Event) {
    this.setKeypoints({ enabled: (event.target as HTMLInputElement).checked });
  }

  onMinVisibleChange(event: Event) {
    this.setKeypoints({ minVisible: Number((event.target as HTMLInputElement).value) });
  }

  onPresetChange(event: Event) {
    this.setKeypoints({ preset: (event.target as HTMLSelectElement).value as SearchFilters['keypoints']['preset'] });
  }

  onKptZoneModeChange(event: Event) {
    this.setKeypoints({ zoneMode: (event.target as HTMLSelectElement).value as SearchFilters['keypoints']['zoneMode'] });
  }

  onKptNameChange(event: Event) {
    this.setKeypoints({ zoneKeypoint: (event.target as HTMLSelectElement).value });
  }

  setZones(zones: Zone[]) {
    this.zones.set(zones);
    this.filters.update((filters) => ({ ...filters, zones }));
  }

  addZone() {
    const zone: Zone = {
      id: crypto.randomUUID?.() ?? `${Date.now()}`,
      name: `Zona ${this.zones().length + 1}`,
      rect: { x: 0.2, y: 0.2, width: 0.3, height: 0.3 },
      enabled: true,
      color: ZONE_COLOR,
    };
    this.setZones([...this.zones(), zone]);
    this.ensureZoneMode();
  }

  removeZone(id: string) {
    this.setZones(this.zones().filter((zone) => zone.id !== id));
  }

  toggleZone(id: string) {
    this.setZones(
      this.zones().map((zone) => (zone.id === id ? { ...zone, enabled: !zone.enabled } : zone))
    );
  }

  toggleEditZone() {
    this.editingZone.update((value) => !value);
    this.draftStart = null;
    this.draftRect = null;
  }

  onZonePointerDown(event: PointerEvent) {
    if (!this.editingZone()) return;
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;
    const point = this.canvasPoint(event, canvas);
    this.draftStart = point;
    this.draftRect = { x: point.x, y: point.y, width: 0, height: 0 };
  }

  onZonePointerMove(event: PointerEvent) {
    if (!this.editingZone() || !this.draftStart) return;
    const canvas = this.canvasRef()?.nativeElement;
    if (!canvas) return;
    const point = this.canvasPoint(event, canvas);
    this.draftRect = rectFromPoints(this.draftStart, point);
    requestAnimationFrame(() => this.draw());
  }

  onZonePointerUp() {
    if (this.draftRect && this.draftRect.width > 0.005 && this.draftRect.height > 0.005) {
      const zone: Zone = {
        id: crypto.randomUUID?.() ?? `${Date.now()}`,
        name: `Zona ${this.zones().length + 1}`,
        rect: this.draftRect,
        enabled: true,
        color: ZONE_COLOR,
      };
      this.setZones([...this.zones(), zone]);
      this.ensureZoneMode();
    }
    this.draftStart = null;
    this.draftRect = null;
    this.editingZone.set(false);
    requestAnimationFrame(() => this.draw());
  }

  /** Activa automáticamente la intersección si hay zonas pero ningún modo elegido. */
  private ensureZoneMode() {
    if (this.filters().bbox.zoneMode === 'none') {
      this.setBbox({ zoneMode: 'intersect' });
    }
  }

  hasEnabledZones(): boolean {
    return this.zones().some((zone) => zone.enabled);
  }

  private canvasPoint(event: PointerEvent, canvas: HTMLCanvasElement): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, nx)), y: Math.max(0, Math.min(1, ny)) };
  }

  runSearch() {
    const size = this.imageSize();
    if (!size) return;
    const searchResult = searchFrames(this.frames(), this.filters(), size);
    this.result.set(searchResult);
    const events = searchResult.events;
    if (events.length > 0) {
      this.selectEvent(events[0]);
    } else {
      this.selectedEvent.set(null);
    }
  }

  selectEvent(event: SuggestionEvent) {
    this.selectedEvent.set(event);
    this.goToFrame(event.startFrame);
  }

  selectEventIndex(index: number) {
    const events = this.result()?.events ?? [];
    if (index >= 0 && index < events.length) this.selectEvent(events[index]);
  }

  nextEvent() {
    const events = this.result()?.events ?? [];
    const index = this.currentEventIdx();
    if (index >= 0 && index < events.length - 1) this.selectEvent(events[index + 1]);
  }

  prevEvent() {
    const events = this.result()?.events ?? [];
    const index = this.currentEventIdx();
    if (index > 0) this.selectEvent(events[index - 1]);
  }

  currentFrameSuggestions(): number {
    const event = this.selectedEvent();
    if (!event) return 0;
    const frame = this.currentFrame();
    if (!frame) return 0;
    return event.frames.filter((suggestion) => suggestion.frameIndex === frame.index).length;
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
    this.drawZones(canvasEl);
    this.evict(30);
  }

  private drawZones(canvasEl: HTMLCanvasElement) {
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    const drawZone = (rect: NormalizedRect, color: string, dashed: boolean) => {
      const x = rect.x * canvasEl.width;
      const y = rect.y * canvasEl.height;
      const width = rect.width * canvasEl.width;
      const height = rect.height * canvasEl.height;

      ctx.fillStyle = color + '22';
      ctx.fillRect(x, y, width, height);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([6, 3]);
      ctx.strokeRect(x, y, width, height);
      if (dashed) ctx.setLineDash([]);
    };

    for (const zone of this.zones()) {
      if (zone.enabled) drawZone(zone.rect, zone.color, false);
    }

    if (this.draftRect) drawZone(this.draftRect, ZONE_COLOR, true);
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

  formatDuration(seconds: number): string {
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
