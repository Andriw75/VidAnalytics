import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { VideoStoreService, VideoSession } from './video-store.service';
import { InferenceService } from './inference.service';
import { VideoExtractionService, ProgressStats } from './video-extraction.service';
import { SessionPlayerComponent, SessionFrame } from './session-player/session-player';
import { Detection, VideoMeta } from './detection';

@Component({
  selector: 'app-root',
  imports: [SessionPlayerComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements AfterViewInit, OnDestroy {
  @ViewChild('imageCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('sourceImage') imageRef!: ElementRef<HTMLImageElement>;
  @ViewChild('sourceVideo') videoRef!: ElementRef<HTMLVideoElement>;

  imageUrl: string | null = null;
  selectedModel = 'yolo26n.tflite';
  models = ['yolo26n.tflite'];
  status = 'Inicializando LiteRT...';
  isLoading = true;
  detections: Detection[] = [];
  processing = false;
  inferenceTime: number | null = null;
  inferenceDone = false;

  videoUrl: string | null = null;
  videoFileName: string | null = null;
  videoMeta: VideoMeta | null = null;
  videoLoading = false;

  private videoFile: File | null = null;

  salto = 15;
  applyInference = false;
  extractEstimate: { fpsKept: number | null; count: number } | null = null;
  extracting = false;
  extractAbort = false;
  extractProgress: { done: number; total: number } | null = null;
  extractStatus = '';
  private extractStartTime = 0;

  sessions: VideoSession[] = [];
  selectedSession: VideoSession | null = null;
  sessionFrames: SessionFrame[] = [];

  constructor(
    private cdr: ChangeDetectorRef,
    private store: VideoStoreService,
    private inference: InferenceService,
    private extractSvc: VideoExtractionService
  ) {}

  async ngAfterViewInit() {
    try {
      await this.inference.init();
      this.status = 'LiteRT listo. Cargando modelo...';
      this.cdr.detectChanges();
      await this.loadModel();
    } catch (err) {
      this.status = `Error al iniciar LiteRT: ${err}`;
      this.isLoading = false;
      this.cdr.detectChanges();
    }
  }

  async loadModel() {
    try {
      const shape = await this.inference.loadModel(this.selectedModel);
      this.status = `Modelo listo (entrada: ${shape}). Selecciona una imagen.`;
      this.isLoading = false;
      this.loadSessions();
      this.cdr.detectChanges();
    } catch (err) {
      this.status = `Error al cargar modelo: ${err}`;
      this.isLoading = false;
      this.cdr.detectChanges();
    }
  }

  async onModelChange(event: Event) {
    const select = event.target as HTMLSelectElement;
    this.selectedModel = select.value;
    this.isLoading = true;
    this.status = 'Cargando modelo...';
    this.detections = [];
    this.inferenceDone = false;
    this.inferenceTime = null;
    this.cdr.detectChanges();
    await this.loadModel();
  }

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
      this.imageUrl = URL.createObjectURL(file);
      this.detections = [];
      this.inferenceDone = false;
      this.inferenceTime = null;
      this.cdr.detectChanges();
    }
  }

  async onVideoSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    input.value = '';
    this.videoFile = file;
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    if (this.videoRef) this.videoRef.nativeElement.pause();
    this.videoUrl = URL.createObjectURL(file);
    this.videoFileName = file.name;
    this.videoMeta = null;
    this.videoLoading = true;
    this.cdr.detectChanges();
  }

  async onVideoLoaded() {
    const video = this.videoRef.nativeElement;

    let duration = video.duration;
    if (!isFinite(duration) || duration <= 0) {
      duration = await this.extractSvc.getAccurateDuration(video);
    }

    const fps = await this.extractSvc.measureFps(video);
    const totalFrames = fps && duration > 0 ? Math.round(fps * duration) : null;

    this.videoMeta = {
      fileName: this.videoFileName ?? 'video',
      width: video.videoWidth,
      height: video.videoHeight,
      duration,
      fps,
      totalFrames,
    };
    this.salto = fps ? Math.max(1, Math.round(fps / 2)) : 15;
    this.computeExtractEstimate();
    this.videoLoading = false;
    this.cdr.detectChanges();
  }

  get saltoMax(): number {
    const meta = this.videoMeta;
    return meta?.fps ? Math.max(1, Math.ceil(meta.fps)) : 60;
  }

  computeExtractEstimate() {
    const meta = this.videoMeta;
    if (!meta) {
      this.extractEstimate = null;
      return;
    }
    this.extractEstimate = this.extractSvc.estimate(meta, this.salto);
  }

  onSaltoInput(event: Event) {
    const input = event.target as HTMLInputElement;
    const v = parseInt(input.value, 10);
    this.salto = Number.isFinite(v) && v > 0 ? v : 1;
    this.computeExtractEstimate();
  }

  onApplyInferenceChange(event: Event) {
    const input = event.target as HTMLInputElement;
    this.applyInference = input.checked;
  }

  formatFps(v: number | null): string {
    if (v === null) return 'fps no medible';
    return `${Math.round(v * 100) / 100} fps`;
  }

  sessionDate(ts: number): string {
    return new Date(ts).toLocaleString();
  }

  formatDuration(seconds: number): string {
    const s = Math.max(0, seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const secStr = sec.toFixed(2);
    if (h > 0) {
      return `${h}:${String(m).padStart(2, '0')}:${secStr.padStart(5, '0')}`;
    }
    return `${m}:${secStr.padStart(5, '0')}`;
  }

  cocoClass(classId: number): string {
    return this.inference.cocoClass(classId);
  }

  private formatInferProgress(done: number, total: number, stats?: ProgressStats): string {
    const pct = total > 0 ? (done >= total ? 100 : Math.floor((done / total) * 100)) : 0;
    let line = `Infiriendo ${done}/${total} (${pct}%)`;
    if (stats) {
      const elapsed = performance.now() - this.extractStartTime;
      const eta = Math.max(0, stats.etaMs);
      line +=
        ` · ${Math.round(stats.lastMs)}ms/frame` +
        (stats.avgMs ? ` · prom ${Math.round(stats.avgMs)}ms` : '') +
        ` · ETA ${this.fmtElapsed(eta)}` +
        ` · elapsed ${this.fmtElapsed(elapsed)}`;
    }
    return line;
  }

  private fmtElapsed(ms: number): string {
    const s = Math.max(0, ms / 1000);
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return `${m}m ${r}s`;
  }

  async extractFrames() {
    const meta = this.videoMeta;
    if (!meta || !this.videoRef || this.extracting) return;
    const estimate = this.extractSvc.estimate(meta, this.salto);
    if (estimate.count <= 0) return;

    const video = this.videoRef.nativeElement;

    this.extracting = true;
    this.extractAbort = false;
    this.extractStartTime = performance.now();
    this.extractProgress = { done: 0, total: estimate.count };
    this.extractStatus = this.applyInference
      ? 'Preparando extracción e inferencia...'
      : 'Preparando extracción...';
    this.cdr.detectChanges();

    try {
      const inferFn = this.applyInference
        ? (blob: Blob) => this.inference.inferBlob(blob)
        : undefined;

      const frames = await this.extractSvc.extract(
        this.videoFile,
        video,
        meta,
        this.salto,
        {
          infer: inferFn,
          onProgress: (done, total) => {
            this.extractProgress = { done, total };
            this.extractStatus =
              `Extrayendo ${done}/${total} (${Math.floor((done / total) * 100)}%) · ` +
              this.fmtElapsed(performance.now() - this.extractStartTime);
            this.cdr.detectChanges();
          },
          onFinalize: (done, total) => {
            this.extractProgress = { done, total };
            this.extractStatus =
              `Finalizando extracción ${done}/${total} · ` +
              this.fmtElapsed(performance.now() - this.extractStartTime);
            this.cdr.detectChanges();
          },
          onInferStart: (total) => {
            this.extractProgress = { done: 0, total };
            this.extractStatus = `Preparando inferencia: 0/${total} · elapsed ${this.fmtElapsed(performance.now() - this.extractStartTime)}`;
            this.cdr.detectChanges();
          },
          onInferProgress: (done, total, stats) => {
            this.extractProgress = { done, total };
            this.extractStatus = this.formatInferProgress(done, total, stats);
            this.cdr.detectChanges();
          },
          shouldCancel: () => this.extractAbort,
        }
      );

      if (this.extractAbort) {
        this.extractStatus = 'Extracción cancelada.';
      } else if (frames.length > 0) {
        this.extractStatus = `Guardando sesión: ${frames.length} fotogramas...`;
        this.cdr.detectChanges();
        const thumbnail = await this.extractSvc.makeThumbnail(frames[0].blob);
        const session: VideoSession = {
          name: `${meta.fileName} — salto ${this.salto}`,
          createdAt: Date.now(),
          fileName: meta.fileName,
          width: meta.width,
          height: meta.height,
          duration: meta.duration,
          fps: meta.fps,
          totalFrames: meta.totalFrames,
          salto: this.salto,
          count: frames.length,
          thumbnail,
        };
        try {
          await this.store.saveSession(session, frames);
          this.extractStatus =
            `Sesión guardada: ${frames.length} fotogramas` +
            (this.applyInference ? ' con detecciones.' : '.');
          await this.loadSessions();
        } catch (err) {
          this.extractStatus = `Error al guardar sesión: ${err}`;
        }
      } else {
        this.extractStatus = 'No se pudieron extraer fotogramas.';
      }
    } catch (err) {
      this.extractStatus = `Error en extracción: ${err}`;
    } finally {
      this.extracting = false;
      this.extractProgress = null;
      this.cdr.detectChanges();
    }
  }

  cancelExtract() {
    if (!this.extracting) return;
    this.extractAbort = true;
    this.extractStatus = 'Cancelando...';
    this.cdr.detectChanges();
    this.extractSvc.cancelActive();
    this.inference.cancelPending();
  }

  async loadSessions() {
    for (const s of this.sessions) {
      if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
    }
    const sessions = await this.store.listSessions();
    for (const s of sessions) {
      if (s.thumbnail) s.thumbUrl = URL.createObjectURL(s.thumbnail);
    }
    this.sessions = sessions;
    this.cdr.detectChanges();
  }

  async openSession(session: VideoSession) {
    if (this.selectedSession) this.closeSession();
    this.selectedSession = session;
    this.sessionFrames = [];
    this.cdr.detectChanges();
    const records = await this.store.getFrames(session.id!);
    this.sessionFrames = records.map((r) => ({
      index: r.index,
      timestamp: r.timestamp,
      url: URL.createObjectURL(r.blob),
      detections: r.detections,
    }));
    this.cdr.detectChanges();
  }

  closeSession() {
    for (const f of this.sessionFrames) URL.revokeObjectURL(f.url);
    this.sessionFrames = [];
    this.selectedSession = null;
  }

  async deleteSession(id: number) {
    await this.store.deleteSession(id);
    if (this.selectedSession?.id === id) this.closeSession();
    await this.loadSessions();
  }

  async clearSessions() {
    await this.store.clearAll();
    this.closeSession();
    await this.loadSessions();
  }

  ngOnDestroy() {
    if (this.videoUrl) URL.revokeObjectURL(this.videoUrl);
    if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
    this.closeSession();
    for (const s of this.sessions) {
      if (s.thumbUrl) URL.revokeObjectURL(s.thumbUrl);
    }
  }

  async runInference() {
    if (!this.inference.isReady || !this.imageUrl) return;
    if (this.processing) return;

    this.processing = true;
    this.inferenceDone = false;
    this.inferenceTime = null;
    this.status = 'Ejecutando inferencia...';
    this.cdr.detectChanges();
    await new Promise((r) => setTimeout(r, 0));

    const start = performance.now();

    try {
      const img = this.imageRef.nativeElement;
      this.detections = await this.inference.inferImage(img);

      this.inferenceTime = performance.now() - start;
      this.inference.drawFrame(this.canvasRef.nativeElement, img, this.detections);
      this.inferenceDone = true;
      this.status = `Inferencia completada en ${(this.inferenceTime / 1000).toFixed(2)}s — ${this.detections.length} objeto(s) detectado(s).`;
    } catch (err) {
      this.status = `Error en inferencia: ${err}`;
    }
    this.processing = false;
    this.cdr.detectChanges();
  }
}
