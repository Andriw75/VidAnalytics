import { Injectable } from '@angular/core';
import { Detection, VideoMeta } from './detection';

export interface ExtractedFrame {
  index: number;
  timestamp: number;
  blob: Blob;
  detections?: Detection[];
}

export interface ExtractionEstimate {
  fpsKept: number | null;
  count: number;
}

export interface ProgressStats {
  lastMs: number;
  avgMs: number;
  elapsedMs: number;
  etaMs: number;
}

export interface ExtractionOptions {
  infer?: (blob: Blob) => Promise<Detection[]>;
  onProgress?: (done: number, total: number) => void;
  onFinalize?: (done: number, total: number) => void;
  onInferStart?: (total: number) => void;
  onInferProgress?: (done: number, total: number, stats?: ProgressStats) => void;
  shouldCancel?: () => boolean;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime?: number }) => void) => number;
};

class CancellationToken {
  cancelled = false;
  private handlers = new Set<() => void>();

  add(handler: () => void): () => void {
    if (this.cancelled) {
      handler();
      return () => {};
    }
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  cancel(): void {
    if (this.cancelled) return;
    this.cancelled = true;
    const handlers = [...this.handlers];
    this.handlers.clear();
    for (const handler of handlers) handler();
  }
}

@Injectable({ providedIn: 'root' })
export class VideoExtractionService {
  private readonly supportsWebp = this.detectWebp();
  private activeToken: CancellationToken | null = null;

  cancelActive(): void {
    this.activeToken?.cancel();
  }

  estimate(meta: VideoMeta, salto: number): ExtractionEstimate {
    const stride = Math.max(1, Math.round(salto));
    if (meta.fps && meta.totalFrames) {
      return { fpsKept: meta.fps / stride, count: Math.ceil(meta.totalFrames / stride) };
    }
    if (meta.fps && meta.duration > 0) {
      const tf = Math.round(meta.fps * meta.duration);
      return { fpsKept: meta.fps / stride, count: Math.ceil(tf / stride) };
    }
    return { fpsKept: null, count: Math.ceil(meta.duration * stride) };
  }

  /**
   * Método unificado de extracción. Usa WebCodecs (VideoDecoder + mp4box)
   * cuando es posible (máxima velocidad, frame-exacto); si no, cae a la
   * reproducción del <video> a la máxima velocidad permitida.
   */
  async extract(
    file: File | null,
    video: HTMLVideoElement,
    meta: VideoMeta,
    salto: number,
    opts: ExtractionOptions = {}
  ): Promise<ExtractedFrame[]> {
    const token = new CancellationToken();
    this.activeToken = token;

    try {
      let useWebCodecs = false;
      if (
        file &&
        typeof VideoDecoder !== 'undefined' &&
        typeof Worker !== 'undefined' &&
        typeof OffscreenCanvas !== 'undefined'
      ) {
        useWebCodecs = await this.isMp4(file);
      }
      if (token.cancelled) return [];
      if (useWebCodecs) {
        return await this.extractWebCodecs(file!, meta, salto, opts, token);
      }
      return await this.extractByPlayback(video, meta, salto, opts, token);
    } finally {
      if (this.activeToken === token) this.activeToken = null;
    }
  }

  private async extractWebCodecs(
    file: File,
    meta: VideoMeta,
    salto: number,
    opts: ExtractionOptions = {},
    token: CancellationToken
  ): Promise<ExtractedFrame[]> {
    const stride = Math.max(1, Math.round(salto));
    const buffer = await this.readFileBuffer(file, token);
    if (!buffer || token.cancelled) return [];
    const mime = this.captureMime;
    const quality = this.captureQuality;
    const estimate = this.estimate(meta, stride);

    return new Promise<ExtractedFrame[]>((resolve, reject) => {
      const worker = new Worker(new URL('./frame-extract.worker.ts', import.meta.url), {
        type: 'module',
      });
      const byIndex = new Map<number, ExtractedFrame>();
      let expectedTotal = 0;
      let terminated = false;
      let currentFrames: ExtractedFrame[] = [];
      let removeCancel = () => {};

      const timeout = window.setTimeout(() => {
        rejectOnce(new Error('Timeout en la extracción'));
      }, 120000);

      const settle = (frames: ExtractedFrame[]) => {
        if (terminated) return;
        terminated = true;
        window.clearTimeout(timeout);
        removeCancel();
        worker.terminate();
        resolve(frames);
      };

      const rejectOnce = (error: Error) => {
        if (terminated) return;
        terminated = true;
        window.clearTimeout(timeout);
        removeCancel();
        worker.terminate();
        reject(error);
      };

      removeCancel = token.add(() => {
        currentFrames = [...byIndex.values()]
          .sort((a, b) => a.index - b.index)
          .filter((frame) => frame.blob.size > 0);
        settle(currentFrames);
      });

      worker.onerror = (event) => {
        rejectOnce(new Error(`Error del worker: ${event.message || 'desconocido'}`));
      };

      worker.onmessageerror = () => {
        rejectOnce(new Error('Error de mensaje en el worker'));
      };

      worker.onmessage = async (event) => {
        if (terminated) return;
        const msg = event.data;

        switch (msg.type) {
          case 'meta':
            expectedTotal = msg.totalFrames;
            break;

          case 'frame': {
            if (token.cancelled) return;
            const blob = new Blob([msg.buffer], { type: mime });
            byIndex.set(msg.index, {
              index: msg.index,
              timestamp: msg.timestamp,
              blob,
            });
            const total = expectedTotal > 0 ? Math.ceil(expectedTotal / stride) : estimate.count;
            opts.onProgress?.(byIndex.size, total);
            break;
          }

          case 'finalizing': {
            const total = expectedTotal > 0 ? Math.ceil(expectedTotal / stride) : estimate.count;
            opts.onFinalize?.(byIndex.size, total);
            break;
          }

          case 'done': {
            window.clearTimeout(timeout);
            const ordered = [...byIndex.values()].sort((a, b) => a.index - b.index);
            currentFrames = ordered;
            worker.terminate();
            try {
              if (opts.infer && !opts.shouldCancel?.()) {
                opts.onInferStart?.(ordered.length);
                const start = performance.now();
                for (let i = 0; i < ordered.length; i++) {
                  if (token.cancelled || opts.shouldCancel?.()) {
                    token.cancel();
                    return;
                  }
                  const t0 = performance.now();
                  ordered[i].detections = await opts.infer(ordered[i].blob);
                  const now = performance.now();
                  const lastMs = now - t0;
                  const elapsedMs = now - start;
                  const avgMs = elapsedMs / (i + 1);
                  const etaMs = avgMs * (ordered.length - (i + 1));
                  opts.onInferProgress?.(i + 1, ordered.length, { lastMs, avgMs, elapsedMs, etaMs });
                }
              }
            } catch (err) {
              if (token.cancelled || opts.shouldCancel?.()) {
                settle(currentFrames);
              } else {
                rejectOnce(err instanceof Error ? err : new Error(String(err)));
              }
              return;
            }
            if (token.cancelled || opts.shouldCancel?.()) {
              settle(currentFrames);
            } else {
              settle(ordered);
            }
            break;
          }

          case 'error':
            rejectOnce(new Error(msg.message));
            break;
        }
      };

      worker.postMessage({ type: 'start', buffer, stride, maxDim: 640, mime, quality }, [buffer]);
    });
  }

  private readFileBuffer(file: File, token: CancellationToken): Promise<ArrayBuffer | null> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      const removeCancel = token.add(() => reader.abort());
      reader.onload = () => {
        removeCancel();
        resolve(reader.result as ArrayBuffer);
      };
      reader.onerror = () => {
        removeCancel();
        reject(reader.error ?? new Error('No se pudo leer el video'));
      };
      reader.onabort = () => {
        removeCancel();
        resolve(null);
      };
      reader.readAsArrayBuffer(file);
    });
  }

  private async isMp4(file: File): Promise<boolean> {
    try {
      const buf = new Uint8Array(await file.slice(0, 16).arrayBuffer());
      return String.fromCharCode(buf[4], buf[5], buf[6], buf[7]) === 'ftyp';
    } catch {
      return false;
    }
  }

  async getAccurateDuration(video: HTMLVideoElement): Promise<number> {
    return new Promise((resolve) => {
      const onDuration = () => resolve(video.duration);
      const onError = () => resolve(video.duration || 0);
      video.currentTime = 1e7;
      video.addEventListener('durationchange', onDuration, { once: true });
      video.addEventListener('error', onError, { once: true });
      window.setTimeout(() => resolve(video.duration || 0), 3000);
    });
  }

  measureFps(video: HTMLVideoElement): Promise<number | null> {
    return new Promise((resolve) => {
      const v = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: (now: number) => void) => number;
      };

      if (typeof v.requestVideoFrameCallback !== 'function') {
        resolve(null);
        return;
      }

      let firstNow: number | null = null;
      let frames = 0;
      let done = false;

      const finish = () => {
        if (done) return;
        done = true;
        video.pause();
        const elapsed = firstNow !== null ? (performance.now() - firstNow) / 1000 : 0;
        const fps = firstNow !== null && elapsed > 0 ? frames / elapsed : null;
        resolve(fps ? Math.round(fps * 100) / 100 : null);
      };

      const timer = window.setTimeout(finish, 2500);
      video.addEventListener('ended', finish, { once: true });

      const step = (now: number) => {
        if (firstNow === null) firstNow = now;
        frames++;
        if (firstNow !== null && (performance.now() - firstNow) / 1000 >= 1) {
          window.clearTimeout(timer);
          finish();
          return;
        }
        v.requestVideoFrameCallback!(step);
      };

      v.requestVideoFrameCallback(step);
      video.muted = true;
      video.playbackRate = 1;
      video.currentTime = 0;
      video.play().catch(() => {
        window.clearTimeout(timer);
        finish();
      });
    });
  }

  async makeThumbnail(blob: Blob): Promise<Blob | undefined> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const maxDim = 160;
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        canvas.getContext('2d')!.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        canvas.toBlob(
          (out) => resolve(out ?? undefined),
          this.supportsWebp ? 'image/webp' : 'image/jpeg',
          0.7
        );
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(undefined);
      };
      img.src = url;
    });
  }

  /**
   * Extracción por reproducción (fallback): reproduce el video a la máxima
   * velocidad que el navegador permita y captura cuando currentTime cruza
   * cada instante objetivo (derivado del salto).
   */
  async extractByPlayback(
    video: HTMLVideoElement,
    meta: VideoMeta,
    salto: number,
    opts: ExtractionOptions = {},
    token: CancellationToken
  ): Promise<ExtractedFrame[]> {
    const stride = Math.max(1, Math.round(salto));
    const fps = meta.fps && meta.fps > 0 ? meta.fps : null;
    const estimate = this.estimate(meta, stride);
    const total = estimate.count;
    if (total <= 0) return [];

    const canvas = document.createElement('canvas');
    const maxDim = 640;
    const scale = Math.min(1, maxDim / Math.max(meta.width, meta.height));
    canvas.width = Math.max(1, Math.round(meta.width * scale));
    canvas.height = Math.max(1, Math.round(meta.height * scale));
    const ctx = canvas.getContext('2d')!;

    const mime = this.captureMime;
    const quality = this.captureQuality;

    const captureBlob = (): Promise<Blob | null> =>
      new Promise((resolve) => canvas.toBlob(resolve, mime, quality));

    const targetTime = (idx: number) =>
      Math.min(fps ? (idx * stride) / fps : idx / stride, Math.max(0, meta.duration - 0.001));

    const pending = new Map<number, ExtractedFrame>();
    const tasks: Promise<void>[] = [];
    let nextIndex = 0;
    let finished = false;
    let cancelCurrentSeek = () => {};

    const capture = (time: number) => {
      const idx = nextIndex;
      nextIndex++;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const rec: ExtractedFrame = { index: idx, timestamp: time, blob: new Blob() };
      pending.set(idx, rec);
      const task = captureBlob()
        .then(async (blob) => {
          if (!blob) return;
          rec.blob = blob;
          if (opts.infer) {
            rec.detections = await opts.infer(blob);
          }
        })
        .catch(() => {});
      tasks.push(task);
      opts.onProgress?.(nextIndex, total);
    };

    const seekCapture = (time: number): Promise<void> =>
      new Promise((resolve) => {
        let timeout = 0;
        let resolved = false;
        const cleanup = () => {
          video.removeEventListener('seeked', onSeeked);
          video.removeEventListener('error', onError);
          window.clearTimeout(timeout);
          cancelCurrentSeek = () => {};
        };
        const complete = (captureFrame: boolean) => {
          if (resolved) return;
          resolved = true;
          cleanup();
          if (captureFrame && !token.cancelled && !settled) capture(time);
          resolve();
        };
        const onSeeked = () => complete(true);
        const onError = () => complete(false);
        cancelCurrentSeek = () => complete(false);
        if (Math.abs(video.currentTime - time) < 0.001) {
          complete(true);
          return;
        }
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', onError, { once: true });
        timeout = window.setTimeout(onSeeked, 2000);
        video.currentTime = time;
      });

    let resolvePromise!: (frames: ExtractedFrame[]) => void;
    const done = new Promise<ExtractedFrame[]>((res) => (resolvePromise = res));
    let settled = false;

    const settle = (frames: ExtractedFrame[]) => {
      if (settled) return;
      settled = true;
      resolvePromise(frames);
    };

    const finish = async () => {
      if (finished || settled) return;
      finished = true;
      video.removeEventListener('ended', onEnded);

      if (!opts.shouldCancel?.() && nextIndex < total) {
        while (nextIndex < total) {
          await seekCapture(targetTime(nextIndex));
        }
      }

      await Promise.all(tasks);
      video.pause();
      video.currentTime = 0;

      const frames = [...pending.values()]
        .sort((a, b) => a.index - b.index)
        .filter((f) => f.blob.size > 0);
      settle(frames);
    };

    function onEnded() {
      void finish();
    }

    token.add(() => {
      finished = true;
      video.pause();
      video.removeEventListener('ended', onEnded);
      cancelCurrentSeek();
      const frames = [...pending.values()]
        .sort((a, b) => a.index - b.index)
        .filter((frame) => frame.blob.size > 0);
      settle(frames);
    });

    video.addEventListener('ended', onEnded);

    const v = video as VideoWithRVFC;
    const useRVFC = typeof v.requestVideoFrameCallback === 'function';

    const step = () => {
      if (finished) return;
      if (token.cancelled || opts.shouldCancel?.()) {
        token.cancel();
        void finish();
        return;
      }
      const t = video.currentTime;
      if (nextIndex < total && t >= targetTime(nextIndex)) {
        capture(t);
      }
      if (nextIndex >= total || video.ended) {
        void finish();
        return;
      }
      if (useRVFC) {
        v.requestVideoFrameCallback!(step);
      } else {
        requestAnimationFrame(step);
      }
    };

    video.pause();
    video.muted = true;
    video.playbackRate = this.maxPlaybackRate(video);
    video.currentTime = 0;
    video.play().catch(() => void finish());
    step();

    return done;
  }

  private get captureMime(): string {
    return this.supportsWebp ? 'image/webp' : 'image/jpeg';
  }

  private get captureQuality(): number {
    return this.supportsWebp ? 0.82 : 0.85;
  }

  private maxPlaybackRate(video: HTMLVideoElement): number {
    try {
      video.playbackRate = 16;
      return video.playbackRate > 0 ? video.playbackRate : 16;
    } catch {
      return 16;
    }
  }

  private detectWebp(): boolean {
    const canvas = document.createElement('canvas');
    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
  }
}
