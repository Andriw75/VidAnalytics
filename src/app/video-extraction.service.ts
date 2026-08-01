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

export interface ExtractionOptions {
  infer?: (blob: Blob) => Promise<Detection[]>;
  onProgress?: (done: number, total: number) => void;
  shouldCancel?: () => boolean;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime?: number }) => void) => number;
};

@Injectable({ providedIn: 'root' })
export class VideoExtractionService {
  private readonly supportsWebp = this.detectWebp();

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
   * Extrae fotogramas reproduciendo el video a alta velocidad en lugar de
   * hacer un "seek" por cada frame. Captura cuando currentTime cruza cada
   * instante objetivo (derivado del salto). Opcionalmente infiere por frame.
   */
  async extractByPlayback(
    video: HTMLVideoElement,
    meta: VideoMeta,
    salto: number,
    playbackRate: number,
    opts: ExtractionOptions = {}
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

    const mime = this.supportsWebp ? 'image/webp' : 'image/jpeg';
    const quality = this.supportsWebp ? 0.82 : 0.85;

    const captureBlob = (): Promise<Blob | null> =>
      new Promise((resolve) => canvas.toBlob(resolve, mime, quality));

    const targetTime = (idx: number) =>
      Math.min(fps ? (idx * stride) / fps : idx / stride, Math.max(0, meta.duration - 0.001));

    const pending = new Map<number, ExtractedFrame>();
    const tasks: Promise<void>[] = [];
    let nextIndex = 0;
    let finished = false;

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
        const cleanup = () => {
          video.removeEventListener('seeked', onSeeked);
          video.removeEventListener('error', onError);
          window.clearTimeout(timeout);
        };
        const onSeeked = () => {
          cleanup();
          capture(time);
          resolve();
        };
        const onError = () => {
          cleanup();
          resolve();
        };
        if (Math.abs(video.currentTime - time) < 0.001) {
          capture(time);
          resolve();
          return;
        }
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', onError, { once: true });
        timeout = window.setTimeout(onSeeked, 2000);
        video.currentTime = time;
      });

    let resolvePromise!: (frames: ExtractedFrame[]) => void;
    const done = new Promise<ExtractedFrame[]>((res) => (resolvePromise = res));

    const finish = async () => {
      if (finished) return;
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
      resolvePromise(frames);
    };

    const onEnded = () => {
      void finish();
    };
    video.addEventListener('ended', onEnded);

    const v = video as VideoWithRVFC;
    const useRVFC = typeof v.requestVideoFrameCallback === 'function';

    const step = () => {
      if (finished) return;
      if (opts.shouldCancel?.()) {
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
    video.playbackRate = playbackRate;
    video.currentTime = 0;
    video.play().catch(() => void finish());
    step();

    return done;
  }

  private detectWebp(): boolean {
    const canvas = document.createElement('canvas');
    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
  }
}
