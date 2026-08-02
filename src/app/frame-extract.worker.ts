import { createFile, DataStream } from 'mp4box';
import type { ISOFile, VisualSampleEntry } from 'mp4box';

interface StartMessage {
  type: 'start';
  buffer: ArrayBuffer;
  stride: number;
  maxDim: number;
  mime: string;
  quality: number;
}

type WorkerContext = {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent) => void) | null;
};

const ctx = self as unknown as WorkerContext;

ctx.onmessage = (event: MessageEvent) => {
  const msg = event.data as StartMessage;
  if (msg?.type !== 'start') return;
  runExtraction(msg).catch((err) => {
    ctx.postMessage({
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  });
};

function codecDescription(file: ISOFile, trackId: number): Uint8Array | undefined {
  const trak = file.getTrackById(trackId);
  const entries = trak?.mdia?.minf?.stbl?.stsd?.entries ?? [];
  for (const entry of entries) {
    const sampleEntry = entry as VisualSampleEntry;
    const box =
      sampleEntry.avcC ??
      sampleEntry.hvcC ??
      sampleEntry.vpcC ??
      sampleEntry.av1C ??
      (sampleEntry as unknown as { vvcC?: unknown }).vvcC;
    if (!box) continue;
    try {
      const stream = new DataStream();
      (box as unknown as { write(stream: DataStream): void }).write(stream);
      const bytes = new Uint8Array(stream.buffer, 0, stream.getPosition());
      return bytes.slice(8);
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function runExtraction(msg: StartMessage): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = createFile();
    const stride = Math.max(1, Math.round(msg.stride));
    const maxDim = Math.max(1, msg.maxDim);

    let decoder: VideoDecoder | null = null;
    let width = 0;
    let height = 0;
    let timescale = 0;
    let totalFrames = 0;
    let framesSubmitted = 0;
    let frameCounter = 0;
    let keptCount = 0;
    let finished = false;

    const keptPromises: Promise<void>[] = [];
    const canvas = new OffscreenCanvas(1, 1);
    let canvasCtx: OffscreenCanvasRenderingContext2D | null = null;

    const fail = (message: string) => {
      if (finished) return;
      finished = true;
      try {
        decoder?.close();
      } catch {
        /* noop */
      }
      reject(new Error(message));
    };

    const finish = async () => {
      if (finished) return;
      finished = true;
      try {
        if (decoder) await decoder.flush();
        await Promise.all(keptPromises);
      } catch {
        /* noop */
      } finally {
        try {
          decoder?.close();
        } catch {
          /* noop */
        }
      }
      ctx.postMessage({ type: 'done', total: keptCount });
      resolve();
    };

    file.onError = (module, message) => fail(`${module}: ${message}`);

    file.onReady = async (info) => {
      try {
        const vtrack = info.videoTracks[0];
        if (!vtrack) {
          fail('El archivo no tiene pista de video');
          return;
        }
        width = vtrack.video?.width ?? 0;
        height = vtrack.video?.height ?? 0;
        timescale = vtrack.timescale;
        totalFrames = vtrack.nb_samples || 0;
        if (width <= 0 || height <= 0 || timescale <= 0) {
          fail('No se pudieron leer los metadatos del video');
          return;
        }

        ctx.postMessage({
          type: 'meta',
          totalFrames,
          width,
          height,
          fps:
            totalFrames > 0 && info.duration > 0
              ? (totalFrames * info.timescale) / info.duration
              : 0,
        });

        const config: VideoDecoderConfig = {
          codec: vtrack.codec,
          codedWidth: width,
          codedHeight: height,
        };
        const description = codecDescription(file, vtrack.id);
        if (description) config.description = description;

        // onSamples se dispara de forma síncrona durante appendBuffer (tras
        // file.start()). Por eso encolamos los chunks aquí y decodificamos
        // después, cuando el VideoDecoder ya esté configurado.
        const queued: EncodedVideoChunk[] = [];
        file.onSamples = (_id, _user, samples) => {
          try {
            for (const s of samples) {
              if (!s.data) continue;
              queued.push(
                new EncodedVideoChunk({
                  type: s.is_sync ? 'key' : 'delta',
                  timestamp: Math.max(0, Math.round((s.cts * 1e6) / timescale)),
                  duration: s.duration
                    ? Math.max(0, Math.round((s.duration * 1e6) / timescale))
                    : undefined,
                  data: s.data,
                })
              );
            }
          } catch (err) {
            fail(err instanceof Error ? err.message : String(err));
          }
        };
        file.setExtractionOptions(vtrack.id, null, {
          nbSamples: Math.max(totalFrames, 1000),
        });
        file.start();

        let support: VideoDecoderSupport;
        try {
          support = await VideoDecoder.isConfigSupported(config);
        } catch {
          fail('Codec no soportado: ' + vtrack.codec);
          return;
        }
        if (!support.supported) {
          fail('Codec no soportado: ' + vtrack.codec);
          return;
        }

        const scale = Math.min(1, maxDim / Math.max(width, height));
        const outW = Math.max(1, Math.round(width * scale));
        const outH = Math.max(1, Math.round(height * scale));

        decoder = new VideoDecoder({
          output: (frame) => {
            frameCounter++;
            const isKept = frameCounter % stride === 0;
            if (isKept) {
              const index = keptCount;
              keptCount++;
              keptPromises.push(
                (async () => {
                  const timestamp = Math.max(0, frame.timestamp / 1e6);
                  try {
                    if (canvas.width !== outW || canvas.height !== outH) {
                      canvas.width = outW;
                      canvas.height = outH;
                      canvasCtx = canvas.getContext('2d');
                    }
                    canvasCtx!.drawImage(frame, 0, 0, outW, outH);
                    const blob = await canvas.convertToBlob({
                      type: msg.mime,
                      quality: msg.quality,
                    });
                    const buf = await blob.arrayBuffer();
                    ctx.postMessage({ type: 'frame', index, timestamp, buffer: buf }, [buf]);
                  } finally {
                    frame.close();
                  }
                })().catch((err) => fail(err instanceof Error ? err.message : String(err)))
              );
            } else {
              frame.close();
            }
          },
          error: (err) => fail(String(err?.message ?? err)),
        });
        decoder.configure(config);

        for (const chunk of queued) {
          if (finished || !decoder) break;
          framesSubmitted++;
          decoder.decode(chunk);
        }
        void finish();
      } catch (err) {
        fail(err instanceof Error ? err.message : String(err));
      }
    };

    try {
      const buffer = msg.buffer as ArrayBuffer & { fileStart: number };
      buffer.fileStart = 0;
      file.appendBuffer(buffer, true);
      file.flush();
    } catch (err) {
      fail(err instanceof Error ? err.message : String(err));
    }
  });
}
