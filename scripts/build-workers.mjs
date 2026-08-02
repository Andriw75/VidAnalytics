import { build } from 'esbuild';

await build({
  entryPoints: ['src/app/inference.worker.ts'],
  outfile: 'public/inference.worker.js',
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  legalComments: 'none',
  logLevel: 'warning',
});
