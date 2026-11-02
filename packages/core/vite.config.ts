import { defineConfig } from 'vite-plus';

export default defineConfig({
  pack: {
    entry: ['src/index.ts'],
    platform: 'browser',
    dts: true,
    copy: ['crate/target/wasm32-unknown-unknown/release/n64core.wasm'],
    loader: { '.wgsl': 'text' },
    plugins: [
      // The shaders are imported with Vite's `?raw` suffix so the app can consume the sources directly; pack has no such suffix.
      {
        name: 'raw',
        resolveId(source: string, importer?: string) {
          if (source.endsWith('?raw')) return this.resolve(source.slice(0, -'?raw'.length), importer);
        },
      },
    ],
  },
});
