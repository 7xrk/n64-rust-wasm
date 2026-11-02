import { defineConfig } from 'vite-plus';

export default defineConfig({
  worker: {
    format: 'es',
  },
  server: {
    // the agent's in-page helpers and the scratch files in tmp/ are loaded through /@fs and change while a TAS
    // session runs; reloading the page would drop the movie
    watch: { ignored: ['**/.claude/**', '**/tmp/**'] },
  },
});
