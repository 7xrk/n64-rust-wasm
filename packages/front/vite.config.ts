import { defineConfig } from 'vite-plus';

export default defineConfig({
  // Relative asset URLs, so the build works from a subpath such as a GitHub Pages project site.
  base: './',
  worker: {
    format: 'es',
  },
  server: {
    // the agent's in-page helpers and the scratch files in tmp/ are loaded through /@fs and change while a TAS
    // session runs; reloading the page would drop the movie
    watch: { ignored: ['**/.claude/**', '**/tmp/**'] },
  },
});
