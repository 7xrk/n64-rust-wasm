import { defineConfig } from 'vite-plus';

export default defineConfig({
  fmt: {
    singleQuote: true,
    printWidth: 140,
    ignorePatterns: ['.serena/**', 'packages/core/crate/**'],
  },
  lint: {
    ignorePatterns: ['packages/core/crate/**'],
  },
});
