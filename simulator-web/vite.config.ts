import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        simulator: "index.html",
        visualRunner: "visual-runner.html",
      },
    },
  },
});
