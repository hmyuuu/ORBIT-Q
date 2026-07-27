import { defineConfig } from "vitest/config";


export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    exclude: ["tests/rendered-html.test.mjs", "node_modules/**"],
  },
});
