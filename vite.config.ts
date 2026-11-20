import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { celestrakPlugin } from './server/celestrakPlugin.ts';

// Cesium ships pre-built Workers / Assets / Widgets that it loads at runtime
// from CESIUM_BASE_URL, so they are copied verbatim instead of bundled.
const cesiumSource = 'node_modules/cesium/Build/Cesium';
const cesiumBaseUrl = 'cesium';

export default defineConfig({
  define: {
    CESIUM_BASE_URL: JSON.stringify(`/${cesiumBaseUrl}`),
  },
  plugins: [
    viteStaticCopy({
      targets: [
        // vite-plugin-static-copy v4 keeps the matched path structure, so strip
        // the 4 leading segments (node_modules/cesium/Build/Cesium) to end up
        // with /cesium/Workers/..., /cesium/Assets/... etc.
        {
          src: `${cesiumSource}/{ThirdParty,Workers,Assets,Widgets}/**/*`,
          dest: cesiumBaseUrl,
          rename: { stripBase: 4 },
        },
      ],
    }),
    celestrakPlugin(),
  ],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 6000,
  },
});
