/**
 * Vite plugin mounting the orbital-element API (/api/elements,
 * /api/elements/status) on the dev and preview servers.
 * All logic lives in ./elementsService.ts.
 */
import type { Logger, Plugin } from 'vite';
import type { Server as HttpServer } from 'node:http';
import type { Http2SecureServer } from 'node:http2';
import { createElementsService, type ElementsService, type ElementsServiceOptions } from './elementsService.ts';

export type CelestrakPluginOptions = Partial<Omit<ElementsServiceOptions, 'log'>>;

export function celestrakPlugin(opts: CelestrakPluginOptions = {}): Plugin {
  let root = process.cwd();

  function mount(
    middlewares: { use(fn: ElementsService['handle']): unknown },
    logger: Logger,
    httpServer: HttpServer | Http2SecureServer | null,
  ): ElementsService {
    const service = createElementsService({
      ...opts,
      rootDir: opts.rootDir ?? root,
      log: (msg) => logger.info(`[elements] ${msg}`, { timestamp: true }),
    });
    middlewares.use(service.handle);
    httpServer?.once('close', () => service.close());
    // Startup refresh: never blocks; the cache / bundled snapshot is served meanwhile.
    service.refreshIfDue();
    return service;
  }

  return {
    name: 'starlink-tracker:celestrak-elements',
    configResolved(config) {
      root = config.root;
    },
    configureServer(server) {
      mount(server.middlewares, server.config.logger, server.httpServer);
    },
    configurePreviewServer(server) {
      mount(server.middlewares, server.config.logger, server.httpServer);
    },
  };
}
