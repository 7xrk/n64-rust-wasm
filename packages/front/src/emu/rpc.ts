/** Minimal typed RPC over postMessage. The worker exposes an object of async
 * functions; the main thread gets a proxy with the same signatures. */

type AnyFn = (...args: never[]) => unknown;
export type Api = Record<string, AnyFn>;

export type Remote<T extends Api> = {
  [K in keyof T]: (...args: Parameters<T[K]>) => Promise<Awaited<ReturnType<T[K]>>>;
};

interface Call {
  id: number;
  method: string;
  args: unknown[];
}

interface Reply {
  id: number;
  result?: unknown;
  error?: string;
}

export interface EventMessage<E> {
  event: E;
}

/** Marks values to transfer instead of copy, for both arguments and results. */
const transfers = new WeakMap<object, Transferable[]>();

export function transfer<T extends object>(value: T, list: Transferable[]): T {
  transfers.set(value, list);
  return value;
}

export function expose(api: Api, scope: DedicatedWorkerGlobalScope): void {
  scope.onmessage = async (e: MessageEvent<Call>) => {
    const { id, method, args } = e.data;
    try {
      const result = await (api[method] as (...a: unknown[]) => unknown)(...args);
      const list = result && typeof result === 'object' ? (transfers.get(result) ?? []) : [];
      scope.postMessage({ id, result } satisfies Reply, list);
    } catch (err) {
      scope.postMessage({ id, error: err instanceof Error ? err.message : String(err) } satisfies Reply);
    }
  };
}

export function connect<T extends Api, E>(worker: Worker, onEvent: (event: E) => void): Remote<T> {
  let next = 1;
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  worker.onmessage = (e: MessageEvent<Reply | EventMessage<E>>) => {
    if ('event' in e.data) {
      onEvent(e.data.event);
      return;
    }
    const reply = e.data;
    const p = pending.get(reply.id);
    if (!p) return;
    pending.delete(reply.id);
    if (reply.error !== undefined) p.reject(new Error(reply.error));
    else p.resolve(reply.result);
  };
  return new Proxy({} as Remote<T>, {
    get:
      (_, method: string) =>
      (...args: unknown[]) =>
        new Promise((resolve, reject) => {
          const id = next++;
          pending.set(id, { resolve, reject });
          const list = args.flatMap((a) => (a && typeof a === 'object' ? (transfers.get(a) ?? []) : []));
          worker.postMessage({ id, method, args } satisfies Call, list);
        }),
  });
}
