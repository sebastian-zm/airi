import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Mock Worker globally since it's not available in Node. Eventa's webworkers
// main adapter (`createContext(worker)`) drives the worker through the
// `onmessage`/`onerror`/`onmessageerror` properties and `postMessage`, while
// the adapter attaches its own `addEventListener('error', …)` for device-loss
// resilience — so the mock supports both.
class MockWorker {
  static instances: MockWorker[] = []

  onmessage: ((event: any) => void) | null = null
  onerror: ((event: any) => void) | null = null
  onmessageerror: ((event: any) => void) | null = null

  listeners = new Map<string, Set<(event: any) => void>>()
  addEventListener = vi.fn((type: string, listener: (event: any) => void) => {
    if (!this.listeners.has(type))
      this.listeners.set(type, new Set())
    this.listeners.get(type)!.add(listener)
  })

  removeEventListener = vi.fn((type: string, listener: (event: any) => void) => {
    this.listeners.get(type)?.delete(listener)
  })

  postMessage = vi.fn()
  terminate = vi.fn()

  constructor() {
    MockWorker.instances.push(this)
  }

  /** Simulate a fatal worker 'error' event (e.g. WebGPU device loss). */
  emitError(error: unknown): void {
    for (const listener of this.listeners.get('error') ?? [])
      listener({ error })
  }
}
vi.stubGlobal('Worker', MockWorker)

// Mock dependencies that require browser APIs or Vue
vi.mock('../../../composables/use-inference-status', () => ({
  updateInferenceStatus: vi.fn(),
  removeInferenceStatus: vi.fn(),
}))

const recordDeviceLoss = vi.fn()
const enqueueMock = vi.fn((_id: string, _p: number, loader: () => Promise<unknown>) => loader())
vi.mock('../coordinator', () => ({
  getGPUCoordinator: () => ({
    requestAllocation: vi.fn(() => ({ modelId: 'test', estimatedBytes: 0 })),
    release: vi.fn(),
    touch: vi.fn(),
    recordDeviceLoss,
  }),
  getLoadQueue: () => ({
    enqueue: enqueueMock,
  }),
  MODEL_VRAM_ESTIMATES: {},
}))

vi.mock('@proj-airi/stage-shared', () => ({
  defaultPerfTracer: {
    withMeasure: vi.fn((_cat: string, _name: string, fn: () => unknown) => fn()),
  },
}))

describe('kokoro adapter - singleton recovery', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('should create adapter with idle state', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()
    expect(adapter.state).toBe('idle')
  })

  it('should transition to terminated state after calling terminate', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()
    adapter.terminate()
    expect(adapter.state).toBe('terminated')
  })

  it('should expose state getter correctly across transitions', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()

    expect(adapter.state).toBe('idle')
    adapter.terminate()
    expect(adapter.state).toBe('terminated')
  })

  it('should reject generation before the model is ready without changing lifecycle state', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()

    await expect(adapter.generate('hello', 'af_heart' as any)).rejects.toThrow('Model not loaded. Call loadModel() first.')
    expect(adapter.state).toBe('idle')
  })
})

describe('classifyError phase integration', () => {
  it('should produce LOAD_FAILED for load-phase errors', async () => {
    const { classifyError } = await import('../protocol')
    expect(classifyError(new Error('shader compilation failed'), 'load')).toBe('LOAD_FAILED')
  })

  it('should produce INFERENCE_FAILED for inference-phase errors', async () => {
    const { classifyError } = await import('../protocol')
    expect(classifyError(new Error('tensor shape mismatch'), 'inference')).toBe('INFERENCE_FAILED')
  })
})

describe('kokoro adapter - device loss resilience', () => {
  beforeEach(() => {
    recordDeviceLoss.mockClear()
    enqueueMock.mockClear()
    enqueueMock.mockImplementation((_id: string, _p: number, loader: () => Promise<unknown>) => loader())
    MockWorker.instances.length = 0
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('should start with zero device-loss count and null manifest', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()

    expect(adapter.deviceLossCount).toBe(0)
    expect(adapter.manifest).toBeNull()
  })

  it('should expose manifest and deviceLossCount as readonly getters', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()

    // Getters should be defined and callable
    expect(typeof adapter.deviceLossCount).toBe('number')
    // Manifest is explicitly null before any load
    expect(adapter.manifest).toBeNull()
  })

  it('should reject a load whose signal is already aborted with AbortError', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()
    const controller = new AbortController()
    controller.abort('cancel preload')

    await expect(adapter.loadModel('q4', 'webgpu', { signal: controller.signal }))
      .rejects
      .toMatchObject({ name: 'AbortError' })
  })

  it('should pass the caller abort signal through to the load queue', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()
    const controller = new AbortController()

    const loading = adapter.loadModel('q4', 'webgpu', { signal: controller.signal }).catch(() => {})

    await vi.waitFor(() => expect(enqueueMock).toHaveBeenCalled())

    expect(enqueueMock).toHaveBeenCalledWith(
      'kokoro-q4',
      expect.any(Number),
      expect.any(Function),
      { signal: controller.signal },
    )
    const worker = MockWorker.instances.at(-1)!
    // Eventa forwards the load request over the wire; the exact envelope is
    // internal, but a request must have been posted to the worker.
    expect(worker.postMessage).toHaveBeenCalled()

    controller.abort('cancel preload')
    await loading
  })

  it('should classify worker device-loss errors before restarting', async () => {
    const { createKokoroAdapter } = await import('./kokoro')
    const adapter = createKokoroAdapter()

    enqueueMock.mockImplementationOnce(() => new Promise(() => {}))
    const loading = adapter.loadModel('q4', 'webgpu').catch(error => error)

    await vi.waitFor(() => expect(MockWorker.instances.length).toBeGreaterThan(0))

    const worker = MockWorker.instances.at(-1)!
    // Eventa wires worker.onerror → workerErrorEvent → adapter.handleWorkerError.
    worker.emitError(new Error('WebGPU device lost while loading'))

    expect(adapter.deviceLossCount).toBe(1)
    expect(recordDeviceLoss).toHaveBeenCalledWith(expect.objectContaining({
      modelId: 'kokoro-q4',
      reason: 'unknown',
      occurredAt: expect.any(Number),
    }))

    adapter.terminate()
    void loading
  })
})
