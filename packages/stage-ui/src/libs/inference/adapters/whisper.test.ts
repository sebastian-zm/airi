import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MODEL_NAMES } from '../constants'

// Mock Worker globally since it's not available in Node. Eventa's webworkers
// main adapter (`createContext(worker)`) drives the worker through the
// `onmessage`/`onerror`/`onmessageerror` properties and `postMessage`, while
// the adapter attaches its own `addEventListener('error', …)` for device-loss
// resilience — so the mock supports both. Mirrors kokoro.test.ts.
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

const WORKER_URL = 'mock://whisper-worker'

describe('whisper adapter - lifecycle', () => {
  beforeEach(() => {
    MockWorker.instances.length = 0
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('should create adapter with idle state', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)
    expect(adapter.state).toBe('idle')
  })

  it('should transition to terminated state after calling terminate', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)
    adapter.terminate()
    expect(adapter.state).toBe('terminated')
  })

  it('should reject transcription before the model is ready without changing lifecycle state', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)

    await expect(adapter.transcribe({ audioFloat32: new Float32Array(), language: 'en' }))
      .rejects
      .toThrow('Model not loaded. Call load() first.')
    expect(adapter.state).toBe('idle')
  })

  it('should return an unsubscribe function from onMessage', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)

    const unsubscribe = adapter.onMessage(() => {})
    expect(typeof unsubscribe).toBe('function')
    // Unsubscribing twice must not throw.
    expect(() => unsubscribe()).not.toThrow()
    expect(() => unsubscribe()).not.toThrow()
  })
})

describe('whisper adapter - device loss resilience', () => {
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
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)

    expect(adapter.deviceLossCount).toBe(0)
    expect(adapter.manifest).toBeNull()
  })

  it('should reject a load whose signal is already aborted with AbortError', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)
    const controller = new AbortController()
    controller.abort('cancel preload')

    await expect(adapter.load(undefined, { signal: controller.signal }))
      .rejects
      .toMatchObject({ name: 'AbortError' })
  })

  it('should pass the caller abort signal through to the load queue', async () => {
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)
    const controller = new AbortController()

    const loading = adapter.load(undefined, { signal: controller.signal }).catch(() => {})

    await vi.waitFor(() => expect(enqueueMock).toHaveBeenCalled())

    expect(enqueueMock).toHaveBeenCalledWith(
      MODEL_NAMES.WHISPER,
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
    const { createWhisperAdapter } = await import('./whisper')
    const adapter = createWhisperAdapter(WORKER_URL)

    // Default enqueue runs the loader, which lazily creates the worker and opens
    // the (never-answered) load stream; emitting an 'error' then exercises the
    // adapter's device-loss telemetry while the load is still pending.
    const loading = adapter.load().catch(error => error)

    await vi.waitFor(() => expect(MockWorker.instances.length).toBeGreaterThan(0))

    const worker = MockWorker.instances.at(-1)!
    worker.emitError(new Error('WebGPU device lost while loading'))

    expect(adapter.deviceLossCount).toBe(1)
    expect(recordDeviceLoss).toHaveBeenCalledWith(expect.objectContaining({
      modelId: MODEL_NAMES.WHISPER,
      reason: 'unknown',
      occurredAt: expect.any(Number),
    }))

    adapter.terminate()
    void loading
  })
})
