import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MODEL_NAMES } from '../constants'

// Mock Worker globally since it's not available in Node. Eventa's webworkers
// main adapter (`createContext(worker)`) drives the worker through the
// `onmessage`/`onerror`/`onmessageerror` properties and `postMessage`, while
// the adapter attaches its own `addEventListener('error', …)`. Mirrors
// kokoro.test.ts; background removal has no device-loss/restart policy.
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
}
vi.stubGlobal('Worker', MockWorker)

// Mock dependencies that require browser APIs or Vue
vi.mock('../../../composables/use-inference-status', () => ({
  updateInferenceStatus: vi.fn(),
  removeInferenceStatus: vi.fn(),
}))

const enqueueMock = vi.fn((_id: string, _p: number, loader: () => Promise<unknown>) => loader())
vi.mock('../coordinator', () => ({
  getGPUCoordinator: () => ({
    requestAllocation: vi.fn(() => ({ modelId: 'test', estimatedBytes: 0 })),
    release: vi.fn(),
    touch: vi.fn(),
    recordDeviceLoss: vi.fn(),
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

describe('background removal adapter - lifecycle', () => {
  beforeEach(() => {
    enqueueMock.mockClear()
    enqueueMock.mockImplementation((_id: string, _p: number, loader: () => Promise<unknown>) => loader())
    MockWorker.instances.length = 0
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('should create adapter with idle state', async () => {
    const { createBackgroundRemovalAdapter } = await import('./background-removal')
    const adapter = createBackgroundRemovalAdapter()
    expect(adapter.state).toBe('idle')
  })

  it('should transition to terminated state after calling terminate', async () => {
    const { createBackgroundRemovalAdapter } = await import('./background-removal')
    const adapter = createBackgroundRemovalAdapter()
    adapter.terminate()
    expect(adapter.state).toBe('terminated')
  })

  it('should reject processing before the model is ready without changing lifecycle state', async () => {
    const { createBackgroundRemovalAdapter } = await import('./background-removal')
    const adapter = createBackgroundRemovalAdapter()

    // NOTICE:
    // ImageData is a DOM type with no Node global. The not-loaded guard rejects
    // before the adapter ever reads the image, so a structural stand-in is enough.
    // Root cause: this is a node-project unit test, not Vitest browser mode.
    // Removal condition: move to a *.browser.test.ts if real ImageData (alpha
    // mask application) handling needs coverage.
    const fakeImage = { data: new Uint8ClampedArray(4), width: 1, height: 1 } as unknown as ImageData

    await expect(adapter.processImage(fakeImage))
      .rejects
      .toThrow('Model not loaded. Call load() first.')
    expect(adapter.state).toBe('idle')
  })

  it('should reject a load whose signal is already aborted with AbortError', async () => {
    const { createBackgroundRemovalAdapter } = await import('./background-removal')
    const adapter = createBackgroundRemovalAdapter()
    const controller = new AbortController()
    controller.abort('cancel preload')

    await expect(adapter.load(undefined, { signal: controller.signal }))
      .rejects
      .toMatchObject({ name: 'AbortError' })
  })

  it('should pass the caller abort signal through to the load queue', async () => {
    const { createBackgroundRemovalAdapter } = await import('./background-removal')
    const adapter = createBackgroundRemovalAdapter()
    const controller = new AbortController()

    const loading = adapter.load(undefined, { signal: controller.signal }).catch(() => {})

    await vi.waitFor(() => expect(enqueueMock).toHaveBeenCalled())

    expect(enqueueMock).toHaveBeenCalledWith(
      MODEL_NAMES.BG_REMOVAL,
      expect.any(Number),
      expect.any(Function),
      { signal: controller.signal },
    )
    const worker = MockWorker.instances.at(-1)!
    expect(worker.postMessage).toHaveBeenCalled()

    controller.abort('cancel preload')
    await loading
  })
})
