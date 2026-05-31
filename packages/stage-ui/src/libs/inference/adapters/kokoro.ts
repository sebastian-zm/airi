/**
 * Kokoro TTS inference adapter.
 *
 * Talks to the Kokoro worker over the Eventa inference contract
 * (`libs/inference/contract.ts`): load is a server-streaming invoke, generate
 * is a unary invoke whose PCM buffer is transferred back zero-copy. All
 * resilience policy (device-loss restart, load serialization, GPU accounting,
 * mutex) lives here on the main thread.
 */

import type { VoiceKey, Voices } from '../../../workers/kokoro/types'
import type { AllocationToken } from '../gpu-resource-coordinator'
import type { ProgressPayload } from '../protocol'

import { defineInvoke, defineStreamInvoke } from '@moeru/eventa'
import { createContext } from '@moeru/eventa/adapters/webworkers'
import { defaultPerfTracer } from '@proj-airi/stage-shared'
import { Mutex } from 'async-mutex'

import { removeInferenceStatus, updateInferenceStatus } from '../../../composables/use-inference-status'
import { DEVICE_LOSS_WASM_THRESHOLD, MAX_RESTARTS, MODEL_NAMES, RESTART_DELAY_MS, TIMEOUTS } from '../constants'
import { consumeLoadStream, kokoroGenerateEvent, kokoroLoadEvent, signalWithTimeout } from '../contract'
import { getGPUCoordinator, getLoadQueue, MODEL_VRAM_ESTIMATES } from '../coordinator'
import { LOAD_PRIORITY } from '../load-queue'
import { classifyDeviceLossReason, classifyError, InferenceAbortError, throwIfAborted } from '../protocol'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface KokoroAdapter {
  /**
   * Load a TTS model with the given quantization and device.
   * Pass `options.signal` to cancel the load; the returned promise will
   * reject with `InferenceAbortError` (name: `'AbortError'`).
   */
  loadModel: (
    quantization: string,
    device: string,
    options?: {
      onProgress?: (p: ProgressPayload) => void
      signal?: AbortSignal
    },
  ) => Promise<Voices>

  /**
   * Generate speech audio from text.
   * Pass `options.signal` to cancel; rejects with `InferenceAbortError`.
   */
  generate: (
    text: string,
    voice: VoiceKey,
    options?: { signal?: AbortSignal },
  ) => Promise<ArrayBuffer>

  /** Get the voices from the last loaded model */
  getVoices: () => Voices

  /** Terminate the worker */
  terminate: () => void

  /** Current state */
  readonly state: 'idle' | 'loading' | 'ready' | 'running' | 'error' | 'terminated'

  /**
   * Snapshot of the last successful load config, or null if never loaded.
   * `device` reflects the device actually used (post WASM promotion / worker
   * fallback), which may differ from the device requested by the caller.
   */
  readonly manifest: { quantization: string, device: string } | null

  /** Number of WebGPU device-loss events observed by this adapter */
  readonly deviceLossCount: number
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LOAD_MODEL_TIMEOUT = TIMEOUTS.KOKORO_LOAD
const GENERATE_TIMEOUT = TIMEOUTS.KOKORO_GENERATE

// ---------------------------------------------------------------------------
// Audio Encoding
// ---------------------------------------------------------------------------

/**
 * Encode raw PCM Float32Array samples into a WAV ArrayBuffer.
 * This runs on the main thread — intentionally lightweight (just header + int16 conversion).
 */
function encodeWav(samples: Float32Array, sampleRate: number, numChannels = 1): ArrayBuffer {
  const bitsPerSample = 16
  const bytesPerSample = bitsPerSample / 8
  const dataLength = samples.length * bytesPerSample
  const headerLength = 44
  const buffer = new ArrayBuffer(headerLength + dataLength)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataLength, true)
  writeString(view, 8, 'WAVE')

  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true) // byte rate
  view.setUint16(32, numChannels * bytesPerSample, true) // block align
  view.setUint16(34, bitsPerSample, true)

  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, dataLength, true)

  // Convert Float32 [-1, 1] to Int16
  const output = new Int16Array(buffer, headerLength)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
  }

  return buffer
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

interface KokoroManifest {
  quantization: string
  device: string
}

/**
 * Bind the Eventa invoke clients to a freshly created worker. Returns the
 * load (server-streaming) and generate (unary) callables; the rpc type is
 * inferred so call sites stay aligned with the library's option shapes.
 */
function createKokoroRpc(worker: Worker) {
  const { context } = createContext(worker)
  return {
    load: defineStreamInvoke(context, kokoroLoadEvent),
    generate: defineInvoke(context, kokoroGenerateEvent),
  }
}

type KokoroRpc = ReturnType<typeof createKokoroRpc>

export function createKokoroAdapter(): KokoroAdapter {
  let worker: Worker | null = null
  let rpc: KokoroRpc | null = null
  let state: KokoroAdapter['state'] = 'idle'
  let voices: Voices | null = null
  let restartAttempts = 0
  let allocationToken: AllocationToken | null = null
  let currentModelStatusId: string | null = null
  let errorListener: ((event: ErrorEvent) => void) | null = null

  // NOTICE: Device-loss resilience state. `lastManifest` records the last
  // successful load config so scheduleRestart can reconstruct context if the
  // worker died. `deviceLossCount` tracks WebGPU device-loss events so we
  // can promote to WASM after repeated failures (see DEVICE_LOSS_WASM_THRESHOLD).
  let lastManifest: KokoroManifest | null = null
  let deviceLossCount = 0

  const operationMutex = new Mutex()
  const lifecycleMutex = new Mutex()

  function initializeWorker(): void {
    worker = new Worker(
      new URL('../../../workers/kokoro/worker.ts', import.meta.url),
      { type: 'module' },
    )
    rpc = createKokoroRpc(worker)
    // NOTICE: device-loss telemetry + restart. Eventa already rejects in-flight
    // invokes on a fatal worker error (it sets `worker.onerror`); this native
    // 'error' listener coexists with it and owns the resilience policy the
    // adapter is responsible for, mirroring the pre-Eventa error listener.
    errorListener = (event: ErrorEvent) => handleWorkerError(event)
    worker.addEventListener('error', errorListener)
  }

  function handleWorkerError(event: ErrorEvent | Error): void {
    state = 'error'
    operationMutex.cancel()

    // Record device-loss telemetry before teardown so the coordinator sees it
    // even if the adapter is never used again.
    const error = event instanceof Error ? event : (event as ErrorEvent).error ?? event
    const code = classifyError(error)
    if (code === 'DEVICE_LOST') {
      deviceLossCount++
      getGPUCoordinator().recordDeviceLoss({
        modelId: currentModelStatusId ?? MODEL_NAMES.KOKORO,
        reason: classifyDeviceLossReason(error),
        occurredAt: Date.now(),
      })
    }

    destroyWorker()
    scheduleRestart()
  }

  function destroyWorker(): void {
    if (worker) {
      if (errorListener)
        worker.removeEventListener('error', errorListener)
      errorListener = null
      worker.terminate()
      worker = null
    }
    rpc = null
  }

  function scheduleRestart(): void {
    if (restartAttempts >= MAX_RESTARTS) {
      console.error(
        `[KokoroAdapter] Max restart attempts (${MAX_RESTARTS}) reached.`,
      )
      // NOTICE: Transition to 'terminated' so getKokoroAdapter() can detect
      // the dead singleton and create a fresh adapter on next access.
      state = 'terminated'
      return
    }

    restartAttempts++
    const delay = RESTART_DELAY_MS * restartAttempts

    console.warn(
      `[KokoroAdapter] Restarting in ${delay}ms `
      + `(attempt ${restartAttempts}/${MAX_RESTARTS})`,
    )

    setTimeout(() => {
      ensureStarted().catch((err) => {
        console.error('[KokoroAdapter] Restart failed:', err)
      })
    }, delay)
  }

  function onSuccess(): void {
    restartAttempts = 0
  }

  async function ensureStarted(): Promise<void> {
    await lifecycleMutex.runExclusive(async () => {
      if (!worker) {
        initializeWorker()
        state = 'idle'
      }
    })
  }

  // -- Public API -----------------------------------------------------------

  async function loadModel(
    quantization: string,
    device: string,
    options?: {
      onProgress?: (p: ProgressPayload) => void
      signal?: AbortSignal
    },
  ): Promise<Voices> {
    // NOTICE: Proactive WASM promotion. If this adapter has suffered repeated
    // WebGPU device-loss events, webgpu is unreliable on this device and we
    // should not keep retrying. The worker's per-load dtype/device fallback
    // chain handles transient failures; this guard handles persistent ones.
    let effectiveDevice = device
    if (
      device === 'webgpu'
      && deviceLossCount >= DEVICE_LOSS_WASM_THRESHOLD
    ) {
      console.warn(
        `[KokoroAdapter] ${deviceLossCount} device-loss events recorded, `
        + `promoting load from webgpu to wasm.`,
      )
      effectiveDevice = 'wasm'
    }
    throwIfAborted(options?.signal)
    await ensureStarted()

    return defaultPerfTracer.withMeasure('inference', 'kokoro-load-model', () => operationMutex.runExclusive(async () => {
      throwIfAborted(options?.signal)
      state = 'loading'
      const modelStatusId = `kokoro-${quantization}`

      // Clear previous model status when switching models
      if (currentModelStatusId && currentModelStatusId !== modelStatusId)
        removeInferenceStatus(currentModelStatusId)
      currentModelStatusId = modelStatusId

      updateInferenceStatus(modelStatusId, { state: 'downloading', device: effectiveDevice as any })

      // Use the global load queue to serialize model loads across all adapters
      return getLoadQueue().enqueue(modelStatusId, LOAD_PRIORITY.TTS, async () => {
        throwIfAborted(options?.signal)
        if (!rpc)
          throw new Error('Kokoro worker not initialized')

        const stream = rpc.load(
          { device: effectiveDevice as any, dtype: quantization },
          // NOTICE:
          // `raw: {}` satisfies @moeru/eventa@1.0.0-beta.5's stream-invoke
          // options type, which over-includes the inbound emit metadata (`raw`)
          // as a required caller option (asymmetric with unary `defineInvoke`).
          // It is ignored on the send path; only `signal` is consumed.
          // Removal condition: drop when the upstream stream-invoke option type
          // stops requiring `raw` (track @moeru/eventa releases past 1.0.0-beta.5).
          { signal: signalWithTimeout(options?.signal, LOAD_MODEL_TIMEOUT), raw: {} },
        )

        const info = await consumeLoadStream(stream, (progress) => {
          // Update reactive inference status
          updateInferenceStatus(modelStatusId, { progress })
          options?.onProgress?.(progress)
        }).catch((error) => {
          // Normalize caller-driven aborts to InferenceAbortError so the outer
          // catch (and callers) see name === 'AbortError'.
          if (options?.signal?.aborted)
            throw new InferenceAbortError(typeof options.signal.reason === 'string' ? options.signal.reason : undefined)
          throw error
        })

        voices = (info.metadata?.voices as Voices) ?? null

        // Track GPU memory allocation
        const coordinator = getGPUCoordinator()
        if (allocationToken)
          coordinator.release(allocationToken)
        const estimateKey = `kokoro-${quantization}`
        const estimated = MODEL_VRAM_ESTIMATES[estimateKey] ?? 165 * 1024 * 1024
        allocationToken = coordinator.requestAllocation(`kokoro-${quantization}`, estimated)

        // Record manifest so consumers can inspect how the adapter resolved
        // device selection after fallback / WASM promotion.
        lastManifest = { quantization, device: info.device }

        state = 'ready'
        updateInferenceStatus(modelStatusId, { state: 'ready', device: info.device as any })
        onSuccess()
        if (!voices)
          throw new Error('Kokoro worker did not return voice metadata')
        return voices
      }, { signal: options?.signal })
    }), { quantization, device: effectiveDevice }).catch((error) => {
      // Don't route AbortError through handleWorkerError — cancellation is
      // not a worker failure and shouldn't trigger restart logic.
      if ((error as Error)?.name === 'AbortError')
        throw error
      handleWorkerError(error instanceof Error ? error : new Error(String(error)))
      throw error
    })
  }

  async function generate(
    text: string,
    voice: VoiceKey,
    options?: { signal?: AbortSignal },
  ): Promise<ArrayBuffer> {
    throwIfAborted(options?.signal)
    const notReadyError = new Error('Model not loaded. Call loadModel() first.')

    return defaultPerfTracer.withMeasure('inference', 'kokoro-generate', () => operationMutex.runExclusive(async () => {
      throwIfAborted(options?.signal)
      if (!worker || !rpc || state !== 'ready')
        throw notReadyError

      // Update LRU timestamp for memory pressure tracking
      if (allocationToken)
        getGPUCoordinator().touch(allocationToken.modelId)

      state = 'running'

      const result = await rpc.generate(
        { text, voice },
        { signal: signalWithTimeout(options?.signal, GENERATE_TIMEOUT) },
      ).catch((error) => {
        if (options?.signal?.aborted)
          throw new InferenceAbortError(typeof options.signal.reason === 'string' ? options.signal.reason : undefined)
        throw error
      })

      state = 'ready'
      onSuccess()
      return encodeWav(result.samples, result.samplingRate)
    }), { text: text.slice(0, 50), voice }).catch((error) => {
      if (error === notReadyError)
        throw error

      handleWorkerError(error instanceof Error ? error : new Error(String(error)))
      throw error
    })
  }

  function getVoices(): Voices {
    if (!voices)
      throw new Error('Model not loaded. Call loadModel() first.')
    return voices
  }

  function terminateAdapter(): void {
    operationMutex.cancel()
    destroyWorker()
    if (allocationToken) {
      removeInferenceStatus(allocationToken.modelId)
      getGPUCoordinator().release(allocationToken)
      allocationToken = null
    }
    voices = null
    state = 'terminated'
  }

  return {
    loadModel,
    generate,
    getVoices,
    terminate: terminateAdapter,
    get state() { return state },
    get manifest() { return lastManifest },
    get deviceLossCount() { return deviceLossCount },
  }
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

let globalAdapter: KokoroAdapter | null = null
const singletonMutex = new Mutex()

/**
 * Get the global Kokoro adapter instance.
 * Creates and starts the worker on first call.
 * Automatically re-creates the adapter if it has entered a terminal state
 * ('terminated' or 'error' after max restarts exhausted).
 */
export async function getKokoroAdapter(): Promise<KokoroAdapter> {
  return singletonMutex.runExclusive(async () => {
    if (
      !globalAdapter
      || globalAdapter.state === 'terminated'
      || globalAdapter.state === 'error'
    ) {
      globalAdapter = createKokoroAdapter()
    }
    return globalAdapter
  })
}
