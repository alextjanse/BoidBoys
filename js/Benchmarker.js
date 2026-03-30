export const BenchmarkState = {
  IDLE: 0,
  WARMING_UP: 1,
  RECORDING: 2,
  COMPLETED: 3
};

export class BoidBenchmarker
{
  constructor(onResetCallback, onCompleteCallback = null, reportExporter)
  {
    this.state = BenchmarkState.IDLE;
    this.frameTimes = [];
    this.lastFrameTime = 0;
    this.onResetCallback = onResetCallback;
    this.onCompleteCallback = onCompleteCallback;
    this.reportExporter = reportExporter;

    this.WARM_UP_MS = 10000;
    this.RECORD_MS = 10000;

    this.warmUpTimeout = null;
    this.recordTimeout = null;
    this.warmUpEndsAt = 0;
    this.recordEndsAt = 0;

    this.simFrameSamples = [];
    this.renderFrameSamples = [];
    this.onEscHandler = null;
  }

  registerCancelHotkey()
  {
    if (typeof window === 'undefined' || this.onEscHandler) return;
    this.onEscHandler = (event) =>
    {
      if (event.key !== 'Escape') return;
      if (this.state !== BenchmarkState.WARMING_UP && this.state !== BenchmarkState.RECORDING) return;
      event.preventDefault();
      this.cancelBenchmark('Benchmark canceled by user (Esc).');
    };
    window.addEventListener('keydown', this.onEscHandler);
  }

  unregisterCancelHotkey()
  {
    if (typeof window === 'undefined' || !this.onEscHandler) return;
    window.removeEventListener('keydown', this.onEscHandler);
    this.onEscHandler = null;
  }

  finalizeRun()
  {
    if (this.warmUpTimeout) {
      clearTimeout(this.warmUpTimeout);
      this.warmUpTimeout = null;
    }
    if (this.recordTimeout) {
      clearTimeout(this.recordTimeout);
      this.recordTimeout = null;
    }

    this.unregisterCancelHotkey();
    this.state = BenchmarkState.IDLE;
    this.warmUpEndsAt = 0;
    this.recordEndsAt = 0;
    if (this.onCompleteCallback) this.onCompleteCallback();
  }

  cancelBenchmark(reason = 'Benchmark canceled.')
  {
    if (this.state !== BenchmarkState.WARMING_UP && this.state !== BenchmarkState.RECORDING) return;
    this.finalizeRun();
    console.log(reason);
  }

  start()
  {
    if (this.state !== BenchmarkState.IDLE && this.state !== BenchmarkState.COMPLETED) {
      console.warn("Benchmark already in progress.");
      return;
    }

    this.frameTimes = [];
    this.simFrameSamples = [];
    this.renderFrameSamples = [];
    this.lastFrameTime = 0;
    this.state = BenchmarkState.WARMING_UP;
    this.warmUpEndsAt = performance.now() + this.WARM_UP_MS;
    this.recordEndsAt = 0;
    this.registerCancelHotkey();

    this.onResetCallback();
    console.log("Benchmark: WARMING UP (10s)...");

    this.warmUpTimeout = setTimeout(() =>
    {
      this.state = BenchmarkState.RECORDING;
      this.lastFrameTime = performance.now();
      this.recordEndsAt = performance.now() + this.RECORD_MS;
      console.log("Benchmark: RECORDING (10s)...");

      this.recordTimeout = setTimeout(() =>
      {
        this.completeBenchmark();
      }, this.RECORD_MS);

    }, this.WARM_UP_MS);
  }

  recordFrame(now = null, gpuTimestampMs = null)
  {
    const fallbackNow = (typeof performance !== 'undefined' && typeof performance.now === 'function')
      ? performance.now()
      : Date.now();
    const timestamp = Number.isFinite(gpuTimestampMs)
      ? gpuTimestampMs
      : (Number.isFinite(now) ? now : fallbackNow);

    if (this.state !== BenchmarkState.RECORDING) {
      this.lastFrameTime = timestamp;
      return;
    }

    const delta = timestamp - this.lastFrameTime;
    if (delta > 0) {
      this.frameTimes.push(delta);
    }
    this.lastFrameTime = timestamp;
  }

  recordSimulationSample(simulationMs)
  {
    if (this.state === BenchmarkState.RECORDING && Number.isFinite(simulationMs)) {
      this.simFrameSamples.push(simulationMs);
    }
  }

  recordRenderSample(renderMs)
  {
    if (this.state === BenchmarkState.RECORDING && Number.isFinite(renderMs)) {
      this.renderFrameSamples.push(renderMs);
    }
  }

  getStatus(now = performance.now())
  {
    if (this.state === BenchmarkState.WARMING_UP) {
      return {
        visible: true,
        phaseClass: 'warming',
        status: 'Warming Up',
        detail: `${Math.max(0, (this.warmUpEndsAt - now) / 1000).toFixed(1)}s remaining`,
      };
    }

    if (this.state === BenchmarkState.RECORDING) {
      return {
        visible: true,
        phaseClass: 'recording',
        status: 'Recording',
        detail: `${Math.max(0, (this.recordEndsAt - now) / 1000).toFixed(1)}s remaining`,
      };
    }

    if (this.state === BenchmarkState.COMPLETED) {
      return {
        visible: true,
        phaseClass: 'completed',
        status: 'Completed',
        detail: 'Preparing export...',
      };
    }

    return {
      visible: false,
      phaseClass: '',
      status: 'Idle',
      detail: '',
    };
  }

  async completeBenchmark(engineState)
  {
    this.state = BenchmarkState.COMPLETED;
    this.unregisterCancelHotkey();
    console.log(`Benchmark COMPLETED. Captured ${this.frameTimes.length} frames.`);

    if (this.frameTimes.length === 0) {
      console.warn('Benchmark completed without captured frame times.');
      this.finalizeRun();
      return;
    }

    const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    try {
      await this.reportExporter.exportPerformanceReport({
        frameTimes: this.frameTimes,
        settings: engineState,
        hardware: {
          cpu: '',
          gpu: '',
          os: '',
        },
        metrics: {
          avgRenderTime: avg(this.renderFrameSamples),
          avgSimTime: avg(this.simFrameSamples),
        },
      });
    } finally {
      this.finalizeRun();
      console.log('Benchmark flow finished. Ready for next run.');
    }
  }
}
