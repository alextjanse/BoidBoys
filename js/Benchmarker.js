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

    this.warmUpEndsAt = 0;
    this.recordEndsAt = 0;

    this.simFrameSamples = [];
    this.renderFrameSamples = [];
    this.onEscHandler = null;

    // Store current engine settings to ensure completeBenchmark has data
    this.currentEngineState = null;
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
    this.unregisterCancelHotkey();
    this.state = BenchmarkState.IDLE;
    this.warmUpEndsAt = 0;
    this.recordEndsAt = 0;
    this.currentEngineState = null; // Clean up reference
    if (this.onCompleteCallback) this.onCompleteCallback();
  }

  cancelBenchmark(reason = 'Benchmark canceled.')
  {
    if (this.state !== BenchmarkState.WARMING_UP && this.state !== BenchmarkState.RECORDING) return;
    console.log(reason);
    this.finalizeRun();
  }

  start(engineState = null)
  {
    if (this.state !== BenchmarkState.IDLE && this.state !== BenchmarkState.COMPLETED) {
      console.warn("Benchmark already in progress.");
      return;
    }

    this.currentEngineState = engineState;
    this.frameTimes = [];
    this.simFrameSamples = [];
    this.renderFrameSamples = [];

    const now = performance.now();
    this.state = BenchmarkState.WARMING_UP;
    this.warmUpEndsAt = now + this.WARM_UP_MS;
    this.recordEndsAt = 0; // Will be set when recording actually starts

    this.registerCancelHotkey();
    this.onResetCallback();

    console.log("Benchmark: WARMING UP (10s)...");
  }

  /**
   * Main entry point for every frame. 
   * Handles state transitions based on time to ensure sync.
   */
  recordFrame(now = null)
  {
    const timestamp = Number.isFinite(now) ? now : performance.now();

    // PHASE 1: Handle transition from Warmup to Recording
    if (this.state === BenchmarkState.WARMING_UP) {
      if (timestamp >= this.warmUpEndsAt) {
        this.state = BenchmarkState.RECORDING;
        this.recordEndsAt = timestamp + this.RECORD_MS;
        this.lastFrameTime = timestamp; // RESET HERE to prevent massive first delta
        console.log("Benchmark: RECORDING (10s)...");
      }
      return;
    }

    // PHASE 2: Handle Recording logic
    if (this.state === BenchmarkState.RECORDING) {
      // Check for completion
      if (timestamp >= this.recordEndsAt) {
        this.completeBenchmark(this.currentEngineState);
        return;
      }

      const delta = timestamp - this.lastFrameTime;
      if (delta > 0) {
        this.frameTimes.push(delta);
      }
      this.lastFrameTime = timestamp;
    }
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
    const remaining = (endTime) => Math.max(0, (endTime - now) / 1000).toFixed(1);

    switch (this.state) {
      case BenchmarkState.WARMING_UP:
        return { visible: true, phaseClass: 'warming', status: 'Warming Up', detail: `${remaining(this.warmUpEndsAt)}s remaining` };
      case BenchmarkState.RECORDING:
        return { visible: true, phaseClass: 'recording', status: 'Recording', detail: `${remaining(this.recordEndsAt)}s remaining` };
      case BenchmarkState.COMPLETED:
        return { visible: true, phaseClass: 'completed', status: 'Completed', detail: 'Preparing export...' };
      default:
        return { visible: false, phaseClass: '', status: 'Idle', detail: '' };
    }
  }

  async completeBenchmark(engineState)
  {
    // Prevent double-calls if recordFrame is called while async export is running
    if (this.state === BenchmarkState.COMPLETED) return;

    this.state = BenchmarkState.COMPLETED;
    this.unregisterCancelHotkey();

    console.log(`Benchmark COMPLETED. Captured ${this.frameTimes.length} frames.`);

    if (this.frameTimes.length === 0) {
      console.warn('Benchmark completed without captured frame times.');
      this.finalizeRun();
      return;
    }

    const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const safeSettings = engineState || this.currentEngineState || {
      boidCount: 0,
      projectName: 'Unknown',
      version: '1.0.0',
    };

    try {
      await this.reportExporter.exportPerformanceReport({
        frameTimes: this.frameTimes,
        settings: safeSettings,
        hardware: {
          cpu: '',
          gpu: '',
          os: navigator.platform,
        },
        metrics: {
          avgRenderTime: avg(this.renderFrameSamples),
          avgSimTime: avg(this.simFrameSamples),
          avgFPS: 1000 / avg(this.frameTimes)
        },
      });
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      this.finalizeRun();
      console.log('Benchmark flow finished.');
    }
  }
}