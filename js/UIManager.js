import { ParamsIndex } from './constants.js';

export class UIManager
{
  constructor(callbacks)
  {
    this.callbacks = callbacks;
    this.isSimulationRunning = true;
    this.isColorsOn = false;
    this.isBgOn = false;
    this.wasSettingsPanelOpenBeforeBenchmark = null;

    this.boidCountInput = document.getElementById('boid-count');
    this.boidDensityInput = document.getElementById('boid-density');
    this.boidCountUpdateTimer = null;
  }

  init(initialState)
  {
    this.populateInputs(initialState);
    this.setupEventListeners();
    this.updateStartPauseButton();
    this.initTooltips();
  }

  populateInputs(state)
  {
    const p = (v) => parseFloat(v.toPrecision(6));

    document.getElementById('boid-count').value = state.boidCount;
    document.getElementById('boid-density').value = state.boidDensity.toFixed(6);
    document.getElementById('separation').value = p(state.params[ParamsIndex.SEPARATION_DIST]);
    document.getElementById('align').value = p(state.params[ParamsIndex.ALIGN_DIST]);
    document.getElementById('cohesion').value = p(state.params[ParamsIndex.COHESION_DIST]);
    document.getElementById('max_speed').value = p(state.params[ParamsIndex.MAX_SPEED]);
    document.getElementById('max_force').value = p(state.params[ParamsIndex.MAX_FORCE]);
    document.getElementById('sep_weight').value = p(state.params[ParamsIndex.SEPARATION_WEIGHT]);
    document.getElementById('align_weight').value = p(state.params[ParamsIndex.ALIGNMENT_WEIGHT]);
    document.getElementById('coh_weight').value = p(state.params[ParamsIndex.COHESION_WEIGHT]);
    document.getElementById('margin').value = p(state.params[ParamsIndex.MARGIN]);
    document.getElementById('turn_factor').value = p(state.params[ParamsIndex.TURN_FACTOR]);
    // Vision angle displayed in degrees for the UI
    document.getElementById('vision_angle').value = p(state.params[ParamsIndex.VISION_ANGLE] * 180.0 / Math.PI);
  }

  setupEventListeners()
  {
    const applyBoidCountFromInput = () =>
    {
      const n = parseInt(this.boidCountInput.value, 10);
      if (!isNaN(n) && n > 0 && n !== this.callbacks.getBoidCount()) {
        this.callbacks.onRecreateBoids(n, parseFloat(this.boidDensityInput.value));
      }
    };

    this.boidCountInput.addEventListener('change', applyBoidCountFromInput);
    this.boidCountInput.addEventListener('input', () =>
    {
      if (this.boidCountUpdateTimer) clearTimeout(this.boidCountUpdateTimer);
      this.boidCountUpdateTimer = setTimeout(applyBoidCountFromInput, 250);
    });

    this.boidDensityInput.addEventListener('input', e =>
    {
      const d = parseFloat(e.target.value);
      if (!isNaN(d) && d > 0) {
        this.callbacks.onRecreateBoids(parseInt(this.boidCountInput.value, 10), d);
      }
    });

    const inputs = ['separation', 'align', 'cohesion', 'max_speed', 'max_force',
      'sep_weight', 'align_weight', 'coh_weight', 'margin', 'turn_factor', 'vision_angle'];

    inputs.forEach(id =>
    {
      document.getElementById(id).addEventListener('input', () =>
      {
        this.callbacks.onUpdateUniforms(this.getUniformValues());
      });
    });

    document.getElementById('toggle-panel').addEventListener('click', () =>
    {
      const body = document.getElementById('settings-body');
      const bs = bootstrap.Collapse.getOrCreateInstance(body);
      bs.toggle();
    });

    document.getElementById('start-pause-btn').addEventListener('click', () =>
    {
      this.isSimulationRunning = !this.isSimulationRunning;
      this.updateStartPauseButton();
      this.callbacks.onSimulationToggle(this.isSimulationRunning);
    });

    document.getElementById('restart-btn').addEventListener('click', () =>
    {
      const inputCount = parseInt(this.boidCountInput.value, 10);
      let inputDensity = parseFloat(this.boidDensityInput.value);
      if (isNaN(inputDensity) || inputDensity <= 0) inputDensity = this.callbacks.getBoidDensity();
      if (isNaN(inputCount) || inputCount <= 0) inputCount = this.callbacks.getBoidCount();

      this.callbacks.onRecreateBoids(inputCount, inputDensity);

      this.isSimulationRunning = true;
      this.updateStartPauseButton();
      this.callbacks.onSimulationToggle(true);
    });

    document.getElementById('reset-btn').addEventListener('click', () =>
    {
      this.callbacks.onResetSimulation();
    });

    document.getElementById('benchmark-btn').addEventListener('click', () =>
    {
      this.callbacks.onBenchmarkStart();
    });

    document.getElementById('import-report-btn').addEventListener('click', () =>
    {
      const input = document.getElementById('benchmark-json-input');
      if (input) input.click();
    });

    document.getElementById('benchmark-json-input').addEventListener('change', async (event) =>
    {
      const files = event.target.files ? Array.from(event.target.files) : [];
      if (files.length === 0) return;

      try {
        await this.callbacks.onImportReport(files);
      } catch (error) {
        console.error('Failed to import benchmark TeX:', error);
        window.alert('Failed to import benchmark TeX. See console for details.');
      } finally {
        event.target.value = '';
      }
    });

    document.getElementById('toggle-colors-btn').addEventListener('click', () =>
    {
      this.isColorsOn = !this.isColorsOn;
      this.updateColorButton();
      this.callbacks.onColorToggle(this.isColorsOn);

      // Restart with current settings
      const inputCount = parseInt(this.boidCountInput.value, 10);
      let inputDensity = parseFloat(this.boidDensityInput.value);
      if (isNaN(inputDensity) || inputDensity <= 0) inputDensity = this.callbacks.getBoidDensity();
      if (isNaN(inputCount) || inputCount <= 0) inputCount = this.callbacks.getBoidCount();

      this.callbacks.onRecreateBoids(inputCount, inputDensity);

      this.isSimulationRunning = true;
      this.updateStartPauseButton();
      this.callbacks.onSimulationToggle(true);
    }
    );

    document.getElementById('toggle-bg-btn').addEventListener('click', () =>
    {
      this.isBgOn = !this.isBgOn;
      // this.updateBgButton();
      this.callbacks.onBgToggle(this.isBgOn);
    });
  }

  getUniformValues()
  {
    return {
      separation: parseFloat(document.getElementById('separation').value),
      align: parseFloat(document.getElementById('align').value),
      cohesion: parseFloat(document.getElementById('cohesion').value),
      max_speed: parseFloat(document.getElementById('max_speed').value),
      max_force: parseFloat(document.getElementById('max_force').value),
      sep_weight: parseFloat(document.getElementById('sep_weight').value),
      align_weight: parseFloat(document.getElementById('align_weight').value),
      coh_weight: parseFloat(document.getElementById('coh_weight').value),
      margin: parseFloat(document.getElementById('margin').value),
      turn_factor: parseFloat(document.getElementById('turn_factor').value),
      vision_angle: parseFloat(document.getElementById('vision_angle').value)
    };
  }

  updateStartPauseButton()
  {
    const btn = document.getElementById('start-pause-btn');
    const icon = document.getElementById('start-icon');
    if (this.isSimulationRunning) {
      icon.className = 'bi bi-pause-fill';
      btn.classList.add('btn-success');
      btn.classList.remove('btn-warning');
    } else {
      icon.className = 'bi bi-play-fill';
      btn.classList.add('btn-warning');
      btn.classList.remove('btn-success');
    }
  }

  updateColorButton()
  {
    const btn = document.getElementById('toggle-colors-btn');
    const icon = document.getElementById('color-icon');
    if (this.isColorsOn) {
      btn.classList.add('btn-color-on');
      btn.classList.remove('btn-color-off');
      icon.className = 'bi bi-brush-fill';
    } else {
      btn.classList.remove('btn-color-on');
      btn.classList.add('btn-color-off');
      icon.className = 'bi bi-brush';
    }
  }

  initTooltips()
  {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el =>
    {
      bootstrap.Tooltip.getOrCreateInstance(el, { trigger: 'hover' });
    });
  }

  updateFPS(fps, avgSim, avgRen)
  {
    document.getElementById('info-fps').innerText = `FPS: ${fps.toFixed(1)}`;
    if (avgSim !== null) {
      document.getElementById('info-step').innerText = `Sim: ${avgSim.toFixed(2)} ms`;
    }
    if (avgRen !== null) {
      document.getElementById('info-gpu').innerText = `Render: ${avgRen.toFixed(2)} ms`;
    }
  }

  updateInfo(boidCount, numCells)
  {
    document.getElementById('info-boids').innerText = `Boids: ${boidCount}`;
    document.getElementById('gpu-status').innerText = `Cells: ${numCells}`;
  }

  collapseSettingsPanelForBenchmark()
  {
    const body = document.getElementById('settings-body');
    if (!body) return;

    this.wasSettingsPanelOpenBeforeBenchmark = body.classList.contains('show');
    const bs = bootstrap.Collapse.getOrCreateInstance(body, { toggle: false });
    bs.hide();
  }

  restoreSettingsPanelAfterBenchmark()
  {
    if (this.wasSettingsPanelOpenBeforeBenchmark === null) return;

    const body = document.getElementById('settings-body');
    if (!body) {
      this.wasSettingsPanelOpenBeforeBenchmark = null;
      return;
    }

    const bs = bootstrap.Collapse.getOrCreateInstance(body, { toggle: false });
    if (this.wasSettingsPanelOpenBeforeBenchmark) {
      bs.show();
    } else {
      bs.hide();
    }

    this.wasSettingsPanelOpenBeforeBenchmark = null;
  }

  updateBenchmarkHUD(state)
  {
    const hud = document.getElementById('benchmark-hud');
    if (!hud) return;

    const statusEl = hud.querySelector('.status');
    const countdownEl = hud.querySelector('.countdown');
    if (!statusEl || !countdownEl) return;

    hud.classList.remove('warming', 'recording', 'completed');

    if (!state.visible) {
      hud.classList.remove('show');
      return;
    }

    hud.classList.add('show');
    if (state.phaseClass) hud.classList.add(state.phaseClass);
    statusEl.textContent = state.status;
    countdownEl.textContent = state.detail;
  }
}
