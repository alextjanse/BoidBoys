import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PerformanceReportExporter } from './performance-report.ts';

// #region Configuration
let boidCount = 100;
const WORKGROUP_SIZE = 256;
let SIMULATION_SIZE = { x: 800, y: 800, z: 800 };
let boidDensity = 0.000050;
const BASE_SIMULATION_SIZE = { x: 800, y: 800, z: 800 };

// #endregion

// #region Grid config (derived from behavior distances and world size)
let cellSize = 50;
let gridDim = { x: 1, y: 1, z: 1 };
let numCells = 1;

// #endregion

// #region Uniform buffer (20 floats = 80 bytes)
const paramsArray = new Float32Array(20);
// Layout:
// [0] separation_dist    [1] align_dist
// [2] cohesion_dist      [3] max_speed
// [4] max_force          [5] separation_weight
// [6] alignment_weight   [7] cohesion_weight
// [8] margin             [9] turn_factor
// [10] cell_size         [11] _padding
// [12-15] world_max (vec4)
// [16-19] grid_dim (vec4, .w = numCells)

// #endregion

// #region GPU and rendering globals
let scene, camera, renderer, boidInstancedMesh, controls;
let gpuDevice;
let useGPU = false;
let isMapping = false;
let isSimulationRunning = true;
let wasSettingsPanelOpenBeforeBenchmark = null;

// Buffers
let boidBuffer, cellHeadBuffer, boidNextBuffer;
let matrixBuffer, matrixStagingBuffer;
let uniformBuffer;

// Compute pipelines (one per shader entry point)
let clearCellsPipeline, hashInsertPipeline, updateBoidsPipeline, computeMatricesPipeline;

// Bind group layout and bind group
let bindGroupLayout;
let bindGroup;

// Performance instrumentation
let lastFrameTime = 0;
let frameTimes = [];
let lastFPSUpdate = 0;
const FPS_SAMPLE_SIZE = 60;
const FPS_UPDATE_INTERVAL = 500;
let simTimes = [];
let renderTimes = [];
const reportExporter = new PerformanceReportExporter();

// #endregion

// #region Helper functions

// Return spawn bounds for a given world size
function getSpawnBounds(worldSize)
{
  return {
    min: { x: 0, y: 0, z: 0 },
    max: { x: worldSize.x, y: worldSize.y, z: worldSize.z }
  };
}

// Calculate simulation world size based on boid count and density
function calculateSimulationSize(count, density)
{
  const baseVolume = BASE_SIMULATION_SIZE.x * BASE_SIMULATION_SIZE.y * BASE_SIMULATION_SIZE.z;
  const requiredVolume = count / density;
  const scaleFactor = Math.cbrt(requiredVolume / baseVolume);
  return {
    x: BASE_SIMULATION_SIZE.x * scaleFactor,
    y: BASE_SIMULATION_SIZE.y * scaleFactor,
    z: BASE_SIMULATION_SIZE.z * scaleFactor
  };
}

// Compute grid dimensions from current simulation size and cell size
function calculateGridDimensions()
{
  // Cell size = max of behavior distances (ensures correctness)
  cellSize = Math.min(paramsArray[0], paramsArray[1], paramsArray[2], 50);
  gridDim.x = Math.max(1, Math.ceil(SIMULATION_SIZE.x / cellSize));
  gridDim.y = Math.max(1, Math.ceil(SIMULATION_SIZE.y / cellSize));
  gridDim.z = Math.max(1, Math.ceil(SIMULATION_SIZE.z / cellSize));
  numCells = gridDim.x * gridDim.y * gridDim.z;
}

// Update or recreate the visual bounding box for the simulation
function updateVisualBounds()
{
  const oldBox = scene.getObjectByName('boid-bounds');
  if (oldBox) scene.remove(oldBox);

  const boxGeom = new THREE.BoxGeometry(SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z);
  const edges = new THREE.EdgesGeometry(boxGeom);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x444444 }));
  line.name = 'boid-bounds';
  line.position.set(SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2);
  scene.add(line);

  if (controls) {
    controls.target.set(SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2);
  }
}

// #endregion

// #region Params management

// Reset simulation parameters to sensible default values
function resetParamsToDefaults()
{
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);

  // Set behavior distances BEFORE calculating grid (grid depends on these)
  paramsArray[0] = 25.0;  // separation_dist
  paramsArray[1] = 50.0;  // align_dist
  paramsArray[2] = 50.0;  // cohesion_dist

  calculateGridDimensions();
  paramsArray[3] = 5.0;   // max_speed
  paramsArray[4] = 0.1;   // max_force
  paramsArray[5] = 1.5;   // separation_weight
  paramsArray[6] = 1.0;   // alignment_weight
  paramsArray[7] = 0.5;   // cohesion_weight
  paramsArray[8] = 100.0; // margin
  paramsArray[9] = 0.2;   // turn_factor
  paramsArray[10] = cellSize;
  paramsArray[11] = 0.0; // padding
  paramsArray[12] = SIMULATION_SIZE.x;
  paramsArray[13] = SIMULATION_SIZE.y;
  paramsArray[14] = SIMULATION_SIZE.z;
  paramsArray[15] = 0.0;
  paramsArray[16] = gridDim.x;
  paramsArray[17] = gridDim.y;
  paramsArray[18] = gridDim.z;
  paramsArray[19] = numCells;
}

// Write the current params array to the GPU uniform buffer
function syncParamsToGPU()
{
  if (!gpuDevice || !uniformBuffer) return;
  // Always refresh grid & dynamic fields
  calculateGridDimensions();
  paramsArray[10] = cellSize;
  paramsArray[11] = 0.0;
  paramsArray[12] = SIMULATION_SIZE.x;
  paramsArray[13] = SIMULATION_SIZE.y;
  paramsArray[14] = SIMULATION_SIZE.z;
  paramsArray[16] = gridDim.x;
  paramsArray[17] = gridDim.y;
  paramsArray[18] = gridDim.z;
  paramsArray[19] = numCells;
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
}

resetParamsToDefaults();

// #endregion

// #region GPU buffer creation

// Initialize the storage buffer containing boid positions and velocities
function initBoidBuffers(count)
{
  const boidData = new Float32Array(count * 8);
  const spawnBounds = getSpawnBounds(SIMULATION_SIZE);
  for (let i = 0; i < count; i++) {
    boidData[i * 8] = spawnBounds.min.x + Math.random() * (spawnBounds.max.x - spawnBounds.min.x);
    boidData[i * 8 + 1] = spawnBounds.min.y + Math.random() * (spawnBounds.max.y - spawnBounds.min.y);
    boidData[i * 8 + 2] = spawnBounds.min.z + Math.random() * (spawnBounds.max.z - spawnBounds.min.z);
    boidData[i * 8 + 3] = 1.0;
    boidData[i * 8 + 4] = (Math.random() - 0.5) * 4;
    boidData[i * 8 + 5] = (Math.random() - 0.5) * 4;
    boidData[i * 8 + 6] = (Math.random() - 0.5) * 4;
    boidData[i * 8 + 7] = 0.0;
  }

  boidBuffer = gpuDevice.createBuffer({
    size: boidData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(boidBuffer.getMappedRange()).set(boidData);
  boidBuffer.unmap();
}

// Initialize buffers used by the spatial hash (cell heads and next indices)
function initSpatialHashBuffers()
{
  calculateGridDimensions();

  cellHeadBuffer = gpuDevice.createBuffer({
    size: Math.max(4, numCells * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  boidNextBuffer = gpuDevice.createBuffer({
    size: Math.max(4, boidCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

// Create buffers for instance matrices and a staging buffer for readback
function initMatrixBuffers()
{
  const matSize = boidCount * 16 * 4; // 16 floats per mat4, 4 bytes per float

  matrixBuffer = gpuDevice.createBuffer({
    size: Math.max(4, matSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  matrixStagingBuffer = gpuDevice.createBuffer({
    size: Math.max(4, matSize),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

// #endregion

// #region Bind groups

// Create bind groups for compute pipelines
function createBindGroups()
{
  bindGroup = gpuDevice.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: cellHeadBuffer } },
      { binding: 3, resource: { buffer: boidNextBuffer } },
      { binding: 4, resource: { buffer: matrixBuffer } },
    ]
  });
}

// #endregion

// #region WebGPU initialization

// Initialize WebGPU device, shader modules, pipelines and buffers
async function initWebGPU()
{
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    document.getElementById('info-app').innerText = "WebGPU not supported";
    return false;
  }

  gpuDevice = await adapter.requestDevice();

  const shaderCode = await fetch('compute-shader.wgsl').then(r => r.text());
  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

  // Create GPU buffers
  initBoidBuffers(boidCount);
  initSpatialHashBuffers();
  initMatrixBuffers();

  // Uniform buffer
  uniformBuffer = gpuDevice.createBuffer({
    size: paramsArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  syncParamsToGPU();

  // Bind group layout — 5 bindings, all in group 0
  bindGroupLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]
  });

  const pipelineLayout = gpuDevice.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // Create compute pipelines for each entry point
  const makePipeline = (entryPoint) => gpuDevice.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint }
  });

  clearCellsPipeline = makePipeline('clear_cells');
  hashInsertPipeline = makePipeline('hash_insert');
  updateBoidsPipeline = makePipeline('update_boids');
  computeMatricesPipeline = makePipeline('compute_matrices');

  // Create bind groups
  createBindGroups();

  // Initialize UI after pipelines are ready
  initUI();

  useGPU = true;
  document.getElementById('info-app').innerText = "WebGPU Running";
  return true;
}

// #endregion

// #region Three.js initialization

// Initialize Three.js scene, camera, renderer and controls
function initThree()
{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000005);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 20000);
  const _initHalf = SIMULATION_SIZE.x / 2;
  const _initD = SIMULATION_SIZE.x;
  camera.position.set(_initHalf + _initD, _initHalf + _initD, _initHalf + _initD);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  const container = document.getElementById('canvas-container');
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Visual bounds
  const boxGeom = new THREE.BoxGeometry(SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z);
  const edges = new THREE.EdgesGeometry(boxGeom);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x444444 }));
  line.name = 'boid-bounds';
  line.position.set(SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2);
  scene.add(line);

  createInstancedMesh();
  scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0xffffff, 0.3));

  window.addEventListener('resize', () =>
  {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

// Create or recreate the instanced mesh used to render boids
function createInstancedMesh()
{
  if (boidInstancedMesh) {
    scene.remove(boidInstancedMesh);
    boidInstancedMesh.geometry.dispose();
    boidInstancedMesh.material.dispose();
  }
  const geometry = new THREE.ConeGeometry(2, 6, 5).rotateX(Math.PI / 2);
  const material = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
  boidInstancedMesh = new THREE.InstancedMesh(geometry, material, boidCount);
  boidInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(boidInstancedMesh);
}

// #endregion

// #region Recreate and reset helpers

// Recreate GPU buffers and instance mesh for a new boid count
function recreateBoids(newCount)
{
  boidCount = newCount;
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
  resetParamsToDefaults();

  initBoidBuffers(boidCount);
  initSpatialHashBuffers();
  initMatrixBuffers();
  syncParamsToGPU();

  createInstancedMesh();
  updateVisualBounds();
  createBindGroups();

  isSimulationRunning = true;
  updateStartPauseButton();
}

// Reinitialize boids for benchmark
function resetBoidsForBenchmark()
{
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);

  initBoidBuffers(boidCount);
  initSpatialHashBuffers();
  initMatrixBuffers();
  syncParamsToGPU();

  createInstancedMesh();
  updateVisualBounds();
  createBindGroups();

  isSimulationRunning = true;
  updateStartPauseButton();
}

// Uniform updates from the UI

// Read UI inputs and sync them into the params array
function updateUniforms()
{
  paramsArray[0] = parseFloat(document.getElementById('separation').value);
  paramsArray[1] = parseFloat(document.getElementById('align').value);
  paramsArray[2] = parseFloat(document.getElementById('cohesion').value);
  paramsArray[3] = parseFloat(document.getElementById('max_speed').value);
  paramsArray[4] = parseFloat(document.getElementById('max_force').value);
  paramsArray[5] = parseFloat(document.getElementById('sep_weight').value);
  paramsArray[6] = parseFloat(document.getElementById('align_weight').value);
  paramsArray[7] = parseFloat(document.getElementById('coh_weight').value);
  paramsArray[8] = parseFloat(document.getElementById('margin').value);
  paramsArray[9] = parseFloat(document.getElementById('turn_factor').value);

  syncParamsToGPU();
}

// #endregion

// #region UI initialization
const p = (v) => parseFloat(v.toPrecision(6));

// Initialize UI controls and wire up event handlers
function initUI()
{
  // Populate inputs with defaults
  document.getElementById('boid-count').value = boidCount;
  document.getElementById('boid-density').value = boidDensity.toFixed(6);
  document.getElementById('separation').value = p(paramsArray[0]);
  document.getElementById('align').value = p(paramsArray[1]);
  document.getElementById('cohesion').value = p(paramsArray[2]);
  document.getElementById('max_speed').value = p(paramsArray[3]);
  document.getElementById('max_force').value = p(paramsArray[4]);
  document.getElementById('sep_weight').value = p(paramsArray[5]);
  document.getElementById('align_weight').value = p(paramsArray[6]);
  document.getElementById('coh_weight').value = p(paramsArray[7]);
  document.getElementById('margin').value = p(paramsArray[8]);
  document.getElementById('turn_factor').value = p(paramsArray[9]);

  // Boid count inputs and handlers
  const boidCountInput = document.getElementById('boid-count');
  const boidDensityInput = document.getElementById('boid-density');
  let boidCountUpdateTimer = null;

  const applyBoidCountFromInput = () =>
  {
    const n = parseInt(boidCountInput.value, 10);
    if (!isNaN(n) && n > 0 && n !== boidCount) {
      recreateBoids(n);
    }
  };

  boidCountInput.addEventListener('change', applyBoidCountFromInput);
  boidCountInput.addEventListener('input', () =>
  {
    if (boidCountUpdateTimer) clearTimeout(boidCountUpdateTimer);
    boidCountUpdateTimer = setTimeout(applyBoidCountFromInput, 250);
  });

  boidDensityInput.addEventListener('input', e =>
  {
    const d = parseFloat(e.target.value);
    if (!isNaN(d) && d > 0) {
      boidDensity = d;
      SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
      resetParamsToDefaults();
      updateVisualBounds();
      // Need to recreate spatial hash buffers for new world size
      initSpatialHashBuffers();
      syncParamsToGPU();
      createBindGroups();
    }
  });

  // Parameter inputs
  const inputs = ['separation', 'align', 'cohesion', 'max_speed', 'max_force',
    'sep_weight', 'align_weight', 'coh_weight', 'margin', 'turn_factor'];
  inputs.forEach(id =>
  {
    document.getElementById(id).addEventListener('input', () =>
    {
      const oldNumCells = numCells;
      updateUniforms();
      // If grid dimensions changed, rebuild spatial hash buffers
      if (numCells !== oldNumCells) {
        initSpatialHashBuffers();
        createBindGroups();
      }
    });
  });

  // Panel collapse
  document.getElementById('toggle-panel').addEventListener('click', () =>
  {
    const body = document.getElementById('settings-body');
    const bs = bootstrap.Collapse.getOrCreateInstance(body);
    bs.toggle();
  });

  // Start/Pause control
  document.getElementById('start-pause-btn').addEventListener('click', () =>
  {
    isSimulationRunning = !isSimulationRunning;
    updateStartPauseButton();
  });

  // Restart control
  document.getElementById('restart-btn').addEventListener('click', () =>
  {
    const inputCount = parseInt(boidCountInput.value, 10);
    const inputDensity = parseFloat(boidDensityInput.value);
    if (!isNaN(inputDensity) && inputDensity > 0) boidDensity = inputDensity;
    if (!isNaN(inputCount) && inputCount > 0) {
      recreateBoids(inputCount);
    } else {
      recreateBoids(boidCount);
    }
    isSimulationRunning = true;
    updateStartPauseButton();
  });

  // Reset control
  document.getElementById('reset-btn').addEventListener('click', resetSimulation);

  // Benchmark control
  document.getElementById('benchmark-btn').addEventListener('click', () =>
  {
    benchmarker.start();
  });

  // Import benchmark LaTeX and open copy/comparison preview
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
      await reportExporter.openBenchmarkPreviewFromTexFiles(files);
    } catch (error) {
      console.error('Failed to import benchmark TeX:', error);
      window.alert('Failed to import benchmark TeX. See console for details.');
    } finally {
      event.target.value = '';
    }
  });

  updateStartPauseButton();

  // Initialize Bootstrap tooltips on all settings inputs
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el =>
  {
    bootstrap.Tooltip.getOrCreateInstance(el, { trigger: 'hover' });
  });
}

// #endregion

// #region Main frame loop

// #region Benchmarker
const BenchmarkState = {
  IDLE: 0,
  WARMING_UP: 1,
  RECORDING: 2,
  COMPLETED: 3
};

// Manages the benchmarking flow, including warm-up, recording, and result export
class BoidBenchmarker
{
  constructor(onResetCallback, onCompleteCallback = null)
  {
    this.state = BenchmarkState.IDLE;
    this.frameTimes = [];
    this.lastFrameTime = 0;
    this.onResetCallback = onResetCallback;
    this.onCompleteCallback = onCompleteCallback;
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

  // Register a hotkey (Esc) to allow the user to cancel the benchmark early
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

  // Unregister the hotkey when benchmark is completed or canceled
  unregisterCancelHotkey()
  {
    if (typeof window === 'undefined' || !this.onEscHandler) return;
    window.removeEventListener('keydown', this.onEscHandler);
    this.onEscHandler = null;
  }

  // Clean up timers, reset state, and call completion callback if provided
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

  // Cancel the benchmark early with an optional reason message
  cancelBenchmark(reason = 'Benchmark canceled.')
  {
    if (this.state !== BenchmarkState.WARMING_UP && this.state !== BenchmarkState.RECORDING) return;
    this.finalizeRun();
    console.log(reason);
  }

  // Start the benchmark flow: warm-up phase followed by recording phase, with appropriate state management and callbacks
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

  // Record a frame time sample, using GPU timestamp if provided, and only if currently in the recording phase
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

  // Record a simulation time sample, only if currently in the recording phase
  recordSimulationSample(simulationMs)
  {
    if (this.state === BenchmarkState.RECORDING && Number.isFinite(simulationMs)) {
      this.simFrameSamples.push(simulationMs);
    }
  }

  // Record a render time sample, only if currently in the recording phase
  recordRenderSample(renderMs)
  {
    if (this.state === BenchmarkState.RECORDING && Number.isFinite(renderMs)) {
      this.renderFrameSamples.push(renderMs);
    }
  }

  // Return the current benchmark status, including phase, time remaining, and visibility for the HUD display
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

  // Complete the benchmark by exporting the results, including frame times, settings, hardware info, and average simulation/render times, then finalize the run and call completion callback
  async completeBenchmark()
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
      await reportExporter.exportPerformanceReport({
        frameTimes: this.frameTimes,
        settings: {
          boidCount,
          separationWeight: paramsArray[5],
          alignmentWeight: paramsArray[6],
          cohesionWeight: paramsArray[7],
          maxSpeed: paramsArray[3],
          updateFrequency: WORKGROUP_SIZE,
          projectName: 'Boid Boys',
          groupName: 'Boid Boys',
          version: 'v1.0.0',
        },
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

const benchmarker = new BoidBenchmarker(
  () =>
  {
    collapseSettingsPanelForBenchmark();
    resetBoidsForBenchmark();
    lockCameraForBenchmark();
  },
  () =>
  {
    unlockCameraAfterBenchmark();
    restoreSettingsPanelAfterBenchmark();
  }
);
// #endregion

// Animation frame: run simulation and render
function frame()
{
  requestAnimationFrame(frame);

  const now = performance.now();

  benchmarker.recordFrame(now);
  updateBenchmarkHUD(now);

  // FPS tracking
  if (lastFrameTime) {
    const dt = now - lastFrameTime;
    frameTimes.push(dt);
    if (frameTimes.length > FPS_SAMPLE_SIZE) frameTimes.shift();

    if (now - lastFPSUpdate >= FPS_UPDATE_INTERVAL) {
      const avgFrame = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
      const fps = 1000 / avgFrame;
      document.getElementById('info-fps').innerText = `FPS: ${fps.toFixed(1)}`;

      // Sim/render timing display
      if (simTimes.length > 0) {
        const avgSim = simTimes.reduce((a, b) => a + b, 0) / simTimes.length;
        document.getElementById('info-step').innerText = `Sim: ${avgSim.toFixed(2)} ms`;
      }
      if (renderTimes.length > 0) {
        const avgRen = renderTimes.reduce((a, b) => a + b, 0) / renderTimes.length;
        document.getElementById('info-gpu').innerText = `Render: ${avgRen.toFixed(2)} ms`;
      }
      lastFPSUpdate = now;
      simTimes = [];
      renderTimes = [];
    }
  }
  lastFrameTime = now;
  document.getElementById('info-boids').innerText = `Boids: ${boidCount}`;
  document.getElementById('gpu-status').innerText = `Cells: ${numCells}`;

  if (controls) controls.update();

  if (useGPU && !isMapping && isSimulationRunning) {
    const simStart = performance.now();

    syncParamsToGPU();

    const encoder = gpuDevice.createCommandEncoder();
    const wgBoids = Math.ceil(boidCount / WORKGROUP_SIZE);
    const wgCells = Math.ceil(numCells / WORKGROUP_SIZE);

    // Pass 1: clear cell heads
    const p1 = encoder.beginComputePass();
    p1.setPipeline(clearCellsPipeline);
    p1.setBindGroup(0, bindGroup);
    p1.dispatchWorkgroups(wgCells);
    p1.end();

    // Pass 2: hash insert
    const p2 = encoder.beginComputePass();
    p2.setPipeline(hashInsertPipeline);
    p2.setBindGroup(0, bindGroup);
    p2.dispatchWorkgroups(wgBoids);
    p2.end();

    // Pass 3: update boids
    const p3 = encoder.beginComputePass();
    p3.setPipeline(updateBoidsPipeline);
    p3.setBindGroup(0, bindGroup);
    p3.dispatchWorkgroups(wgBoids);
    p3.end();

    // Pass 4: compute instance matrices
    const p4 = encoder.beginComputePass();
    p4.setPipeline(computeMatricesPipeline);
    p4.setBindGroup(0, bindGroup);
    p4.dispatchWorkgroups(wgBoids);
    p4.end();

    // Copy matrices to staging for CPU readback
    encoder.copyBufferToBuffer(matrixBuffer, 0, matrixStagingBuffer, 0, matrixBuffer.size);
    gpuDevice.queue.submit([encoder.finish()]);

    const simEnd = performance.now();
    const simDelta = simEnd - simStart;
    simTimes.push(simDelta);
    benchmarker.recordSimulationSample(simDelta);

    // Async readback of instance matrices
    isMapping = true;
    matrixStagingBuffer.mapAsync(GPUMapMode.READ).then(() =>
    {
      const renderStart = performance.now();

      const matData = new Float32Array(matrixStagingBuffer.getMappedRange());
      boidInstancedMesh.instanceMatrix.array.set(matData);
      boidInstancedMesh.instanceMatrix.needsUpdate = true;

      matrixStagingBuffer.unmap();
      isMapping = false;

      const renderDelta = performance.now() - renderStart;
      renderTimes.push(renderDelta);
      benchmarker.recordRenderSample(renderDelta);
    }).catch(() => { isMapping = false; });
  }

  renderer.render(scene, camera);
}

// #endregion

// #region UI helper functions

// Update the start/pause button appearance based on simulation state
function updateStartPauseButton()
{
  const btn = document.getElementById('start-pause-btn');
  const icon = document.getElementById('start-icon');
  if (isSimulationRunning) {
    icon.className = 'bi bi-pause-fill';
    btn.classList.add('btn-success');
    btn.classList.remove('btn-warning');
  } else {
    icon.className = 'bi bi-play-fill';
    btn.classList.add('btn-warning');
    btn.classList.remove('btn-success');
  }
}

// Collapse settings panel at benchmark start while remembering previous state
function collapseSettingsPanelForBenchmark()
{
  const body = document.getElementById('settings-body');
  if (!body) return;

  wasSettingsPanelOpenBeforeBenchmark = body.classList.contains('show');
  const bs = bootstrap.Collapse.getOrCreateInstance(body, { toggle: false });
  bs.hide();
}

// Restore settings panel visibility to its pre-benchmark state
function restoreSettingsPanelAfterBenchmark()
{
  if (wasSettingsPanelOpenBeforeBenchmark === null) return;

  const body = document.getElementById('settings-body');
  if (!body) {
    wasSettingsPanelOpenBeforeBenchmark = null;
    return;
  }

  const bs = bootstrap.Collapse.getOrCreateInstance(body, { toggle: false });
  if (wasSettingsPanelOpenBeforeBenchmark) {
    bs.show();
  } else {
    bs.hide();
  }

  wasSettingsPanelOpenBeforeBenchmark = null;
}

// Lock camera to a fixed default position for a reproducible benchmark view
function lockCameraForBenchmark()
{
  if (!camera || !controls) return;
  const cx = SIMULATION_SIZE.x / 2;
  const cy = SIMULATION_SIZE.y / 2;
  const cz = SIMULATION_SIZE.z / 2;
  const d = SIMULATION_SIZE.x * 1.5;
  camera.position.set(cx + d, cy + d, cz + d);
  controls.target.set(cx, cy, cz);
  controls.update();
  controls.enabled = false;
}

// Re-enable camera controls after benchmark completes
function unlockCameraAfterBenchmark()
{
  if (!controls) return;
  controls.enabled = true;
}

function updateBenchmarkHUD(now)
{
  const hud = document.getElementById('benchmark-hud');
  if (!hud) return;

  const statusEl = hud.querySelector('.status');
  const countdownEl = hud.querySelector('.countdown');
  if (!statusEl || !countdownEl) return;

  const state = benchmarker.getStatus(now);
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

// Reset simulation to default parameters and recreate boids
function resetSimulation()
{
  boidCount = 100000;
  boidDensity = 0.00005;
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
  resetParamsToDefaults();
  recreateBoids(boidCount);

  // Update UI inputs
  document.getElementById('boid-count').value = boidCount;
  document.getElementById('boid-density').value = boidDensity.toFixed(6);
  document.getElementById('separation').value = p(paramsArray[0]);
  document.getElementById('align').value = p(paramsArray[1]);
  document.getElementById('cohesion').value = p(paramsArray[2]);
  document.getElementById('max_speed').value = p(paramsArray[3]);
  document.getElementById('max_force').value = p(paramsArray[4]);
  document.getElementById('sep_weight').value = p(paramsArray[5]);
  document.getElementById('align_weight').value = p(paramsArray[6]);
  document.getElementById('coh_weight').value = p(paramsArray[7]);
  document.getElementById('margin').value = p(paramsArray[8]);
  document.getElementById('turn_factor').value = p(paramsArray[9]);

  syncParamsToGPU();
}

// #endregion

// #region Bootstrap

initWebGPU().then(() =>
{
  initThree();
  frame();
});

// #endregion
