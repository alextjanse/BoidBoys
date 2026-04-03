import * as THREE from 'three';
import { BoidEngine } from './BoidEngine.js';
import { BoidRenderer } from './BoidRenderer.js';
import { BoidBenchmarker } from './Benchmarker.js';
import { UIManager } from './UIManager.js';
import { FPS_SAMPLE_SIZE, FPS_UPDATE_INTERVAL, ParamsIndex } from './constants.js';
import { PerformanceReportExporter } from './performance-report.ts';

let engine;
let renderer;
let benchmarker;
let uiManager;

let isSimulationRunning = true;
let boidCount = 100000;
let boidDensity = 0.000005;

let lastFrameTime = 0;
let frameTimes = [];
let lastFPSUpdate = 0;
let simTimes = [];
let renderTimes = [];

let isColor;

const reportExporter = new PerformanceReportExporter();
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

function onWindowMouseMove(event)
{
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

window.addEventListener('mousemove', onWindowMouseMove);

async function init()
{
  const boidData = new Float32Array(boidCount * 8);

  engine = new BoidEngine();
  const useGPU = await engine.init(boidCount, boidDensity, boidData);

  if (!useGPU) return;

  document.getElementById('info-app').innerText = "WebGPU Running";

  renderer = new BoidRenderer('canvas-container', boidData);
  await renderer.init();
  renderer.updateVisualBounds(engine.simulationSize);
  renderer.createInstancedMesh(boidCount, boidDensity);

  benchmarker = new BoidBenchmarker(
    () =>
    {
      uiManager.collapseSettingsPanelForBenchmark();
      // Fully reset the simulation for a clean benchmark run
      engine.recreateBoids(boidCount, boidDensity, boidData);
      renderer.updateVisualBounds(engine.simulationSize);
      renderer.createInstancedMesh(boidCount, boidDensity);
      engine.syncParams();
      lockCameraForBenchmark();
    },
    () =>
    {
      unlockCameraAfterBenchmark();
      uiManager.restoreSettingsPanelAfterBenchmark();
    },
    reportExporter
  );

  const callbacks = {
    getBoidCount: () => boidCount,
    getBoidDensity: () => boidDensity,
    onRecreateBoids: (newCount, newDensity) =>
    {
      boidCount = newCount;
      boidDensity = newDensity;
      engine.recreateBoids(newCount, newDensity, boidData);
      renderer.updateVisualBounds(engine.simulationSize);
      renderer.createInstancedMesh(boidCount, boidDensity);
      uiManager.updateInfo(boidCount, engine.numCells);
    },
    onUpdateUniforms: (values) =>
    {
      const oldNumCells = engine.numCells;

      engine.paramsArray[ParamsIndex.SEPARATION_DIST] = values.separation;
      engine.paramsArray[ParamsIndex.ALIGN_DIST] = values.align;
      engine.paramsArray[ParamsIndex.COHESION_DIST] = values.cohesion;
      engine.paramsArray[ParamsIndex.MAX_SPEED] = values.max_speed;
      engine.paramsArray[ParamsIndex.MAX_FORCE] = values.max_force;
      engine.paramsArray[ParamsIndex.SEPARATION_WEIGHT] = values.sep_weight;
      engine.paramsArray[ParamsIndex.ALIGNMENT_WEIGHT] = values.align_weight;
      engine.paramsArray[ParamsIndex.COHESION_WEIGHT] = values.coh_weight;
      engine.paramsArray[ParamsIndex.MARGIN] = values.margin;
      engine.paramsArray[ParamsIndex.TURN_FACTOR] = values.turn_factor;
      // UI provides vision angle in degrees — convert to radians for the shader
      if (typeof values.vision_angle !== 'undefined') {
        engine.paramsArray[ParamsIndex.VISION_ANGLE] = values.vision_angle * Math.PI / 180.0;
      }

      engine.syncParams();

      if (engine.numCells !== oldNumCells) {
        engine.initSpatialHashBuffers();
        engine.createBindGroups();
      }
    },
    onSimulationToggle: (isRunning) =>
    {
      isSimulationRunning = isRunning;
    },
    onResetSimulation: () =>
    {
      boidCount = 100000;
      boidDensity = 0.000005;
      engine.recreateBoids(boidCount, boidDensity, boidData);
      renderer.updateVisualBounds(engine.simulationSize);
      renderer.createInstancedMesh(boidCount, boidDensity);

      uiManager.populateInputs({ boidCount, boidDensity, params: engine.paramsArray });
    },
    onBenchmarkStart: () =>
    {
      // Capture current state BEFORE starting
      const currentSettings = {
        boidCount: boidCount,
        separationWeight: engine.paramsArray[ParamsIndex.SEPARATION_WEIGHT],
        alignmentWeight: engine.paramsArray[ParamsIndex.ALIGNMENT_WEIGHT],
        cohesionWeight: engine.paramsArray[ParamsIndex.COHESION_WEIGHT],
        maxSpeed: engine.paramsArray[ParamsIndex.MAX_SPEED],
        updateFrequency: 60, // Or your target frequency
        projectName: 'Boid Boys',
        groupName: 'Boid Boys',
        version: 'v1.0.0',
      };

      // Pass this object to the benchmarker
      benchmarker.start(currentSettings);
    },
    onImportReport: async (files) => await reportExporter.openBenchmarkPreviewFromTexFiles(files),
    onColorToggle: () =>
    {
      isColor = !isColor;
    }
  };

  uiManager = new UIManager(callbacks);
  uiManager.init({ boidCount, boidDensity, params: engine.paramsArray });
  uiManager.updateInfo(boidCount, engine.numCells);

  frame();
}

function lockCameraForBenchmark()
{
  if (!renderer.camera || !renderer.controls) return;
  const cx = engine.simulationSize.x / 2;
  const cy = engine.simulationSize.y / 2;
  const cz = engine.simulationSize.z / 2;
  const d = engine.simulationSize.x * 1.5;
  renderer.camera.position.set(cx + d, cy + d, cz + d);
  renderer.controls.target.set(cx, cy, cz);
  renderer.controls.update();
  renderer.controls.enabled = false;
}

function unlockCameraAfterBenchmark()
{
  if (!renderer.controls) return;
  renderer.controls.enabled = true;
}

async function frame()
{
  requestAnimationFrame(frame);

  const now = performance.now();
  benchmarker.recordFrame(now);

  uiManager.updateBenchmarkHUD(benchmarker.getStatus(now));

  if (lastFrameTime) {
    const dt = now - lastFrameTime;
    frameTimes.push(dt);
    if (frameTimes.length > FPS_SAMPLE_SIZE) frameTimes.shift();

    if (now - lastFPSUpdate >= FPS_UPDATE_INTERVAL) {
      const avgFrame = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
      const fps = 1000 / avgFrame;

      let avgSim = null;
      let avgRen = null;

      if (simTimes.length > 0) {
        avgSim = simTimes.reduce((a, b) => a + b, 0) / simTimes.length;
      }
      if (renderTimes.length > 0) {
        avgRen = renderTimes.reduce((a, b) => a + b, 0) / renderTimes.length;
      }

      uiManager.updateFPS(fps, avgSim, avgRen);

      lastFPSUpdate = now;
      simTimes = [];
      renderTimes = [];
    }
  }
  lastFrameTime = now;
  uiManager.updateInfo(boidCount, engine.numCells);

  if (isSimulationRunning) {
    raycaster.setFromCamera(mouse, renderer.camera);
    const origin = raycaster.ray.origin;
    const dir = raycaster.ray.direction;

    engine.paramsArray[ParamsIndex.MOUSE_RAY_ORIGIN] = origin.x;
    engine.paramsArray[ParamsIndex.MOUSE_RAY_ORIGIN + 1] = origin.y;
    engine.paramsArray[ParamsIndex.MOUSE_RAY_ORIGIN + 2] = origin.z;

    engine.paramsArray[ParamsIndex.RAY_DIRECTION] = dir.x;
    engine.paramsArray[ParamsIndex.RAY_DIRECTION + 1] = dir.y;
    engine.paramsArray[ParamsIndex.RAY_DIRECTION + 2] = dir.z;

    engine.paramsArray[ParamsIndex.FLEE_RADIUS] = engine.simulationSize.x * 0.12;

    const results = await engine.step(renderer);
    if (results) {
      simTimes.push(results.simDelta);
      benchmarker.recordSimulationSample(results.simDelta);
      renderTimes.push(results.renderDelta);
      benchmarker.recordRenderSample(results.renderDelta);
    }
  }

  renderer.render(now);
}

init();