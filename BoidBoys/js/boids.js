import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ───── Simulation constants ─────
const BOID_COUNT = 100000;
const WORKGROUP_SIZE = 256;
const SIMULATION_SIZE = { x: 5000, y: 5000, z: 5000 };

// Spatial-hash grid (must match shader constants)
const CELL_SIZE = 50;
const GRID_X = Math.ceil(SIMULATION_SIZE.x / CELL_SIZE); // 20
const GRID_Y = Math.ceil(SIMULATION_SIZE.y / CELL_SIZE); // 12
const GRID_Z = Math.ceil(SIMULATION_SIZE.z / CELL_SIZE); // 12
const NUM_CELLS = GRID_X * GRID_Y * GRID_Z;             // 2 880

// ───── Three.js state ─────
let scene, camera, renderer, controls;
let boidMesh, boidInterleavedBuffer;

// ───── WebGPU state ─────
let gpuDevice;
let clearCountsPipeline, countBoidsPipeline, prefixSumPipeline, scatterPipeline, updateBoidsPipeline;
let boidBuffer, hashBuffer, cellCountBuffer, cellStartBuffer, cellEndBuffer;
let clearCountsBG, countBoidsBG, prefixSumBG, scatterBG, updateBoidsBG;
let stagingBuffers = [];
let stagingMapped = [false, false];
let currentStaging = 0;
let useGPU = false;

// ───── HUD / diagnostics ─────
let infoAppEl, infoFpsEl, infoBoidsEl, infoStepEl, infoGpuEl, gpuStatusEl;
let lastTime = performance.now();
let fps = 0;
const FPS_SMOOTHING = 0;
// Framerate cap: keep renders at or below 60Hz
let lastRenderTime = performance.now();
const MAX_FPS = 60;
const FRAME_INTERVAL = 1000 / MAX_FPS;
let stepCount = 0;
let gpuAdapter = null;
// Update the visible FPS only every N frames to avoid rapid jumps
const FPS_DISPLAY_INTERVAL_FRAMES = 60;
let framesSinceFpsUpdate = 0;
let displayedFps = 0;

// ═══════════════════════════════════════════════════════════════
//  WebGPU initialisation — counting-sort spatial-hash pipeline
// ═══════════════════════════════════════════════════════════════
async function initWebGPU()
{
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    const el = document.getElementById('info-app');
    if (el) el.innerText = 'WebGPU not supported';
    return false;
  }
  gpuAdapter = adapter;
  gpuDevice = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  // ── Load shader module (all 5 entry points in one file) ──
  const shaderCode = await fetch('compute-shader.wgsl').then(r => r.text());
  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

  // ── 1. Boid buffer (position + velocity, 8 floats each) ──
  const boidData = new Float32Array(BOID_COUNT * 8);
  for (let i = 0; i < BOID_COUNT; i++) {
    boidData[i * 8 + 0] = Math.random() * SIMULATION_SIZE.x;
    boidData[i * 8 + 1] = Math.random() * SIMULATION_SIZE.y;
    boidData[i * 8 + 2] = Math.random() * SIMULATION_SIZE.z;
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

  // ── 2. Hash-entry buffer (exactly BOID_COUNT — no padding with counting sort) ──
  hashBuffer = gpuDevice.createBuffer({
    size: BOID_COUNT * 8, // struct HashEntry { cell_hash: u32, boid_index: u32 }
    usage: GPUBufferUsage.STORAGE,
  });

  // ── 3. Cell count buffer (atomic counters, 2880 cells) ──
  cellCountBuffer = gpuDevice.createBuffer({
    size: NUM_CELLS * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // ── 4. Cell start buffer (prefix-sum result) ──
  cellStartBuffer = gpuDevice.createBuffer({
    size: NUM_CELLS * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // ── 5. Cell end buffer (scatter offset → becomes cell_end after scatter) ──
  cellEndBuffer = gpuDevice.createBuffer({
    size: NUM_CELLS * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // ── 6. Double-buffered staging buffers (avoids GPU stalls on readback) ──
  const stagingSize = boidData.byteLength;
  stagingBuffers = [
    gpuDevice.createBuffer({ size: stagingSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    gpuDevice.createBuffer({ size: stagingSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  ];

  // ════════════════ Compute pipelines (auto layout) ════════════════

  clearCountsPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'clear_counts' },
  });

  countBoidsPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'count_boids' },
  });

  prefixSumPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'prefix_sum' },
  });

  scatterPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'scatter' },
  });

  updateBoidsPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'update_boids' },
  });

  // ════════════════ Bind groups ════════════════

  // clear_counts → binding 2
  clearCountsBG = gpuDevice.createBindGroup({
    layout: clearCountsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 2, resource: { buffer: cellCountBuffer } },
    ],
  });

  // count_boids → bindings 0, 2
  countBoidsBG = gpuDevice.createBindGroup({
    layout: countBoidsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 2, resource: { buffer: cellCountBuffer } },
    ],
  });

  // prefix_sum → bindings 2, 3, 4
  prefixSumBG = gpuDevice.createBindGroup({
    layout: prefixSumPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 2, resource: { buffer: cellCountBuffer } },
      { binding: 3, resource: { buffer: cellStartBuffer } },
      { binding: 4, resource: { buffer: cellEndBuffer } },
    ],
  });

  // scatter → bindings 0, 1, 4
  scatterBG = gpuDevice.createBindGroup({
    layout: scatterPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: hashBuffer } },
      { binding: 4, resource: { buffer: cellEndBuffer } },
    ],
  });

  // update_boids → bindings 0, 1, 3, 4
  updateBoidsBG = gpuDevice.createBindGroup({
    layout: updateBoidsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: hashBuffer } },
      { binding: 3, resource: { buffer: cellStartBuffer } },
      { binding: 4, resource: { buffer: cellEndBuffer } },
    ],
  });

  useGPU = true;
  const infoApp = document.getElementById('info-app');
  if (infoApp) infoApp.innerText = 'WebGPU Running (Counting Sort)';
  if (gpuStatusEl) gpuStatusEl.innerText = gpuAdapter?.name ?? 'WebGPU Adapter';
  if (infoGpuEl) infoGpuEl.innerText = 'Yes';
  return true;
}

function initThree()
{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000005);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 10000);
  camera.position.set(-500, 600, 1000);

  renderer = new THREE.WebGLRenderer({ antialias: false }); // AA off for perf at high counts
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // cap DPR

  // FIX: Append to the specific container, not the body!
  const container = document.getElementById('canvas-container');
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(500, 300, 300);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Visual Bounds
  const boxGeom = new THREE.BoxGeometry(SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z);
  const edges = new THREE.EdgesGeometry(boxGeom);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x444444 }));
  line.position.set(500, 300, 300);
  scene.add(line);

  // Boids — InstancedBufferGeometry + custom shader (orientation on GPU, zero CPU math)
  const coneGeo = new THREE.ConeGeometry(2, 6, 3); // 3 segments for lower vertex count
  coneGeo.rotateX(Math.PI / 2); // tip along +Z
  const instancedGeo = new THREE.InstancedBufferGeometry();
  instancedGeo.index = coneGeo.index;
  instancedGeo.setAttribute('position', coneGeo.getAttribute('position'));
  instancedGeo.setAttribute('normal', coneGeo.getAttribute('normal'));
  instancedGeo.instanceCount = BOID_COUNT;

  // Interleaved instance buffer — layout matches GPU boid struct [pos.xyzw, vel.xyzw]
  boidInterleavedBuffer = new THREE.InstancedInterleavedBuffer(
    new Float32Array(BOID_COUNT * 8), 8
  );
  boidInterleavedBuffer.setUsage(THREE.DynamicDrawUsage);
  instancedGeo.setAttribute('boidPos', new THREE.InterleavedBufferAttribute(boidInterleavedBuffer, 3, 0));
  instancedGeo.setAttribute('boidVel', new THREE.InterleavedBufferAttribute(boidInterleavedBuffer, 3, 4));

  const boidMaterial = new THREE.ShaderMaterial({
    vertexShader: /* glsl */`
      attribute vec3 boidPos;
      attribute vec3 boidVel;
      varying vec3 vNormal;

      void main() {
        // Orient cone tip along velocity
        vec3 dir = length(boidVel) > 0.01 ? normalize(boidVel) : vec3(0.0, 0.0, 1.0);
        vec3 worldUp = abs(dir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 right = normalize(cross(worldUp, dir));
        vec3 up = cross(dir, right);
        mat3 rot = mat3(right, up, dir);

        vec3 transformed = rot * position + boidPos;
        vNormal = normalMatrix * (rot * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3 vNormal;

      void main() {
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normalize(vNormal), lightDir), 0.0);
        vec3 col = vec3(0.0, 1.0, 0.533) * (0.3 + 0.7 * diff);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    side: THREE.DoubleSide,
  });

  boidMesh = new THREE.Mesh(instancedGeo, boidMaterial);
  boidMesh.frustumCulled = false;
  scene.add(boidMesh);

  scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0xffffff, 0.3));

  // Wire HUD elements (they exist in the page)
  infoAppEl = document.getElementById('info-app');
  infoFpsEl = document.getElementById('info-fps');
  infoBoidsEl = document.getElementById('info-boids');
  infoStepEl = document.getElementById('info-step');
  infoGpuEl = document.getElementById('info-gpu');
  gpuStatusEl = document.getElementById('gpu-status');

  if (infoBoidsEl) infoBoidsEl.innerText = `Boids: ${BOID_COUNT}`;
  if (infoGpuEl) infoGpuEl.innerText = useGPU ? 'Yes' : 'No';
  if (gpuStatusEl && gpuAdapter) gpuStatusEl.innerText = gpuAdapter.name || 'WebGPU Adapter';

  window.addEventListener('resize', () =>
  {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function frame()
{
  requestAnimationFrame(frame);

  const now = performance.now();

  // Throttle renders to MAX_FPS (60Hz by default). Still update controls between frames.
  if (now - lastRenderTime < FRAME_INTERVAL) {
    if (controls) controls.update();
    return;
  }

  const delta = now - lastRenderTime;
  lastRenderTime = now;
  const instantFps = 1000 / delta;
  fps = fps > 0 ? FPS_SMOOTHING * fps + (1 - FPS_SMOOTHING) * instantFps : instantFps;

  // Always update controls so the camera movement is fluid
  if (controls) controls.update();

  if (useGPU) {
    // Double-buffered staging: pick a buffer that isn't being mapped
    const si = stagingMapped[currentStaging] ? 1 - currentStaging : currentStaging;
    currentStaging = 1 - si;

    if (!stagingMapped[si]) {
      const encoder = gpuDevice.createCommandEncoder();
      let pass;

      // ── Pass 1: Clear cell counts ──
      pass = encoder.beginComputePass();
      pass.setPipeline(clearCountsPipeline);
      pass.setBindGroup(0, clearCountsBG);
      pass.dispatchWorkgroups(Math.ceil(NUM_CELLS / WORKGROUP_SIZE));
      pass.end();

      // ── Pass 2: Count boids per cell ──
      pass = encoder.beginComputePass();
      pass.setPipeline(countBoidsPipeline);
      pass.setBindGroup(0, countBoidsBG);
      pass.dispatchWorkgroups(Math.ceil(BOID_COUNT / WORKGROUP_SIZE));
      pass.end();

      // ── Pass 3: Prefix sum (single workgroup) ──
      pass = encoder.beginComputePass();
      pass.setPipeline(prefixSumPipeline);
      pass.setBindGroup(0, prefixSumBG);
      pass.dispatchWorkgroups(1);
      pass.end();

      // ── Pass 4: Scatter into sorted order ──
      pass = encoder.beginComputePass();
      pass.setPipeline(scatterPipeline);
      pass.setBindGroup(0, scatterBG);
      pass.dispatchWorkgroups(Math.ceil(BOID_COUNT / WORKGROUP_SIZE));
      pass.end();

      // ── Pass 5: Update boids via spatial-hash neighbour search ──
      pass = encoder.beginComputePass();
      pass.setPipeline(updateBoidsPipeline);
      pass.setBindGroup(0, updateBoidsBG);
      pass.dispatchWorkgroups(Math.ceil(BOID_COUNT / WORKGROUP_SIZE));
      pass.end();

      stepCount++;

      // ── Copy results to staging buffer for CPU read-back ──
      encoder.copyBufferToBuffer(boidBuffer, 0, stagingBuffers[si], 0, boidBuffer.size);
      gpuDevice.queue.submit([encoder.finish()]);

      stagingMapped[si] = true;
      stagingBuffers[si].mapAsync(GPUMapMode.READ).then(() =>
      {
        const data = new Float32Array(stagingBuffers[si].getMappedRange());
        boidInterleavedBuffer.array.set(data);
        boidInterleavedBuffer.needsUpdate = true;
        stagingBuffers[si].unmap();
        stagingMapped[si] = false;
      }).catch(() => { stagingMapped[si] = false; });
    }
  }

  // Update HUD — only refresh the visible FPS every FPS_DISPLAY_INTERVAL_FRAMES
  framesSinceFpsUpdate++;
  if (framesSinceFpsUpdate >= FPS_DISPLAY_INTERVAL_FRAMES) {
    displayedFps = Math.round(fps);
    framesSinceFpsUpdate = 0;
    if (infoFpsEl) infoFpsEl.innerText = `FPS: ${displayedFps}`;
  }
  if (infoStepEl) infoStepEl.innerText = `Step: ${stepCount}`;
  if (infoGpuEl) infoGpuEl.innerText = useGPU ? 'Yes' : 'No';

  renderer.render(scene, camera);
}

initWebGPU().then(() =>
{
  initThree();
  frame();
});