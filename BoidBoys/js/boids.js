import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const BOID_COUNT = 100000;
const WORKGROUP_SIZE = 256;
const SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };

let scene, camera, renderer, boidInstancedMesh, controls;
let gpuDevice, computePipeline, boidBuffer, stagingBuffer, kdTreeBuffer, bindGroup;
let useGPU = false;
let isMapping = false;

let frames = 0;
let prevTime = performance.now();

const KD_NODE_INTS = 4;
let kdIndices, kdNodes, boidDataCopy, pendingKDTree;

// Pre-allocate math objects to save memory and CPU cycles
const _matrix = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _orient = new THREE.Quaternion();
const _vel = new THREE.Vector3();
const _up = new THREE.Vector3(0, 0, 1);

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

  const boidData = new Float32Array(BOID_COUNT * 8);
  for (let i = 0; i < BOID_COUNT; i++) {
    boidData[i * 8] = Math.random() * SIMULATION_SIZE.x;
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

  stagingBuffer = gpuDevice.createBuffer({
    size: boidData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // --- K-d tree ---
  // Pre-allocate working arrays
  kdIndices = new Uint32Array(BOID_COUNT);
  kdNodes = new Int32Array(BOID_COUNT * KD_NODE_INTS);
  boidDataCopy = new Float32Array(BOID_COUNT * 8);

  // Build the initial tree from the starting positions
  pendingKDTree = buildKDTree(boidData);

  kdTreeBuffer = gpuDevice.createBuffer({
    size: kdNodes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]
  });

  bindGroup = gpuDevice.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: kdTreeBuffer } },
    ]
  });

  computePipeline = gpuDevice.createComputePipeline({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'main' }
  });

  useGPU = true;
  document.getElementById('info-app').innerText = "WebGPU Running";
  return true;
}

function initThree()
{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000005);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 10000);
  camera.position.set(-500, 600, 1000);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

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

  // Boids
  const geometry = new THREE.ConeGeometry(2, 6, 5).rotateX(Math.PI / 2);
  const material = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
  boidInstancedMesh = new THREE.InstancedMesh(geometry, material, BOID_COUNT);
  boidInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(boidInstancedMesh);

  document.getElementById('info-boids').innerText = `Boids: ${BOID_COUNT}`;

  scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0xffffff, 0.3));

  window.addEventListener('resize', () =>
  {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function nthElement(start, end, nth, axis, data)
{
  while (start < end - 1) {
    const mid = (start + end) >> 1;
    const a = data[kdIndices[start] * 8 + axis];
    const b = data[kdIndices[mid] * 8 + axis];
    const c = data[kdIndices[end - 1] * 8 + axis];

    let pivotIdx;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = start;
    else pivotIdx = end - 1;

    const pivotVal = data[kdIndices[pivotIdx] * 8 + axis];

    let tmp = kdIndices[pivotIdx];
    kdIndices[pivotIdx] = kdIndices[end - 1];
    kdIndices[end - 1] = tmp;

    let store = start;
    for (let i = start; i < end - 1; i++) {
      if (data[kdIndices[i] * 8 + axis] < pivotVal) {
        tmp = kdIndices[store];
        kdIndices[store] = kdIndices[i];
        kdIndices[i] = tmp;
        store++;
      }
    }
    tmp = kdIndices[store];
    kdIndices[store] = kdIndices[end - 1];
    kdIndices[end - 1] = tmp;

    if (store === nth) return;
    else if (store < nth) start = store + 1;
    else end = store;
  }
}

function buildKDTree(boidData)
{
  for (let i = 0; i < BOID_COUNT; i++) kdIndices[i] = i;
  let nodeCount = 0;

  function build(start, end, depth)
  {
    if (start >= end) return -1;

    const axis = depth % 3;
    const mid = (start + end) >> 1;

    nthElement(start, end, mid, axis, boidData);

    const ni = nodeCount++;
    const off = ni * KD_NODE_INTS;
    kdNodes[off] = kdIndices[mid];  // boid_idx
    kdNodes[off + 1] = axis;          // split_axis
    kdNodes[off + 2] = build(start, mid, depth + 1);   // left
    kdNodes[off + 3] = build(mid + 1, end, depth + 1); // right
    return ni;
  }

  build(0, BOID_COUNT, 0);
  return kdNodes;
}

function frame()
{
  requestAnimationFrame(frame);

  const time = performance.now();
  frames++;

  if (time - prevTime >= 1000) {
    const fps = Math.round((frames * 1000) / (time - prevTime));
    document.getElementById('info-fps').innerText = `FPS: ${fps}`;
    frames = 0;
    prevTime = time;
  }

  // Always update controls so the camera movement is fluid
  if (controls) controls.update();

  if (useGPU && !isMapping) {
    if (pendingKDTree) {
      gpuDevice.queue.writeBuffer(kdTreeBuffer, 0, pendingKDTree);
      pendingKDTree = null;
    }

    const commandEncoder = gpuDevice.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(BOID_COUNT / WORKGROUP_SIZE));
    pass.end();

    commandEncoder.copyBufferToBuffer(boidBuffer, 0, stagingBuffer, 0, boidBuffer.size);
    gpuDevice.queue.submit([commandEncoder.finish()]);

    isMapping = true;
    stagingBuffer.mapAsync(GPUMapMode.READ).then(() =>
    {
      const data = new Float32Array(stagingBuffer.getMappedRange());

      boidDataCopy.set(data);

      for (let i = 0; i < BOID_COUNT; i++) {
        const stride = i * 8;
        _pos.set(data[stride], data[stride + 1], data[stride + 2]);
        _vel.set(data[stride + 4], data[stride + 5], data[stride + 6]);

        if (_vel.lengthSq() > 0.01) {
          _orient.setFromUnitVectors(_up, _vel.clone().normalize());
        }

        _matrix.compose(_pos, _orient, { x: 1, y: 1, z: 1 });
        boidInstancedMesh.setMatrixAt(i, _matrix);
      }

      boidInstancedMesh.instanceMatrix.needsUpdate = true;
      stagingBuffer.unmap();
      isMapping = false;

      pendingKDTree = buildKDTree(boidDataCopy);
    }).catch(() => { isMapping = false; });
  }

  document.getElementById('info-gpu').innerText = `GPU: ${useGPU ? "Yes" : "No"}`;
  document.getElementById('info-step').innerText = `Step: ${0}`;
  renderer.render(scene, camera);
}

initWebGPU().then(() =>
{
  initThree();
  frame();
});