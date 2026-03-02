import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

let boidCount = 15000;
const WORKGROUP_SIZE = 256;
const SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };

// buffers & pipeline globals so we can rebind when settings change
let bindGroupLayout, uniformBuffer;

// Create a small uniform buffer holding global scene/simulation parameters
const paramsArray = new Float32Array(16); // 16 floats = 64 bytes
// layout: separation, align, cohesion, max_speed,
//         max_force, sep_w, align_w, coh_w,
//         margin, turn_factor, padding0, padding1,
//         world_max.x, world_max.y, world_max.z, world_max.w
function resetParamsToDefaults() {
  paramsArray[0] = 25.0; // separation_dist
  paramsArray[1] = 50.0; // align_dist
  paramsArray[2] = 50.0; // cohesion_dist
  paramsArray[3] = 5.0;  // max_speed
  paramsArray[4] = 0.1;  // max_force
  paramsArray[5] = 1.5;  // separation_weight
  paramsArray[6] = 1.0;  // alignment_weight
  paramsArray[7] = 0.5;  // cohesion_weight
  paramsArray[8] = 100.0; // margin
  paramsArray[9] = 0.2;   // turn_factor
  paramsArray[10] = 0.0;  // padding0
  paramsArray[11] = 0.0;  // padding1
  paramsArray[12] = SIMULATION_SIZE.x;
  paramsArray[13] = SIMULATION_SIZE.y;
  paramsArray[14] = SIMULATION_SIZE.z;
  paramsArray[15] = 0.0;
}
resetParamsToDefaults();

let scene, camera, renderer, boidInstancedMesh, controls;
let gpuDevice, computePipeline, boidBuffer, stagingBuffer, bindGroup;
let useGPU = false;
let isMapping = false;
let isSimulationRunning = true;
let lastFrameTime = 0;

// Pre-allocate math objects to save memory and CPU cycles
const _matrix = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _orient = new THREE.Quaternion();
const _vel = new THREE.Vector3();
const _up = new THREE.Vector3( 0, 0, 1 );

function initBoidBuffers(count) {
  const boidData = new Float32Array( count * 8 );
  for ( let i = 0; i < count; i++ )
  {
    boidData[ i * 8 ] = Math.random() * SIMULATION_SIZE.x;
    boidData[ i * 8 + 1 ] = Math.random() * SIMULATION_SIZE.y;
    boidData[ i * 8 + 2 ] = Math.random() * SIMULATION_SIZE.z;
    boidData[ i * 8 + 3 ] = 1.0;
    boidData[ i * 8 + 4 ] = ( Math.random() - 0.5 ) * 4;
    boidData[ i * 8 + 5 ] = ( Math.random() - 0.5 ) * 4;
    boidData[ i * 8 + 6 ] = ( Math.random() - 0.5 ) * 4;
    boidData[ i * 8 + 7 ] = 0.0;
  }

  boidBuffer = gpuDevice.createBuffer( {
    size: boidData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  } );
  new Float32Array( boidBuffer.getMappedRange() ).set( boidData );
  boidBuffer.unmap();

  stagingBuffer = gpuDevice.createBuffer( {
    size: boidData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  } );
}

async function initWebGPU ()
{
  const adapter = await navigator.gpu?.requestAdapter();
  if ( !adapter )
  {
    document.getElementById( 'info-app' ).innerText = "WebGPU not supported";
    return false;
  }
  gpuDevice = await adapter.requestDevice();

  const shaderCode = await fetch( 'compute-shader.wgsl' ).then( r => r.text() );
  const shaderModule = gpuDevice.createShaderModule( { code: shaderCode } );


  initBoidBuffers(boidCount);

  // uniform buffer (reuse if already created)
  if (!uniformBuffer) {
    uniformBuffer = gpuDevice.createBuffer({
      size: paramsArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);

  bindGroupLayout = gpuDevice.createBindGroupLayout( {
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]
  } );

  bindGroup = gpuDevice.createBindGroup( {
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } }
    ]
  } );

  computePipeline = gpuDevice.createComputePipeline( {
    layout: gpuDevice.createPipelineLayout( { bindGroupLayouts: [ bindGroupLayout ] } ),
    compute: { module: shaderModule, entryPoint: 'main' }
  } );

  // after pipeline ready we can initialize UI handlers
  initUI();

  useGPU = true;
  document.getElementById( 'info-app' ).innerText = "WebGPU Running";
  return true;
}

function initThree ()
{
  scene = new THREE.Scene();
  scene.background = new THREE.Color( 0x000005 );

  camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 1, 10000 );
  camera.position.set( -500, 600, 1000 );

  renderer = new THREE.WebGLRenderer( { antialias: true } );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.setPixelRatio( window.devicePixelRatio );

  // FIX: Append to the specific container, not the body!
  const container = document.getElementById( 'canvas-container' );
  container.appendChild( renderer.domElement );

  controls = new OrbitControls( camera, renderer.domElement );
  controls.target.set( 500, 300, 300 );
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Visual Bounds
  const boxGeom = new THREE.BoxGeometry( SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z );
  const edges = new THREE.EdgesGeometry( boxGeom );
  const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x444444 } ) );
  line.position.set( 500, 300, 300 );
  scene.add( line );

  // Boids mesh
  createInstancedMesh();

  scene.add( new THREE.DirectionalLight( 0xffffff, 1 ), new THREE.AmbientLight( 0xffffff, 0.3 ) );

  window.addEventListener( 'resize', () =>
  {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
  } );
}

function createInstancedMesh() {
  if (boidInstancedMesh) {
    scene.remove(boidInstancedMesh);
    boidInstancedMesh.geometry.dispose();
    boidInstancedMesh.material.dispose();
  }
  const geometry = new THREE.ConeGeometry( 2, 6, 5 ).rotateX( Math.PI / 2 );
  const material = new THREE.MeshPhongMaterial( { color: 0x00ff88 } );
  boidInstancedMesh = new THREE.InstancedMesh( geometry, material, boidCount );
  boidInstancedMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
  scene.add( boidInstancedMesh );
}

function recreateBoids(newCount) {
  boidCount = newCount;
  initBoidBuffers(boidCount);
  createInstancedMesh();
  
  // Rebind the new buffer to the compute pipeline
  bindGroup = gpuDevice.createBindGroup( {
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } }
    ]
  } );
  
  isSimulationRunning = true;
  updateStartPauseButton();
}

function updateUniforms() {
  // read each field from UI inputs (IDs match below)
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
  // world_max remains static from SIMULATION_SIZE
  paramsArray[12] = SIMULATION_SIZE.x;
  paramsArray[13] = SIMULATION_SIZE.y;
  paramsArray[14] = SIMULATION_SIZE.z;
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
}

function initUI() {
  // populate inputs with current defaults
  document.getElementById('boid-count').value = boidCount;
  document.getElementById('separation').value = paramsArray[0];
  document.getElementById('align').value = paramsArray[1];
  document.getElementById('cohesion').value = paramsArray[2];
  document.getElementById('max_speed').value = paramsArray[3];
  document.getElementById('max_force').value = paramsArray[4];
  document.getElementById('sep_weight').value = paramsArray[5];
  document.getElementById('align_weight').value = paramsArray[6];
  document.getElementById('coh_weight').value = paramsArray[7];
  document.getElementById('margin').value = paramsArray[8];
  document.getElementById('turn_factor').value = paramsArray[9];

  document.getElementById('boid-count').addEventListener('change', e => {
    const n = parseInt(e.target.value);
    if (!isNaN(n) && n > 0) {
      recreateBoids(n);
    }
  });

  const inputs = ['separation','align','cohesion','max_speed','max_force','sep_weight','align_weight','coh_weight','margin','turn_factor'];
  inputs.forEach(id => {
    document.getElementById(id).addEventListener('input', updateUniforms);
  });

  // toggle panel collapse via bootstrap
  document.getElementById('toggle-panel').addEventListener('click', () => {
    const body = document.getElementById('settings-body');
    const bs = bootstrap.Collapse.getOrCreateInstance(body);
    bs.toggle();
  });

  // Start/Pause button
  document.getElementById('start-pause-btn').addEventListener('click', () => {
    isSimulationRunning = !isSimulationRunning;
    updateStartPauseButton();
  });
  
  updateStartPauseButton();
}

function frame ()
{
  requestAnimationFrame( frame );

  // update stats
  const now = performance.now();
  if ( lastFrameTime ) {
    const fps = 1000 / (now - lastFrameTime);
    document.getElementById('info-fps').innerText = `FPS: ${fps.toFixed(1)}`;
  }
  lastFrameTime = now;
  document.getElementById('info-boids').innerText = `Boids: ${boidCount}`;

  // Always update controls so the camera movement is fluid
  if ( controls ) controls.update();

  if ( useGPU && !isMapping && isSimulationRunning )
  {
    const commandEncoder = gpuDevice.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline( computePipeline );
    pass.setBindGroup( 0, bindGroup );
    pass.dispatchWorkgroups( Math.ceil( boidCount / WORKGROUP_SIZE ) );
    pass.end();

    commandEncoder.copyBufferToBuffer( boidBuffer, 0, stagingBuffer, 0, boidBuffer.size );
    gpuDevice.queue.submit( [ commandEncoder.finish() ] );

    isMapping = true;
    stagingBuffer.mapAsync( GPUMapMode.READ ).then( () =>
    {
      const data = new Float32Array( stagingBuffer.getMappedRange() );

      for ( let i = 0; i < boidCount; i++ )
      {
        const stride = i * 8;
        _pos.set( data[ stride ], data[ stride + 1 ], data[ stride + 2 ] );
        _vel.set( data[ stride + 4 ], data[ stride + 5 ], data[ stride + 6 ] );

        if ( _vel.lengthSq() > 0.01 )
        {
          _orient.setFromUnitVectors( _up, _vel.clone().normalize() );
        }

        _matrix.compose( _pos, _orient, { x: 1, y: 1, z: 1 } );
        boidInstancedMesh.setMatrixAt( i, _matrix );
      }

        boidInstancedMesh.instanceMatrix.needsUpdate = true; // update finished

      stagingBuffer.unmap();
      isMapping = false;
    } ).catch( () => { isMapping = false; } );
  }

  renderer.render( scene, camera );
}

function updateStartPauseButton() {
  const btn = document.getElementById('start-pause-btn');
  btn.textContent = isSimulationRunning ? 'Pause' : 'Start';
  btn.classList.toggle('btn-success', isSimulationRunning);
  btn.classList.toggle('btn-warning', !isSimulationRunning);
}

initWebGPU().then( () =>
{
  initThree();
  frame();
} );