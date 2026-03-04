import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SimulationUI } from './ui.js';

let boidCount = 15000;
const WORKGROUP_SIZE = 256;
let SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };
let boidDensity = 0.000025; // boids per cubic unit

// Base aspect ratio for simulation size
const BASE_SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };

// buffers & pipeline globals so we can rebind when settings change
let bindGroupLayout, uniformBuffer;

// Create a small uniform buffer holding global scene/simulation parameters
const paramsArray = new Float32Array(16); // 16 floats = 64 bytes
// layout: separation, align, cohesion, max_speed,
//         max_force, sep_w, align_w, coh_w,
//         margin, turn_factor, padding0, padding1,
//         world_max.x, world_max.y, world_max.z, world_max.w
function calculateSimulationSize(count, density) {
  const baseVolume = BASE_SIMULATION_SIZE.x * BASE_SIMULATION_SIZE.y * BASE_SIMULATION_SIZE.z;
  const requiredVolume = count / density;
  const scaleFactor = Math.cbrt(requiredVolume / baseVolume);
  
  return {
    x: BASE_SIMULATION_SIZE.x * scaleFactor,
    y: BASE_SIMULATION_SIZE.y * scaleFactor,
    z: BASE_SIMULATION_SIZE.z * scaleFactor
  };
}

function updateVisualBounds() {
  // Remove old visual bounds
  const oldBox = scene.getObjectByName('boid-bounds');
  if (oldBox) scene.remove(oldBox);
  
  // Create new visual bounds with updated size
  const boxGeom = new THREE.BoxGeometry( SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z );
  const edges = new THREE.EdgesGeometry( boxGeom );
  const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x444444 } ) );
  line.name = 'boid-bounds';
  line.position.set( SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2 );
  scene.add( line );
}

function resetParamsToDefaults() {
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
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
let simUI = null; // Will be initialized after DOM is ready
let lastFrameTime = Date.now();
let frameCount = 0;
let stepCount = 0;
let isSimulationRunning = true;

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
    if ( simUI ) simUI.setInfo( 'info-app', "WebGPU not supported" );
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

  useGPU = true;
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
  line.name = 'boid-bounds';
  line.position.set( SIMULATION_SIZE.x / 2, SIMULATION_SIZE.y / 2, SIMULATION_SIZE.z / 2 );
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

function initSimulationUI() {
  const buttonConfig = {
    'play-pause': {
      // alternate icons depending on state will be managed by setButtonState
      iconStates: { playing: '⏸', paused: '▶' },
      onClick: () => {
        isSimulationRunning = !isSimulationRunning;
        simUI.setButtonState('play-pause', isSimulationRunning ? 'playing' : 'paused');
      }
    },
    'restart': {
      icon: '↻',
      onClick: () => {
        initBoidBuffers(boidCount);
        createInstancedMesh();
        bindGroup = gpuDevice.createBindGroup( {
          layout: bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: boidBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } }
          ]
        } );
        isSimulationRunning = true;
        simUI.setButtonState('play-pause', 'playing');
      }
    },
    'reset': {
      icon: '↺',
      onClick: () => {
        boidCount = 15000;
        boidDensity = 0.00005;
        SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
        resetParamsToDefaults();
        recreateBoids(boidCount);
        isSimulationRunning = true;
        simUI.setButtonState('play-pause', 'playing');
      }
    }
  };

  const paramConfig = {
    'boid-count': {
      value: boidCount,
      min: 100,
      max: 50000,
      step: 100,
      decimals: 0,
      onChange: (val) => {
        const n = parseInt(val);
        if (!isNaN(n) && n > 0) {
          recreateBoids(n);
        }
      }
    },
    'boid-density': {
      value: boidDensity,
      min: 0.00001,
      max: 0.0001,
      step: 0.000001,
      decimals: 6,
      onChange: (val) => {
        const d = parseFloat(val);
        if (!isNaN(d) && d > 0) {
          boidDensity = d;
          SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
          resetParamsToDefaults();
          updateVisualBounds();
          gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
        }
      }
    },
    'separation': {
      value: paramsArray[0],
      min: 5,
      max: 100,
      step: 1,
      decimals: 0,
      onChange: (val) => { paramsArray[0] = parseFloat(val); updateUniforms(); }
    },
    'align': {
      value: paramsArray[1],
      min: 5,
      max: 150,
      step: 1,
      decimals: 0,
      onChange: (val) => { paramsArray[1] = parseFloat(val); updateUniforms(); }
    },
    'cohesion': {
      value: paramsArray[2],
      min: 5,
      max: 150,
      step: 1,
      decimals: 0,
      onChange: (val) => { paramsArray[2] = parseFloat(val); updateUniforms(); }
    },
    'max-speed': {
      value: paramsArray[3],
      min: 1,
      max: 20,
      step: 0.1,
      decimals: 1,
      onChange: (val) => { paramsArray[3] = parseFloat(val); updateUniforms(); }
    },
    'max-force': {
      value: paramsArray[4],
      min: 0.01,
      max: 1,
      step: 0.01,
      decimals: 2,
      onChange: (val) => { paramsArray[4] = parseFloat(val); updateUniforms(); }
    },
    'sep-weight': {
      value: paramsArray[5],
      min: 0.1,
      max: 5,
      step: 0.1,
      decimals: 1,
      onChange: (val) => { paramsArray[5] = parseFloat(val); updateUniforms(); }
    },
    'align-weight': {
      value: paramsArray[6],
      min: 0.1,
      max: 5,
      step: 0.1,
      decimals: 1,
      onChange: (val) => { paramsArray[6] = parseFloat(val); updateUniforms(); }
    },
    'coh-weight': {
      value: paramsArray[7],
      min: 0.1,
      max: 5,
      step: 0.1,
      decimals: 1,
      onChange: (val) => { paramsArray[7] = parseFloat(val); updateUniforms(); }
    },
    'margin': {
      value: paramsArray[8],
      min: 10,
      max: 500,
      step: 10,
      decimals: 0,
      onChange: (val) => { paramsArray[8] = parseFloat(val); updateUniforms(); }
    },
    'turn-factor': {
      value: paramsArray[9],
      min: 0.01,
      max: 1,
      step: 0.01,
      decimals: 2,
      onChange: (val) => { paramsArray[9] = parseFloat(val); updateUniforms(); }
    }
  };

  simUI = new SimulationUI(paramConfig, buttonConfig);
  simUI.setInfo('info-app', 'WebGPU Running');
  simUI.setInfo('info-gpu', 'Press P to toggle params');
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
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
  resetParamsToDefaults();
  
  initBoidBuffers(boidCount);
  createInstancedMesh();
  updateVisualBounds();
  
  // Rebind the new buffer to the compute pipeline
  bindGroup = gpuDevice.createBindGroup( {
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: boidBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } }
    ]
  } );
  
  isSimulationRunning = true;
}

function updateUniforms() {
  // Update uniforms from the simUI sliders
  paramsArray[12] = SIMULATION_SIZE.x;
  paramsArray[13] = SIMULATION_SIZE.y;
  paramsArray[14] = SIMULATION_SIZE.z;
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
}

function frame ()
{
  requestAnimationFrame( frame );

  // update stats
  const now = performance.now();
  if ( lastFrameTime ) {
    const fps = 1000 / (now - lastFrameTime);
    simUI.setInfo('info-fps', `FPS: ${fps.toFixed(1)}`);
  }
  lastFrameTime = now;
  simUI.setInfo('info-boids', `Boids: ${boidCount}`);

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

function resetSimulation() {
  boidCount = 15000;
  boidDensity = 0.00005;
  SIMULATION_SIZE = calculateSimulationSize(boidCount, boidDensity);
  resetParamsToDefaults();
  recreateBoids(boidCount);
  
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, paramsArray.buffer, paramsArray.byteOffset, paramsArray.byteLength);
}

initWebGPU().then( () =>
{
  initThree();
  initSimulationUI();
  // initialize button visuals (especially play/pause icon)
  simUI.setButtonState('play-pause', isSimulationRunning ? 'playing' : 'paused');
  frame();
} );