const BOID_COUNT = 15000;
const WORKGROUP_SIZE = 64;
const SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };

// Safe DOM update function
function updateStatus ( elementId, value )
{
  try
  {
    const el = document.getElementById( elementId );
    if ( el ) el.textContent = value;
  } catch ( e )
  {
    console.error( "DOM update error for " + elementId + ":", e );
  }
}

let scene, camera, renderer, boidMeshes = [];
let gpuDevice = null;
let computePipeline = null;
let boidBuffer = null;
let stagingBuffer = null;
let bindGroup = null;
let useGPU = false;
let frameCount = 0;
let lastFrameTime = performance.now();
let controls = null;
let COMPUTE_SHADER = null;

async function loadComputeShader ()
{
  try
  {
    const response = await fetch( 'compute-shader.wgsl' );
    if ( !response.ok )
    {
      throw new Error( 'Failed to load compute shader: ' + response.statusText );
    }
    COMPUTE_SHADER = await response.text();
    console.log( "Compute shader loaded" );
    return COMPUTE_SHADER;
  } catch ( e )
  {
    console.error( "Error loading compute shader:", e );
    return null;
  }
}

async function initWebGPU ()
{
  try
  {
    console.log( "Attempting WebGPU initialization..." );
    updateStatus( 'gpu-status', 'Checking...' );

    // Load the compute shader before proceeding
    if ( !COMPUTE_SHADER )
    {
      await loadComputeShader();
      if ( !COMPUTE_SHADER )
      {
        console.error( "Failed to load compute shader" );
        updateStatus( 'gpu-status', 'CPU Fallback' );
        return false;
      }
    }

    const adapter = await navigator.gpu?.requestAdapter();
    if ( !adapter )
    {
      console.warn( "WebGPU adapter not found" );
      updateStatus( 'gpu-status', 'CPU Fallback' );
      return false;
    }

    gpuDevice = await adapter.requestDevice();
    console.log( "Got GPU device" );

    const shaderModule = gpuDevice.createShaderModule( {
      code: COMPUTE_SHADER
    } );
    console.log( "Shader module created" );

    const boidData = new Float32Array( BOID_COUNT * 8 );
    for ( let i = 0; i < BOID_COUNT; i++ )
    {
      boidData[ i * 8 ] = Math.random() * SIMULATION_SIZE.x;
      boidData[ i * 8 + 1 ] = Math.random() * SIMULATION_SIZE.y;
      boidData[ i * 8 + 2 ] = Math.random() * SIMULATION_SIZE.z;
      boidData[ i * 8 + 3 ] = 1.0;
      boidData[ i * 8 + 4 ] = ( Math.random() - 0.5 ) * 2;
      boidData[ i * 8 + 5 ] = ( Math.random() - 0.5 ) * 2;
      boidData[ i * 8 + 6 ] = ( Math.random() - 0.5 ) * 2;
      boidData[ i * 8 + 7 ] = 1.0;
    }

    boidBuffer = gpuDevice.createBuffer( {
      size: boidData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    } );
    new Float32Array( boidBuffer.getMappedRange() ).set( boidData );
    boidBuffer.unmap();
    console.log( "Boid buffer created" );

    stagingBuffer = gpuDevice.createBuffer( {
      size: boidData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    } );

    const bindGroupLayout = gpuDevice.createBindGroupLayout( {
      entries: [ {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      } ]
    } );

    bindGroup = gpuDevice.createBindGroup( {
      layout: bindGroupLayout,
      entries: [ {
        binding: 0,
        resource: { buffer: boidBuffer }
      } ]
    } );
    console.log( "Bind group created" );

    const pipelineLayout = gpuDevice.createPipelineLayout( {
      bindGroupLayouts: [ bindGroupLayout ]
    } );

    computePipeline = gpuDevice.createComputePipeline( {
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'main' }
    } );
    console.log( "Compute pipeline created" );

    useGPU = true;
    updateStatus( 'gpu-status', 'WebGPU ✓' );
    console.log( "WebGPU fully initialized and ready to compute" );
    return true;
  } catch ( e )
  {
    console.error( "WebGPU initialization FAILED:", e.message );
    updateStatus( 'gpu-status', 'CPU Fallback' );
    return false;
  }
}

async function updateBoidsGPU ()
{
  if ( !gpuDevice || !computePipeline ) return;

  try
  {
    const commandEncoder = gpuDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline( computePipeline );
    passEncoder.setBindGroup( 0, bindGroup );

    const workgroups = Math.ceil( BOID_COUNT / WORKGROUP_SIZE );
    passEncoder.dispatchWorkgroups( workgroups );
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
      boidBuffer, 0,
      stagingBuffer, 0,
      boidBuffer.size
    );

    const commandBuffer = commandEncoder.finish();
    gpuDevice.queue.submit( [ commandBuffer ] );
  } catch ( e )
  {
    console.error( "GPU update error:", e );
  }
}

let readErrorCount = 0;
let pendingMapPromise = null;
let mapRequestFrame = -1;

function requestBoidsReadback ()
{
  if ( !gpuDevice || !stagingBuffer || pendingMapPromise )
  {
    return;
  }

  // Start async map without awaiting - GPU continues working
  pendingMapPromise = stagingBuffer.mapAsync( GPUMapMode.READ ).catch( e =>
  {
    if ( readErrorCount++ < 5 ) console.error( "GPU map request error:", e );
    pendingMapPromise = null;
  } );
  mapRequestFrame = performance.now();
}

function tryReadBoidsFromGPU ()
{
  if ( !pendingMapPromise )
  {
    return null;
  }

  try
  {
    // Check if map is ready (non-blocking)
    const boidData = new Float32Array( stagingBuffer.getMappedRange() ).slice();
    stagingBuffer.unmap();
    pendingMapPromise = null;

    if ( readErrorCount > 0 )
    {
      console.log( "GPU read successful, count=" + ( boidData.length / 8 ) + " boids" );
      readErrorCount = 0;
    }
    return boidData;
  } catch ( e )
  {
    // Map not ready yet, will try again next frame
    return null;
  }
}

function init ()
{
  console.log( "init() called, useGPU=", useGPU );

  scene = new THREE.Scene();
  scene.background = new THREE.Color( 0x000011 );

  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    10000
  );
  camera.position.set( -222, 717, 1340 );
  camera.lookAt( 500, 300, 300 );

  const container = document.getElementById( 'canvas-container' );
  renderer = new THREE.WebGLRenderer( { antialias: true } );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.shadowMap.enabled = true;
  container.appendChild( renderer.domElement );
  console.log( "WebGL renderer created" );

  controls = new THREE.OrbitControls( camera, renderer.domElement );
  controls.target.set( 500, 300, 300 );
  controls.autoRotate = false;
  controls.autoRotateSpeed = 1;
  controls.update();

  const ambientLight = new THREE.AmbientLight( 0xffffff, 0.6 );
  scene.add( ambientLight );

  const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.8 );
  directionalLight.position.set( 500, 500, 500 );
  directionalLight.castShadow = true;
  scene.add( directionalLight );

  const boxGeometry = new THREE.BoxGeometry( SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z );
  const boxMaterial = new THREE.LineBasicMaterial( { color: 0x00ff88 } );
  const boxEdges = new THREE.EdgesGeometry( boxGeometry );
  const boundingBox = new THREE.LineSegments( boxEdges, boxMaterial );
  boundingBox.position.set( 500, 300, 300 );
  scene.add( boundingBox );
  console.log( "Scene geometry added" );

  const coneGeometry = new THREE.ConeGeometry( 2, 6, 8 );
  coneGeometry.rotateX( Math.PI / 2 ); // Rotate 90° to point forward (negative Z)
  const boidMaterial = new THREE.MeshPhongMaterial( {
    color: 0x00ff88,
    shininess: 100
  } );

  // Use InstancedMesh for better performance
  const boidInstancedMesh = new THREE.InstancedMesh( coneGeometry, boidMaterial, BOID_COUNT );
  boidInstancedMesh.castShadow = true;

  scene.add( boidInstancedMesh );
  boidMeshes.push( boidInstancedMesh ); // Keep for compatibility
  console.log( "Created InstancedMesh with", BOID_COUNT, "boid instances" );

  let simCounter = 0;
  let currentFPS = 0;
  let lastBoidData = null;
  let prevBoidData = null;
  let firstFrame = true;
  let animateRunCount = 0;
  const READBACK_FREQUENCY = 2; // Request readback every N frames, read on next frame
  const velVector = new THREE.Vector3(); // Reuse vector to avoid allocations
  const matrix = new THREE.Matrix4(); // Reuse matrix for instance updates
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3( 1, 1, 1 ); // Reuse scale vector
  const dir = new THREE.Vector3(); // Reuse direction vector
  const prevPos = new THREE.Vector3(); // Reuse previous position vector
  const zAxis = new THREE.Vector3( 0, 0, 1 ); // Default forward direction

  function animate ()
  {
    requestAnimationFrame( animate );
    animateRunCount++;

    frameCount++;
    const now = performance.now();
    if ( now >= lastFrameTime + 1000 )
    {
      currentFPS = Math.round( ( frameCount * 1000 ) / ( now - lastFrameTime ) );
      frameCount = 0;
      lastFrameTime = now;
    }

    // Log first frame
    if ( firstFrame )
    {
      console.log( "First animate frame rendered" );
      firstFrame = false;
    }

    updateBoidsGPU();

    // Asynchronous readback: request on odd frames, read on even frames
    // This allows GPU to continue working while we wait for data
    if ( animateRunCount % READBACK_FREQUENCY === 1 )
    {
      requestBoidsReadback();
    } else if ( animateRunCount % READBACK_FREQUENCY === 0 )
    {
      const data = tryReadBoidsFromGPU();
      if ( data && boidMeshes.length > 0 )
      {
        prevBoidData = lastBoidData;
        lastBoidData = data;
      }
    }

    // Update instances every frame using interpolation between last two readbacks
    if ( lastBoidData && boidMeshes.length > 0 )
    {
      const boidInstancedMesh = boidMeshes[ 0 ];
      const interpolationPhase = ( animateRunCount % READBACK_FREQUENCY ) / READBACK_FREQUENCY;

      for ( let i = 0; i < BOID_COUNT; i++ )
      {
        const pos = i * 8;
        position.set( lastBoidData[ pos ], lastBoidData[ pos + 1 ], lastBoidData[ pos + 2 ] );

        // If we have previous data, interpolate between old and new positions
        if ( prevBoidData )
        {
          prevPos.set( prevBoidData[ pos ], prevBoidData[ pos + 1 ], prevBoidData[ pos + 2 ] );
          position.lerpVectors( prevPos, position, interpolationPhase );
        }

        // Calculate rotation from velocity
        velVector.set( lastBoidData[ pos + 4 ], lastBoidData[ pos + 5 ], lastBoidData[ pos + 6 ] );
        if ( velVector.length() > 0.1 )
        {
          dir.copy( velVector ).normalize();
          quaternion.setFromUnitVectors( zAxis, dir );
        } else
        {
          quaternion.identity();
        }

        matrix.compose( position, quaternion, scale );
        boidInstancedMesh.setMatrixAt( i, matrix );
      }
      boidInstancedMesh.instanceMatrix.needsUpdate = true;
    }

    if ( controls ) controls.update();
    renderer.render( scene, camera );

    // Update debug state display
    if ( animateRunCount % 60 === 0 )
    {  // Update state display every 60 frames (~1 sec at 60fps)
      document.getElementById( 'info-fps' ).textContent = 'FPS:' + currentFPS;
      document.getElementById( 'info-boids' ).textContent = 'Boids:' + BOID_COUNT;
      document.getElementById( 'info-step' ).textContent = 'Step:' + animateRunCount;
      document.getElementById( 'info-gpu' ).textContent = 'GPU:' + ( useGPU ? 'Yes' : 'No' );
    }
  }

  animate();
  console.log( "Boid renderer started with", BOID_COUNT, "boids, WebGPU=" + useGPU );

  // Expose debug info globally
  window.boidsDebug = {
    boidCount: BOID_COUNT,
    boidMeshesCreated: boidMeshes.length,
    useGPU: useGPU,
    gpuDevice: !!gpuDevice,
    computePipeline: !!computePipeline,
    boidBuffer: !!boidBuffer,
    stagingBuffer: !!stagingBuffer,
    bindGroup: !!bindGroup,
    animationRunning: true
  };
  console.log( "Debug info exposed as window.boidsDebug" );
}

async function startSimulation ()
{
  console.log( "Simulation started, useGPU:", useGPU );
}

window.addEventListener( 'resize', () =>
{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize( window.innerWidth, window.innerHeight );
} );

// Start initialization
console.log( "=== Boids Application Starting ===" );
document.getElementById( 'info-app' ).textContent = 'Initializing WebGPU...';

initWebGPU().then( success =>
{
  console.log( "WebGPU initialization:", success ? "SUCCESS ✓" : "FAILED - using fallback" );
  document.getElementById( 'info-app' ).textContent = success ? 'WebGPU Ready' : 'CPU Mode';

  init();
  document.getElementById( 'info-app' ).textContent = 'Running';
  console.log( "=== Application Ready ===" );
} ).catch( err =>
{
  console.error( "FATAL ERROR:", err );
  document.getElementById( 'info-app' ).textContent = 'ERROR!';
  document.getElementById( 'info-gpu' ).textContent = 'Error';
  try
  {
    init();
  } catch ( e )
  {
    console.error( "Init also failed:", e );
  }
} );
