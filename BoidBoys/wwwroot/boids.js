const BOID_COUNT = 1000;
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

const COMPUTE_SHADER = `
struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  // Craig Reynolds parameters
  let separation_dist = 25.0;
  let align_dist = 50.0;
  let cohesion_dist = 50.0;
  let max_speed = 5.0;
  let max_force = 0.05;

  let separation_weight = 1.5;
  let alignment_weight = 1.0;
  let cohesion_weight = 1.0;

  var sep_sum = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum = vec3<f32>(0.0);
  var count_sep: u32 = 0u;
  var count_align: u32 = 0u;
  var count_coh: u32 = 0u;

  let my_pos = boids[idx].position.xyz;
  let my_vel = boids[idx].velocity.xyz;

  // Accumulate neighbor info
  for (var i = 0u; i < total; i = i + 1u) {
    if (i == idx) { continue; }
    let other_pos = boids[i].position.xyz;
    let other_vel = boids[i].velocity.xyz;
    let dist = distance(my_pos, other_pos);

    if (dist < separation_dist && dist > 0.0) {
      // stronger repulsion when closer
      let diff = my_pos - other_pos;
      sep_sum = sep_sum + (normalize(diff) / dist);
      count_sep = count_sep + 1u;
    }

    if (dist < align_dist && dist > 0.0) {
      align_sum = align_sum + other_vel;
      count_align = count_align + 1u;
    }

    if (dist < cohesion_dist && dist > 0.0) {
      coh_sum = coh_sum + other_pos;
      count_coh = count_coh + 1u;
    }
  }

  // Steering helpers
  var steer_sep = vec3<f32>(0.0);
  if (count_sep > 0u) {
    let avg = sep_sum / f32(count_sep);
    if (length(avg) > 0.0) {
      let desired = normalize(avg) * max_speed;
      steer_sep = desired - my_vel;
      if (length(steer_sep) > max_force) {
        steer_sep = normalize(steer_sep) * max_force;
      }
    }
  }

  var steer_align = vec3<f32>(0.0);
  if (count_align > 0u) {
    let avg_vel = align_sum / f32(count_align);
    if (length(avg_vel) > 0.0) {
      let desired = normalize(avg_vel) * max_speed;
      steer_align = desired - my_vel;
      if (length(steer_align) > max_force) {
        steer_align = normalize(steer_align) * max_force;
      }
    }
  }

  var steer_coh = vec3<f32>(0.0);
  if (count_coh > 0u) {
    let center = coh_sum / f32(count_coh);
    let to_center = center - my_pos;
    if (length(to_center) > 0.0) {
      let desired = normalize(to_center) * max_speed;
      steer_coh = desired - my_vel;
      if (length(steer_coh) > max_force) {
        steer_coh = normalize(steer_coh) * max_force;
      }
    }
  }

  // Combine Reynolds steering forces
  var accel = steer_sep * separation_weight + steer_align * alignment_weight + steer_coh * cohesion_weight;

  var new_vel = my_vel + accel;

  // Boundary avoidance (steer away when inside margin)
  let margin = 50.0;
  let world_max_x = 1000.0;
  let world_max_y = 600.0;
  let world_max_z = 600.0;
  let wall_force = 0.5;

  var avoidance = vec3<f32>(0.0);
  if ((my_pos.x) < margin) {
    avoidance.x = (margin - my_pos.x) * wall_force;
  } else if ((my_pos.x) > (world_max_x - margin)) {
    avoidance.x = (world_max_x - margin - my_pos.x) * wall_force;
  }
  if ((my_pos.y) < margin) {
    avoidance.y = (margin - my_pos.y) * wall_force;
  } else if ((my_pos.y) > (world_max_y - margin)) {
    avoidance.y = (world_max_y - margin - my_pos.y) * wall_force;
  }
  if ((my_pos.z) < margin) {
    avoidance.z = (margin - my_pos.z) * wall_force;
  } else if ((my_pos.z) > (world_max_z - margin)) {
    avoidance.z = (world_max_z - margin - my_pos.z) * wall_force;
  }

  new_vel = new_vel + avoidance;

  // Clamp speed
  let sp = length(new_vel);
  if (sp > max_speed) {
    new_vel = normalize(new_vel) * max_speed;
  }

  let new_pos = my_pos + new_vel;

  boids[idx].position = vec4<f32>(new_pos, 1.0);
  boids[idx].velocity = vec4<f32>(new_vel, 1.0);
}`;

async function initWebGPU ()
{
  try
  {
    console.log( "Attempting WebGPU initialization..." );
    updateStatus( 'gpu-status', 'Checking...' );

    const adapter = await navigator.gpu?.requestAdapter();
    if ( !adapter )
    {
      console.warn( "WebGPU adapter not found" );
      updateStatus( 'gpu-status', 'CPU Fallback' );
      return false;
    }

    gpuDevice = await adapter.requestDevice();
    const queue = gpuDevice.queue;
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
async function readBoidsFromGPU ()
{
  if ( !gpuDevice || !stagingBuffer )
  {
    if ( readErrorCount++ < 5 ) console.warn( "GPU read skipped: device=" + !!gpuDevice + " buffer=" + !!stagingBuffer );
    return null;
  }

  try
  {
    await stagingBuffer.mapAsync( GPUMapMode.READ );
    const boidData = new Float32Array( stagingBuffer.getMappedRange() ).slice();
    stagingBuffer.unmap();
    if ( readErrorCount > 0 )
    {
      console.log( "GPU read successful, count=" + ( boidData.length / 8 ) + " boids" );
      readErrorCount = 0;
    }
    return boidData;
  } catch ( e )
  {
    if ( readErrorCount++ < 5 ) console.error( "GPU read error:", e );
    return null;
  }
}

function init ()
{
  console.log( "init() called, useGPU=", useGPU );
  document.getElementById( 'app-status' ).textContent = 'Initializing scene...';
  document.getElementById( 'debug-init' ).textContent = 'running...';

  scene = new THREE.Scene();
  scene.background = new THREE.Color( 0x000011 );

  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    10000
  );
  camera.position.set( 500, 300, 500 );
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
  const boidMaterial = new THREE.MeshPhongMaterial( {
    color: 0x00ff88,
    emissive: 0x0088ff,
    shininess: 100
  } );

  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    const boid = new THREE.Mesh( coneGeometry, boidMaterial.clone() );
    boid.castShadow = true;
    scene.add( boid );
    boidMeshes.push( boid );
  }
  console.log( "Created", BOID_COUNT, "boid meshes" );
  document.getElementById( 'boid-count' ).textContent = BOID_COUNT;
  document.getElementById( 'debug-init' ).textContent = 'meshes OK';

  let simCounter = 0;
  let lastBoidData = null;
  let firstFrame = true;
  let animateRunCount = 0;

  function animate ()
  {
    requestAnimationFrame( animate );
    animateRunCount++;

    frameCount++;
    const now = performance.now();
    if ( now >= lastFrameTime + 1000 )
    {
      document.getElementById( 'fps' ).textContent = frameCount;
      document.getElementById( 'debug-animate' ).textContent = 'run:' + animateRunCount + ' fps:' + frameCount;
      frameCount = 0;
      lastFrameTime = now;
    }

    // Log first frame
    if ( firstFrame )
    {
      console.log( "First animate frame rendered" );
      document.getElementById( 'app-status' ).textContent = 'Rendering...';
      document.getElementById( 'debug-init' ).textContent = 'running OK';
      firstFrame = false;
    }

    // Update simulation every 2 frames
    simCounter++;
    if ( simCounter % 2 === 0 && useGPU )
    {
      updateBoidsGPU();
      readBoidsFromGPU().then( data =>
      {
        if ( data && boidMeshes.length > 0 )
        {
          lastBoidData = data;
          for ( let i = 0; i < Math.min( BOID_COUNT, boidMeshes.length ); i++ )
          {
            const mesh = boidMeshes[ i ];
            const pos = i * 8;
            mesh.position.set( data[ pos ], data[ pos + 1 ], data[ pos + 2 ] );

            const vel = new THREE.Vector3( data[ pos + 4 ], data[ pos + 5 ], data[ pos + 6 ] );
            if ( vel.length() > 0.1 )
            {
              mesh.lookAt(
                mesh.position.x + vel.x,
                mesh.position.y + vel.y,
                mesh.position.z + vel.z
              );
            }
          }
        }
      } ).catch( e =>
      {
        console.error( "GPU readback error:", e );
        document.getElementById( 'app-status' ).textContent = 'GPU Error';
      } );
    }

    if ( controls ) controls.update();
    renderer.render( scene, camera );

    // Update debug state display
    if ( animateRunCount % 60 === 0 )
    {  // Update state display every 60 frames (~1 sec at 60fps)
      document.getElementById( 'debug-state' ).textContent =
        'GPU:' + ( useGPU ? 'Y' : 'N' ) +
        ' Meshes:' + boidMeshes.length +
        ' Dev:' + ( gpuDevice ? '✓' : '-' ) +
        ' Pipe:' + ( computePipeline ? '✓' : '-' );
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
document.getElementById( 'app-status' ).textContent = 'Initializing WebGPU...';

initWebGPU().then( success =>
{
  console.log( "WebGPU initialization:", success ? "SUCCESS ✓" : "FAILED - using fallback" );
  document.getElementById( 'app-status' ).textContent = success ? 'WebGPU Ready' : 'CPU Mode';

  init();
  document.getElementById( 'app-status' ).textContent = 'Running';
  console.log( "=== Application Ready ===" );
} ).catch( err =>
{
  console.error( "FATAL ERROR:", err );
  document.getElementById( 'app-status' ).textContent = 'ERROR!';
  document.getElementById( 'gpu-status' ).textContent = 'Error';
  try
  {
    init();
  } catch ( e )
  {
    console.error( "Init also failed:", e );
  }
} );
