import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const BOID_COUNT = 15000;
const WORKGROUP_SIZE = 256;
const SIMULATION_SIZE = { x: 1000, y: 600, z: 600 };

let scene, camera, renderer, boidInstancedMesh, controls;
let gpuDevice, computePipeline, boidBuffer, stagingBuffer, bindGroup;
let useGPU = false;
let isMapping = false;

// Pre-allocate math objects to save memory and CPU cycles
const _matrix = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _orient = new THREE.Quaternion();
const _vel = new THREE.Vector3();
const _up = new THREE.Vector3( 0, 0, 1 );

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

  const boidData = new Float32Array( BOID_COUNT * 8 );
  for ( let i = 0; i < BOID_COUNT; i++ )
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

  const bindGroupLayout = gpuDevice.createBindGroupLayout( {
    entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } } ]
  } );

  bindGroup = gpuDevice.createBindGroup( {
    layout: bindGroupLayout,
    entries: [ { binding: 0, resource: { buffer: boidBuffer } } ]
  } );

  computePipeline = gpuDevice.createComputePipeline( {
    layout: gpuDevice.createPipelineLayout( { bindGroupLayouts: [ bindGroupLayout ] } ),
    compute: { module: shaderModule, entryPoint: 'main' }
  } );

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

  // Boids
  const geometry = new THREE.ConeGeometry( 2, 6, 5 ).rotateX( Math.PI / 2 );
  const material = new THREE.MeshPhongMaterial( { color: 0x00ff88 } );
  boidInstancedMesh = new THREE.InstancedMesh( geometry, material, BOID_COUNT );
  boidInstancedMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
  scene.add( boidInstancedMesh );

  scene.add( new THREE.DirectionalLight( 0xffffff, 1 ), new THREE.AmbientLight( 0xffffff, 0.3 ) );

  window.addEventListener( 'resize', () =>
  {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
  } );
}

function frame ()
{
  requestAnimationFrame( frame );

  // Always update controls so the camera movement is fluid
  if ( controls ) controls.update();

  if ( useGPU && !isMapping )
  {
    const commandEncoder = gpuDevice.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline( computePipeline );
    pass.setBindGroup( 0, bindGroup );
    pass.dispatchWorkgroups( Math.ceil( BOID_COUNT / WORKGROUP_SIZE ) );
    pass.end();

    commandEncoder.copyBufferToBuffer( boidBuffer, 0, stagingBuffer, 0, boidBuffer.size );
    gpuDevice.queue.submit( [ commandEncoder.finish() ] );

    isMapping = true;
    stagingBuffer.mapAsync( GPUMapMode.READ ).then( () =>
    {
      const data = new Float32Array( stagingBuffer.getMappedRange() );

      for ( let i = 0; i < BOID_COUNT; i++ )
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

      boidInstancedMesh.instanceMatrix.needsUpdate = true;
      stagingBuffer.unmap();
      isMapping = false;
    } ).catch( () => { isMapping = false; } );
  }

  renderer.render( scene, camera );
}

initWebGPU().then( () =>
{
  initThree();
  frame();
} );