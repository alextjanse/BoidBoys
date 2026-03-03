import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SimulationControls } from './controls.js';

// Mutable parameters (can be changed via UI)
let BOID_COUNT = 3000;
let SEPARATION_DISTANCE = 40;
let ALIGNMENT_DISTANCE = 60;
let COHESION_DISTANCE = 100;
let MAX_SPEED = 4;
let MAX_FORCE = 0.1;
let K_NEIGHBORS = 20;

const SIMULATION_SIZE = { x: 1000, y: 600, z: 1 }; // z=1 for 2D plane thickness

let scene, camera, renderer, boidInstancedMesh, controls;
let boidPositions, boidVelocities, boidAccelerations;
let nearestNeighbors, nearestNeighborsDist; // Store k nearest neighbors and their distances
let nearestNeighbors_verify, nearestNeighborsDist_verify; // Verification arrays for brute force
let simControls = null; // Will be initialized after DOM is ready
let verificationEnabled = true; // Toggle verification on/off
let verificationErrors = 0;
let lastFrameTime = Date.now();
let frameCount = 0;

// Pre-allocate math objects to save memory and CPU cycles
const _matrix = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _orient = new THREE.Quaternion();
const _vel = new THREE.Vector3();
const _up = new THREE.Vector3( 0, 0, 1 );

function initBoids ()
{
  boidPositions = new Float32Array( BOID_COUNT * 3 );
  boidVelocities = new Float32Array( BOID_COUNT * 3 );
  boidAccelerations = new Float32Array( BOID_COUNT * 3 );
  nearestNeighbors = new Int32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighborsDist = new Float32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighbors_verify = new Int32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighborsDist_verify = new Float32Array( BOID_COUNT * K_NEIGHBORS );

  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    boidPositions[ i * 3 ] = Math.random() * SIMULATION_SIZE.x;
    boidPositions[ i * 3 + 1 ] = Math.random() * SIMULATION_SIZE.y;
    boidPositions[ i * 3 + 2 ] = 0.5;

    boidVelocities[ i * 3 ] = ( Math.random() - 0.5 ) * 2;
    boidVelocities[ i * 3 + 1 ] = ( Math.random() - 0.5 ) * 2;
    boidVelocities[ i * 3 + 2 ] = 0;

    boidAccelerations[ i * 3 ] = 0;
    boidAccelerations[ i * 3 + 1 ] = 0;
    boidAccelerations[ i * 3 + 2 ] = 0;

    // Initialize neighbors with -1 (invalid)
    for ( let j = 0; j < K_NEIGHBORS; j++ )
    {
      nearestNeighbors[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist[ i * K_NEIGHBORS + j ] = Infinity;
      nearestNeighbors_verify[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist_verify[ i * K_NEIGHBORS + j ] = Infinity;
    }
  }
}

function limit ( value, max )
{
  if ( value > max ) return max;
  if ( value < -max ) return -max;
  return value;
}

function distSq ( dx, dy )
{
  return dx * dx + dy * dy;
}

/**
 * Brute force kNN - O(n²) per frame
 * Checks all boids to find k nearest neighbors
 */
function kNN_brute ()
{
  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    const x = boidPositions[ i * 3 ];
    const y = boidPositions[ i * 3 + 1 ];

    // Reset distances to infinity, indices to -1
    for ( let j = 0; j < K_NEIGHBORS; j++ )
    {
      nearestNeighbors[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist[ i * K_NEIGHBORS + j ] = Infinity;
    }

    // Check all other boids
    for ( let j = 0; j < BOID_COUNT; j++ )
    {
      if ( i === j ) continue;

      const otherX = boidPositions[ j * 3 ];
      const otherY = boidPositions[ j * 3 + 1 ];
      const dx = x - otherX;
      const dy = y - otherY;
      const d2 = distSq( dx, dy );

      // Insert into sorted list if closer than current k-th neighbor
      if ( d2 < nearestNeighborsDist[ i * K_NEIGHBORS + K_NEIGHBORS - 1 ] )
      {
        // Find insertion point (scan from back to front, sorted ascending)
        let insertPos = K_NEIGHBORS - 1;
        while ( insertPos > 0 && d2 < nearestNeighborsDist[ i * K_NEIGHBORS + insertPos - 1 ] )
        {
          insertPos--;
        }

        // Shift elements to the right and insert
        for ( let k = K_NEIGHBORS - 1; k > insertPos; k-- )
        {
          nearestNeighbors[ i * K_NEIGHBORS + k ] = nearestNeighbors[ i * K_NEIGHBORS + k - 1 ];
          nearestNeighborsDist[ i * K_NEIGHBORS + k ] = nearestNeighborsDist[ i * K_NEIGHBORS + k - 1 ];
        }

        nearestNeighbors[ i * K_NEIGHBORS + insertPos ] = j;
        nearestNeighborsDist[ i * K_NEIGHBORS + insertPos ] = d2;
      }
    }
  }
}

/**
 * Optimized kNN using temporal coherence - O(nk²) per frame
 * Only searches among previous frame's k neighbors and their k neighbors
 */
function kNN_optimized ()
{
  // Candidate set for each boid
  const candidates = new Set();

  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    const x = boidPositions[ i * 3 ];
    const y = boidPositions[ i * 3 + 1 ];

    candidates.clear();
    candidates.add( i ); // Add self to avoid checking

    // Gather candidates: previous k neighbors + their k neighbors
    for ( let j = 0; j < K_NEIGHBORS; j++ )
    {
      const nbIdx = nearestNeighbors[ i * K_NEIGHBORS + j ];
      if ( nbIdx === -1 ) break; // No more valid neighbors

      candidates.add( nbIdx );

      // Add neighbors of neighbors
      for ( let k = 0; k < K_NEIGHBORS; k++ )
      {
        const nbNbIdx = nearestNeighbors[ nbIdx * K_NEIGHBORS + k ];
        if ( nbNbIdx === -1 ) break;
        candidates.add( nbNbIdx );
      }
    }

    // Reset for this boid
    for ( let j = 0; j < K_NEIGHBORS; j++ )
    {
      nearestNeighbors[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist[ i * K_NEIGHBORS + j ] = Infinity;
    }

    // Search among candidates only
    for ( const j of candidates )
    {
      if ( i === j ) continue;

      const otherX = boidPositions[ j * 3 ];
      const otherY = boidPositions[ j * 3 + 1 ];
      const dx = x - otherX;
      const dy = y - otherY;
      const d2 = distSq( dx, dy );

      // Insert into sorted list if closer than current k-th neighbor
      if ( d2 < nearestNeighborsDist[ i * K_NEIGHBORS + K_NEIGHBORS - 1 ] )
      {
        // Find insertion point (scan from back to front, sorted ascending)
        let insertPos = K_NEIGHBORS - 1;
        while ( insertPos > 0 && d2 < nearestNeighborsDist[ i * K_NEIGHBORS + insertPos - 1 ] )
        {
          insertPos--;
        }

        // Shift elements to the right and insert
        for ( let k = K_NEIGHBORS - 1; k > insertPos; k-- )
        {
          nearestNeighbors[ i * K_NEIGHBORS + k ] = nearestNeighbors[ i * K_NEIGHBORS + k - 1 ];
          nearestNeighborsDist[ i * K_NEIGHBORS + k ] = nearestNeighborsDist[ i * K_NEIGHBORS + k - 1 ];
        }

        nearestNeighbors[ i * K_NEIGHBORS + insertPos ] = j;
        nearestNeighborsDist[ i * K_NEIGHBORS + insertPos ] = d2;
      }
    }
  }
}

/**
 * Compares optimized vs brute force results
 * Returns true if they match, false otherwise
 */
function verifyKNN ()
{
  let discrepancies = 0;
  let totalChecked = 0;

  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    for ( let k = 0; k < K_NEIGHBORS; k++ )
    {
      totalChecked++;

      const optIdx = nearestNeighbors[ i * K_NEIGHBORS + k ];
      const brutIdx = nearestNeighbors_verify[ i * K_NEIGHBORS + k ];
      const optDist = nearestNeighborsDist[ i * K_NEIGHBORS + k ];
      const brutDist = nearestNeighborsDist_verify[ i * K_NEIGHBORS + k ];

      // Check if indices match
      if ( optIdx !== brutIdx )
      {
        discrepancies++;
      }

      // Check if distances are very close (allow small floating point error)
      if ( !( ( Math.abs( optDist - brutDist ) < 0.001 || ( optDist === Infinity && brutDist === Infinity ) ) ) )
      {
        discrepancies++;
      }
    }
  }

  const matchPercentage = ( ( totalChecked - discrepancies ) / totalChecked * 100 ).toFixed( 2 );
  if ( discrepancies > 0 )
  {
    verificationErrors++;
    console.warn( `Verification mismatch: ${discrepancies} / ${totalChecked} (${matchPercentage}% match)` );
  }

  return discrepancies === 0;
}

function updateBoids ()
{
  // Compute k nearest neighbors using both algorithms for verification
  if ( verificationEnabled )
  {
    // Run brute force and store in verify arrays
    kNN_brute();
    // Copy results to verify arrays
    nearestNeighbors_verify.set( nearestNeighbors );
    nearestNeighborsDist_verify.set( nearestNeighborsDist );

    // Run optimized algorithm
    kNN_optimized();

    // Compare results
    verifyKNN();
  }
  else
  {
    // Just use optimized
    kNN_optimized();
  }

  // Reset accelerations
  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    boidAccelerations[ i * 3 ] = 0;
    boidAccelerations[ i * 3 + 1 ] = 0;
    boidAccelerations[ i * 3 + 2 ] = 0;
  }

  // Boid forces using precomputed neighbors
  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    const x = boidPositions[ i * 3 ];
    const y = boidPositions[ i * 3 + 1 ];
    const vx = boidVelocities[ i * 3 ];
    const vy = boidVelocities[ i * 3 + 1 ];

    let separationX = 0, separationY = 0;
    let alignmentX = 0, alignmentY = 0;
    let cohesionX = 0, cohesionY = 0;
    let separationCount = 0, alignmentCount = 0, cohesionCount = 0;

    // Iterate through k nearest neighbors
    for ( let idx = 0; idx < K_NEIGHBORS; idx++ )
    {
      const j = nearestNeighbors[ i * K_NEIGHBORS + idx ];
      if ( j === -1 ) break; // No more valid neighbors

      const otherX = boidPositions[ j * 3 ];
      const otherY = boidPositions[ j * 3 + 1 ];
      const dx = x - otherX;
      const dy = y - otherY;
      const d2 = distSq( dx, dy );

      // Separation
      if ( d2 < SEPARATION_DISTANCE * SEPARATION_DISTANCE && d2 > 0 )
      {
        separationX += dx / Math.sqrt( d2 );
        separationY += dy / Math.sqrt( d2 );
        separationCount++;
      }

      // Alignment
      if ( d2 < ALIGNMENT_DISTANCE * ALIGNMENT_DISTANCE )
      {
        alignmentX += boidVelocities[ j * 3 ];
        alignmentY += boidVelocities[ j * 3 + 1 ];
        alignmentCount++;
      }

      // Cohesion
      if ( d2 < COHESION_DISTANCE * COHESION_DISTANCE )
      {
        cohesionX += otherX;
        cohesionY += otherY;
        cohesionCount++;
      }
    }

    // Average and apply weights
    let forceX = 0, forceY = 0;

    if ( separationCount > 0 )
    {
      separationX /= separationCount;
      separationY /= separationCount;
      const sepMag = Math.sqrt( separationX * separationX + separationY * separationY );
      if ( sepMag > 0 )
      {
        forceX += ( separationX / sepMag ) * 0.15;
        forceY += ( separationY / sepMag ) * 0.15;
      }
    }

    if ( alignmentCount > 0 )
    {
      alignmentX /= alignmentCount;
      alignmentY /= alignmentCount;
      forceX += ( alignmentX - vx ) * 0.08;
      forceY += ( alignmentY - vy ) * 0.08;
    }

    if ( cohesionCount > 0 )
    {
      cohesionX /= cohesionCount;
      cohesionY /= cohesionCount;
      forceX += ( cohesionX - x ) * 0.01;
      forceY += ( cohesionY - y ) * 0.01;
    }

    // Limit force
    forceX = limit( forceX, MAX_FORCE );
    forceY = limit( forceY, MAX_FORCE );

    boidAccelerations[ i * 3 ] = forceX;
    boidAccelerations[ i * 3 + 1 ] = forceY;
  }

  // Integrate velocities and positions
  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    boidVelocities[ i * 3 ] += boidAccelerations[ i * 3 ];
    boidVelocities[ i * 3 + 1 ] += boidAccelerations[ i * 3 + 1 ];

    // Limit speed
    let speedSq = boidVelocities[ i * 3 ] * boidVelocities[ i * 3 ] + boidVelocities[ i * 3 + 1 ] * boidVelocities[ i * 3 + 1 ];
    if ( speedSq > MAX_SPEED * MAX_SPEED )
    {
      const speed = Math.sqrt( speedSq );
      boidVelocities[ i * 3 ] = ( boidVelocities[ i * 3 ] / speed ) * MAX_SPEED;
      boidVelocities[ i * 3 + 1 ] = ( boidVelocities[ i * 3 + 1 ] / speed ) * MAX_SPEED;
    }

    // Update position
    boidPositions[ i * 3 ] += boidVelocities[ i * 3 ];
    boidPositions[ i * 3 + 1 ] += boidVelocities[ i * 3 + 1 ];

    // Wrap around edges
    if ( boidPositions[ i * 3 ] < 0 ) boidPositions[ i * 3 ] += SIMULATION_SIZE.x;
    if ( boidPositions[ i * 3 ] > SIMULATION_SIZE.x ) boidPositions[ i * 3 ] -= SIMULATION_SIZE.x;
    if ( boidPositions[ i * 3 + 1 ] < 0 ) boidPositions[ i * 3 + 1 ] += SIMULATION_SIZE.y;
    if ( boidPositions[ i * 3 + 1 ] > SIMULATION_SIZE.y ) boidPositions[ i * 3 + 1 ] -= SIMULATION_SIZE.y;
  }
}

function initThree ()
{
  scene = new THREE.Scene();
  scene.background = new THREE.Color( 0x000005 );

  camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 1, 10000 );
  camera.position.set( 500, 300, 800 );

  renderer = new THREE.WebGLRenderer( { antialias: true } );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.setPixelRatio( window.devicePixelRatio );

  const container = document.getElementById( 'canvas-container' );
  container.appendChild( renderer.domElement );

  controls = new OrbitControls( camera, renderer.domElement );
  controls.target.set( 500, 300, 0 );
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Visual Bounds (2D plane)
  const boxGeom = new THREE.BoxGeometry( SIMULATION_SIZE.x, SIMULATION_SIZE.y, SIMULATION_SIZE.z );
  const edges = new THREE.EdgesGeometry( boxGeom );
  const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x444444 } ) );
  line.position.set( 500, 300, 0.5 );
  scene.add( line );

  // Grid on bottom for reference
  const gridHelper = new THREE.GridHelper( SIMULATION_SIZE.x, 20, 0x444444, 0x222222 );
  gridHelper.position.set( 500, 300, -0.5 );
  scene.add( gridHelper );

  // Boids - flat triangles
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

  // Add keyboard controls
  window.addEventListener( 'keydown', ( event ) =>
  {
    if ( event.key.toLowerCase() === 'v' )
    {
      verificationEnabled = !verificationEnabled;
      verificationErrors = 0; // Reset error count when toggling
      console.log( `Verification ${verificationEnabled ? 'enabled' : 'disabled'}` );
    }
  } );
}

function reinitializeBoids ( newCount )
{
  BOID_COUNT = newCount;

  // Reinitialize boid data
  boidPositions = new Float32Array( BOID_COUNT * 3 );
  boidVelocities = new Float32Array( BOID_COUNT * 3 );
  boidAccelerations = new Float32Array( BOID_COUNT * 3 );
  nearestNeighbors = new Int32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighborsDist = new Float32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighbors_verify = new Int32Array( BOID_COUNT * K_NEIGHBORS );
  nearestNeighborsDist_verify = new Float32Array( BOID_COUNT * K_NEIGHBORS );

  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    boidPositions[ i * 3 ] = Math.random() * SIMULATION_SIZE.x;
    boidPositions[ i * 3 + 1 ] = Math.random() * SIMULATION_SIZE.y;
    boidPositions[ i * 3 + 2 ] = 0.5;

    boidVelocities[ i * 3 ] = ( Math.random() - 0.5 ) * 2;
    boidVelocities[ i * 3 + 1 ] = ( Math.random() - 0.5 ) * 2;
    boidVelocities[ i * 3 + 2 ] = 0;

    boidAccelerations[ i * 3 ] = 0;
    boidAccelerations[ i * 3 + 1 ] = 0;
    boidAccelerations[ i * 3 + 2 ] = 0;

    for ( let j = 0; j < K_NEIGHBORS; j++ )
    {
      nearestNeighbors[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist[ i * K_NEIGHBORS + j ] = Infinity;
      nearestNeighbors_verify[ i * K_NEIGHBORS + j ] = -1;
      nearestNeighborsDist_verify[ i * K_NEIGHBORS + j ] = Infinity;
    }
  }

  // Remove old mesh
  scene.remove( boidInstancedMesh );

  // Create new mesh with updated count
  const geometry = new THREE.ConeGeometry( 2, 6, 5 ).rotateX( Math.PI / 2 );
  const material = new THREE.MeshPhongMaterial( { color: 0x00ff88 } );
  boidInstancedMesh = new THREE.InstancedMesh( geometry, material, BOID_COUNT );
  boidInstancedMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
  scene.add( boidInstancedMesh );

  console.log( `Reinitialized with ${BOID_COUNT} boids` );
}

function initControls ()
{
  const paramConfig = {
    SEPARATION_DISTANCE: {
      value: SEPARATION_DISTANCE,
      min: 10,
      max: 150,
      step: 5,
      decimals: 0,
      onChange: (val) => { SEPARATION_DISTANCE = val; }
    },
    ALIGNMENT_DISTANCE: {
      value: ALIGNMENT_DISTANCE,
      min: 10,
      max: 200,
      step: 5,
      decimals: 0,
      onChange: (val) => { ALIGNMENT_DISTANCE = val; }
    },
    COHESION_DISTANCE: {
      value: COHESION_DISTANCE,
      min: 10,
      max: 300,
      step: 10,
      decimals: 0,
      onChange: (val) => { COHESION_DISTANCE = val; }
    },
    MAX_SPEED: {
      value: MAX_SPEED,
      min: 0.5,
      max: 10,
      step: 0.1,
      decimals: 1,
      onChange: (val) => { MAX_SPEED = val; }
    },
    MAX_FORCE: {
      value: MAX_FORCE,
      min: 0.01,
      max: 0.5,
      step: 0.01,
      decimals: 2,
      onChange: (val) => { MAX_FORCE = val; }
    },
    K_NEIGHBORS: {
      value: K_NEIGHBORS,
      min: 5,
      max: 100,
      step: 1,
      decimals: 0,
      onChange: (val) => { K_NEIGHBORS = val; }
    },
    BOID_COUNT: {
      value: BOID_COUNT,
      min: 500,
      max: 5000,
      step: 100,
      decimals: 0,
      onChange: (val) => { reinitializeBoids( val ); }
    }
  };

  simControls = new SimulationControls(paramConfig);
}

let stepCount = 0;

function frame ()
{
  requestAnimationFrame( frame );

  if ( controls ) controls.update();

  // Update simulation
  updateBoids();
  stepCount++;

  // Update instance matrices
  for ( let i = 0; i < BOID_COUNT; i++ )
  {
    const stride = i * 3;
    _pos.set( boidPositions[ stride ], boidPositions[ stride + 1 ], boidPositions[ stride + 2 ] );
    _vel.set( boidVelocities[ stride ], boidVelocities[ stride + 1 ], 0 );

    // Orient towards velocity
    if ( _vel.lengthSq() > 0.01 )
    {
      _orient.setFromUnitVectors( _up, _vel.clone().normalize() );
    }

    _matrix.compose( _pos, _orient, { x: 1, y: 1, z: 1 } );
    boidInstancedMesh.setMatrixAt( i, _matrix );
  }

  boidInstancedMesh.instanceMatrix.needsUpdate = true;

  // Update UI
  const now = Date.now();
  frameCount++;
  const elapsed = now - lastFrameTime;
  if ( elapsed > 500 )
  {
    const fps = Math.round( ( frameCount * 1000 ) / elapsed );
    document.getElementById( 'info-fps' ).innerText = `FPS: ${fps}`;
    frameCount = 0;
    lastFrameTime = now;
  }

  document.getElementById( 'info-boids' ).innerText = `Boids: ${BOID_COUNT}`;
  document.getElementById( 'info-step' ).innerText = `Step: ${stepCount}`;
  
  let statusText = 'CPU Simulation';
  if ( verificationEnabled )
  {
    statusText += ` | Verify: ON (Errors: ${verificationErrors})`;
  }
  else
  {
    statusText += ' | Verify: OFF';
  }
  document.getElementById( 'info-app' ).innerText = statusText;
  document.getElementById( 'info-gpu' ).innerText = 'Keys: P=Params, V=Verify';

  renderer.render( scene, camera );
}

initBoids();
initThree();
initControls();
document.getElementById( 'info-app' ).innerText = 'Running...';
frame();

