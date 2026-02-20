// Minimal ThreeJS integration stub. Replace TODOs with real ThreeJS setup.
// Exports: init(containerId, params, dotNetRef), reset(params), setRunning(running), dispose()

let container;
let canvas;
let animationHandle = null;
let lastTime = 0;
let fps = 0;
let frameCount = 0;
let fpsLastTime = 0;
let step = 0;
let paramsGlobal = {};
let running = true;
let dotNetRefGlobal = null;

export async function init(containerId, params, dotNetRef) {
    container = document.getElementById(containerId);
    if (!container) {
        console.warn('Three container not found:', containerId);
        return;
    }

    paramsGlobal = params || {};
    dotNetRefGlobal = dotNetRef;
    running = paramsGlobal.running ?? true;

    // create a canvas that ThreeJS can use (or let ThreeJS create its own renderer)
    canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    container.innerHTML = ''; // clear
    container.appendChild(canvas);

    // TODO: initialize ThreeJS renderer, scene, camera, lights, boid meshes/points
    // Example (pseudocode):
    // import * as THREE from 'three';
    // renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    // camera = new THREE.PerspectiveCamera(...);
    // scene = new THREE.Scene();
    // create boid geometry/instances based on params.boidCount
    // store whatever state you need in window.boidScene or closure variables.

    lastTime = performance.now();
    fpsLastTime = lastTime;
    frameCount = 0;
    step = 0;

    startLoop();
}

function startLoop() {
    if (animationHandle !== null) {
        cancelAnimationFrame(animationHandle);
    }
    animationHandle = requestAnimationFrame(loop);
}

function loop(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05); // seconds, clamp
    lastTime = now;

    if (running) {
        // TODO: advance boid simulation and update ThreeJS objects
        // For now step just increments
        step++;
    }

    // TODO: render scene with ThreeJS renderer
    // Example: renderer.render(scene, camera);

    // fps calculation
    frameCount++;
    if (now - fpsLastTime >= 500) {
        fps = (frameCount * 1000) / (now - fpsLastTime);
        fpsLastTime = now;
        frameCount = 0;

        // send stats back to .NET (safe-guard dotNetRef)
        if (dotNetRefGlobal) {
            // Use the instance reference's method name: ReceiveStats
            dotNetRefGlobal.invokeMethodAsync('ReceiveStats', fps, paramsGlobal.boidCount ?? 0, step)
                .catch(e => console.debug('dotnet callback failed', e));
        }
    }

    animationHandle = requestAnimationFrame(loop);
}

export function reset(params) {
    // update stored params and reinitialize boids/scene as necessary
    paramsGlobal = params || paramsGlobal;
    running = paramsGlobal.running ?? running;

    // TODO: rebuild boid buffers / instance geometry based on paramsGlobal.boidCount and other parameters
    // Example: dispose old geometry/meshes, create new instances.

    // reset step counter so UI shows new run
    step = 0;
}

export function setRunning(r) {
    running = !!r;
}

export function dispose() {
    if (animationHandle !== null) {
        cancelAnimationFrame(animationHandle);
        animationHandle = null;
    }

    // TODO: dispose ThreeJS renderer, geometries, materials, textures, etc.
    if (container && canvas) {
        container.removeChild(canvas);
    }
    canvas = null;
    container = null;
    dotNetRefGlobal = null;
}