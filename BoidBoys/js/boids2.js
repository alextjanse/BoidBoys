const N_BOIDS = 1000;
const WIDTH = 1200;
const HEIGHT = 800;

const positionsX = Float32Array(N_BOIDS)
const positionsY = Float32Array(N_BOIDS)
const velX = Float32Array(N_BOIDS)
const velY = Float32Array(N_BOIDS)
const nextPositionsX = Float32Array(N_BOIDS)
const nextPositionsY = Float32Array(N_BOIDS)
const nextVelX = Float32Array(N_BOIDS)
const nextVelY = Float32Array(N_BOIDS)
const indices = Int32Array(N_BOIDS)

function compareBoids(i, j) {
    const xi = positionsX[i];
    const yi = positionsY[i];
    const xj = positionsX[j];
    const yj = positionsY[j];

    if (xi < xj) return -1;
    if (xi > xj) return 1;
    if (yi < yj) return -1;
    if (yi > yj) return 1;
    return 0;
}

function randomPos() {
    const x = Math.random() * WIDTH;
    const y = Math.random() * HEIGHT;
    return { x, y };
}

function randomVel() {
    const dx = 2 * Math.random() - 1;
    const dy = 2 * Math.random() - 1;
    const length = dx + dy;
    dx /= length;
    dy /= length;
    return { dx, dy }
}

// Init
for (let index = 0; index < N_BOIDS; index++) {
    const { x, y } = randomPos()
    positionsX[index] = x;
    positionsY[index] = y;

    const { dx, dy } = randomVel();
    velX[index] = dx;
    velY[index] = dy;
}

// Sort
indices.sort(compareBoids);

function bubbleUp() {
    let idx = N_BOIDS - 1;
    const elem = indices[idx];
    while (idx > 0) {
        let 
    }
}

function sweep() {
    const state = Map<
}

requestAnimationFrame(() => {

})

