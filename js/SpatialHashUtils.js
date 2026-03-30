import { BASE_SIMULATION_SIZE, ParamsIndex } from './constants.js';

export function getSpawnBounds(worldSize)
{
  return {
    min: { x: 0, y: 0, z: 0 },
    max: { x: worldSize.x, y: worldSize.y, z: worldSize.z }
  };
}

export function calculateSimulationSize(count, density)
{
  const baseVolume = BASE_SIMULATION_SIZE.x * BASE_SIMULATION_SIZE.y * BASE_SIMULATION_SIZE.z;
  const requiredVolume = count / density;
  const scaleFactor = Math.cbrt(requiredVolume / baseVolume);
  return {
    x: BASE_SIMULATION_SIZE.x * scaleFactor,
    y: BASE_SIMULATION_SIZE.y * scaleFactor,
    z: BASE_SIMULATION_SIZE.z * scaleFactor
  };
}

export function calculateGridDimensions(paramsArray, simulationSize)
{
  const cellSize = Math.min(paramsArray[ParamsIndex.SEPARATION_DIST], paramsArray[ParamsIndex.ALIGN_DIST], paramsArray[ParamsIndex.COHESION_DIST], 50);
  const gridDim = {
    x: Math.max(1, Math.ceil(simulationSize.x / cellSize)),
    y: Math.max(1, Math.ceil(simulationSize.y / cellSize)),
    z: Math.max(1, Math.ceil(simulationSize.z / cellSize))
  };
  const numCells = gridDim.x * gridDim.y * gridDim.z;
  return { cellSize, gridDim, numCells };
}
