import { WORKGROUP_SIZE, ParamsIndex } from './constants.js';
import { getSpawnBounds, calculateGridDimensions, calculateSimulationSize } from './SpatialHashUtils.js';

export class BoidEngine
{
  constructor()
  {
    this.gpuDevice = null;
    this.paramsArray = new Float32Array(28);
    this.boidCount = null;
    this.simulationSize = null;
    this.cellSize = 50;
    this.gridDim = { x: 1, y: 1, z: 1 };
    this.numCells = 1;

    this.boidBuffer = null;
    this.cellHeadBuffer = null;
    this.boidNextBuffer = null;
    this.matrixBuffer = null;
    this.matrixStagingBuffer = null;
    this.uniformBuffer = null;

    this.clearCellsPipeline = null;
    this.hashInsertPipeline = null;
    this.updateBoidsPipeline = null;
    this.computeMatricesPipeline = null;

    this.bindGroupLayout = null;
    this.bindGroup = null;
    this.isMapping = false;

  }

  async init(boidCount, boidDensity)
  {
    this.boidCount = boidCount;
    this.boidDensity = boidDensity;
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      console.error("WebGPU not supported");
      return false;
    }

    this.gpuDevice = await adapter.requestDevice();

    const shaderCode = await fetch('compute-shader.wgsl').then(r => r.text());
    const shaderModule = this.gpuDevice.createShaderModule({ code: shaderCode });

    this.resetParamsToDefaults(boidCount, boidDensity);

    this.initBoidBuffers();
    this.initSpatialHashBuffers();
    this.initMatrixBuffers();

    // Uniform buffer
    this.uniformBuffer = this.gpuDevice.createBuffer({
      size: this.paramsArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.syncParams();

    // Bind group layout
    this.bindGroupLayout = this.gpuDevice.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });

    const pipelineLayout = this.gpuDevice.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    const makePipeline = (entryPoint) => this.gpuDevice.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint }
    });

    this.clearCellsPipeline = makePipeline('clear_cells');
    this.hashInsertPipeline = makePipeline('hash_insert');
    this.updateBoidsPipeline = makePipeline('update_boids');
    this.computeMatricesPipeline = makePipeline('compute_matrices');

    this.createBindGroups();

    return true;
  }

  resetParamsToDefaults(boidCount, boidDensity)
  {
    this.boidCount = boidCount;
    this.simulationSize = calculateSimulationSize(boidCount, boidDensity);

    this.paramsArray[ParamsIndex.SEPARATION_DIST] = 25.0;  // separation_dist
    this.paramsArray[ParamsIndex.ALIGN_DIST] = 50.0;  // align_dist
    this.paramsArray[ParamsIndex.COHESION_DIST] = 50.0;  // cohesion_dist

    this.updateGrid();

    this.paramsArray[ParamsIndex.MAX_SPEED] = 5.0;   // max_speed
    this.paramsArray[ParamsIndex.MAX_FORCE] = 0.1;   // max_force
    this.paramsArray[ParamsIndex.SEPARATION_WEIGHT] = 1.5;   // separation_weight
    this.paramsArray[ParamsIndex.ALIGNMENT_WEIGHT] = 1.0;   // alignment_weight
    this.paramsArray[ParamsIndex.COHESION_WEIGHT] = 0.5;   // cohesion_weight
    this.paramsArray[ParamsIndex.MARGIN] = 100.0; // margin
    this.paramsArray[ParamsIndex.TURN_FACTOR] = 0.2;   // turn_factor

    this.updateParamsArrayFromGrid();
  }

  updateGrid()
  {
    const gridData = calculateGridDimensions(this.paramsArray, this.simulationSize);
    this.cellSize = gridData.cellSize;
    this.gridDim = gridData.gridDim;
    this.numCells = gridData.numCells;
  }

  updateParamsArrayFromGrid()
  {
    this.paramsArray[ParamsIndex.CELL_SIZE] = this.cellSize;
    this.paramsArray[ParamsIndex.PADDING] = 0.0;

    this.paramsArray[ParamsIndex.WORLD_MAX] = this.simulationSize.x;
    this.paramsArray[ParamsIndex.WORLD_MAX + 1] = this.simulationSize.y;
    this.paramsArray[ParamsIndex.WORLD_MAX + 2] = this.simulationSize.z;
    this.paramsArray[ParamsIndex.WORLD_MAX + 3] = 0.0;

    this.paramsArray[ParamsIndex.GRID_DIM] = this.gridDim.x;
    this.paramsArray[ParamsIndex.GRID_DIM + 1] = this.gridDim.y;
    this.paramsArray[ParamsIndex.GRID_DIM + 2] = this.gridDim.z;
    this.paramsArray[ParamsIndex.GRID_DIM + 3] = this.numCells;
  }

  syncParams()
  {
    if (!this.gpuDevice || !this.uniformBuffer) return;
    this.updateGrid();
    this.updateParamsArrayFromGrid();

    this.gpuDevice.queue.writeBuffer(this.uniformBuffer, 0, this.paramsArray.buffer, this.paramsArray.byteOffset, this.paramsArray.byteLength);
  }

  initBoidBuffers()
  {
    if (this.boidBuffer) {
      this.boidBuffer.destroy();
    }
    const boidData = new Float32Array(this.boidCount * 8);
    const spawnBounds = getSpawnBounds(this.simulationSize);
    for (let i = 0; i < this.boidCount; i++) {
      boidData[i * 8] = spawnBounds.min.x + Math.random() * (spawnBounds.max.x - spawnBounds.min.x);
      boidData[i * 8 + 1] = spawnBounds.min.y + Math.random() * (spawnBounds.max.y - spawnBounds.min.y);
      boidData[i * 8 + 2] = spawnBounds.min.z + Math.random() * (spawnBounds.max.z - spawnBounds.min.z);
      boidData[i * 8 + 3] = 1.0;
      boidData[i * 8 + 4] = (Math.random() - 0.5) * 4;
      boidData[i * 8 + 5] = (Math.random() - 0.5) * 4;
      boidData[i * 8 + 6] = (Math.random() - 0.5) * 4;
      boidData[i * 8 + 7] = 0.0;
    }

    this.boidBuffer = this.gpuDevice.createBuffer({
      size: boidData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.boidBuffer.getMappedRange()).set(boidData);
    this.boidBuffer.unmap();
  }

  initSpatialHashBuffers()
  {
    if (this.cellHeadBuffer) {
      this.cellHeadBuffer.destroy();
    }
    if (this.boidNextBuffer) {
      this.boidNextBuffer.destroy();
    }

    this.updateGrid();

    this.cellHeadBuffer = this.gpuDevice.createBuffer({
      size: Math.max(4, this.numCells * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.boidNextBuffer = this.gpuDevice.createBuffer({
      size: Math.max(4, this.boidCount * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  initMatrixBuffers()
  {
    if (this.matrixBuffer) {
      this.matrixBuffer.destroy();
    }
    if (this.matrixStagingBuffer) {
      this.matrixStagingBuffer.destroy();
    }
    const matSize = this.boidCount * 16 * 4; // 16 floats per mat4, 4 bytes per float

    this.matrixBuffer = this.gpuDevice.createBuffer({
      size: Math.max(4, matSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.matrixStagingBuffer = this.gpuDevice.createBuffer({
      size: Math.max(4, matSize),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  createBindGroups()
  {
    this.bindGroup = this.gpuDevice.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.boidBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
        { binding: 2, resource: { buffer: this.cellHeadBuffer } },
        { binding: 3, resource: { buffer: this.boidNextBuffer } },
        { binding: 4, resource: { buffer: this.matrixBuffer } },
      ]
    });
  }

  recreateBoids(newCount, newDensity)
  {
    this.boidCount = newCount;
    this.simulationSize = calculateSimulationSize(newCount, newDensity);
    this.resetParamsToDefaults(newCount, newDensity);

    this.initBoidBuffers();
    this.initSpatialHashBuffers();
    this.initMatrixBuffers();
    this.syncParams();

    this.createBindGroups();
  }

  async step(sceneContext)
  {
    if (!this.gpuDevice || this.isMapping) return null;

    const simStart = performance.now();
    this.syncParams();

    const encoder = this.gpuDevice.createCommandEncoder();
    const wgBoids = Math.ceil(this.boidCount / WORKGROUP_SIZE);
    const wgCells = Math.ceil(this.numCells / WORKGROUP_SIZE);

    const p1 = encoder.beginComputePass();
    p1.setPipeline(this.clearCellsPipeline);
    p1.setBindGroup(0, this.bindGroup);
    p1.dispatchWorkgroups(wgCells);
    p1.end();

    const p2 = encoder.beginComputePass();
    p2.setPipeline(this.hashInsertPipeline);
    p2.setBindGroup(0, this.bindGroup);
    p2.dispatchWorkgroups(wgBoids);
    p2.end();

    const p3 = encoder.beginComputePass();
    p3.setPipeline(this.updateBoidsPipeline);
    p3.setBindGroup(0, this.bindGroup);
    p3.dispatchWorkgroups(wgBoids);
    p3.end();

    const p4 = encoder.beginComputePass();
    p4.setPipeline(this.computeMatricesPipeline);
    p4.setBindGroup(0, this.bindGroup);
    p4.dispatchWorkgroups(wgBoids);
    p4.end();

    encoder.copyBufferToBuffer(this.matrixBuffer, 0, this.matrixStagingBuffer, 0, this.matrixBuffer.size);
    this.gpuDevice.queue.submit([encoder.finish()]);

    const simDelta = performance.now() - simStart;

    this.isMapping = true;
    try {
      await this.matrixStagingBuffer.mapAsync(GPUMapMode.READ);
      const renderStart = performance.now();
      const matData = new Float32Array(this.matrixStagingBuffer.getMappedRange());

      sceneContext.updateInstances(matData);

      this.matrixStagingBuffer.unmap();
      this.isMapping = false;

      const renderDelta = performance.now() - renderStart;
      return { simDelta, renderDelta };

    } catch (e) {
      this.isMapping = false;
      return null;
    }
  }
}
