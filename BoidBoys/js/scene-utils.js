/**
 * Scene utilities for separating ThreeJS scene management from boid simulation logic
 */

export class SceneManager {
  constructor(containerId, simulationSize = { x: 1000, y: 600, z: 600 }) {
    this.container = document.getElementById(containerId);
    this.simulationSize = simulationSize;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.boidInstancedMesh = null;
  }

  init() {
    this.initScene();
    this.initCamera();
    this.initRenderer();
    this.initControls();
    this.setupLights();
    this.addVisualizationBounds();
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000005);
  }

  initCamera() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(75, width / height, 1, 10000);
    this.camera.position.set(
      this.simulationSize.x * -0.5,
      this.simulationSize.y * 0.6,
      this.simulationSize.z || 1000
    );
  }

  initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.container.appendChild(this.renderer.domElement);

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  initControls() {
    const THREE = require('three'); // This should be imported at module level in actual use
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    const centerX = this.simulationSize.x / 2;
    const centerY = this.simulationSize.y / 2;
    const centerZ = (this.simulationSize.z || 1) / 2;
    this.controls.target.set(centerX, centerY, centerZ);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
  }

  setupLights() {
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    this.scene.add(directionalLight, ambientLight);
  }

  addVisualizationBounds() {
    const boxGeom = new THREE.BoxGeometry(
      this.simulationSize.x,
      this.simulationSize.y,
      this.simulationSize.z
    );
    const edges = new THREE.EdgesGeometry(boxGeom);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x444444 })
    );
    line.position.set(
      this.simulationSize.x / 2,
      this.simulationSize.y / 2,
      (this.simulationSize.z || 1) / 2
    );
    this.scene.add(line);

    // Add grid for 2D simulations
    if (this.simulationSize.z <= 1) {
      const gridHelper = new THREE.GridHelper(
        this.simulationSize.x,
        20,
        0x444444,
        0x222222
      );
      gridHelper.position.set(
        this.simulationSize.x / 2,
        this.simulationSize.y / 2,
        -0.5
      );
      this.scene.add(gridHelper);
    }
  }

  createBoidMesh(boidCount) {
    if (this.boidInstancedMesh) {
      this.scene.remove(this.boidInstancedMesh);
      this.boidInstancedMesh.geometry.dispose();
      this.boidInstancedMesh.material.dispose();
    }

    const geometry = new THREE.ConeGeometry(2, 6, 5).rotateX(Math.PI / 2);
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff88 });
    this.boidInstancedMesh = new THREE.InstancedMesh(geometry, material, boidCount);
    this.boidInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.boidInstancedMesh);

    return this.boidInstancedMesh;
  }

  updateBoidTransforms(positions, velocities, callback) {
    const matrix = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const orient = new THREE.Quaternion();
    const vel = new THREE.Vector3();
    const up = new THREE.Vector3(0, 0, 1);

    for (let i = 0; i < positions.length / 3; i++) {
      const stride = i * 3;
      pos.set(positions[stride], positions[stride + 1], positions[stride + 2]);
      vel.set(velocities[stride], velocities[stride + 1], 0);

      if (vel.lengthSq() > 0.01) {
        orient.setFromUnitVectors(up, vel.clone().normalize());
      }

      matrix.compose(pos, orient, { x: 1, y: 1, z: 1 });
      this.boidInstancedMesh.setMatrixAt(i, matrix);
    }

    this.boidInstancedMesh.instanceMatrix.needsUpdate = true;
    if (callback) callback();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  dispose() {
    this.renderer.dispose();
  }
}
