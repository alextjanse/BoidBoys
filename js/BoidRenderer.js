import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';

export class BoidRenderer
{
  constructor(containerId)
  {
    this.scene = new THREE.Scene();
    // this.scene.background = new THREE.Color(0x999999);
    const loader = new EXRLoader();
    loader.load('./resources/meadow_4k.exr', (texture) => {
    texture.mapping = THREE.EquirectangularReflectionMapping;
    this.scene.background = texture;
    })

    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 20000);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    document.getElementById(containerId).appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    this.scene.add(new THREE.DirectionalLight(0xffffff, 1), new THREE.AmbientLight(0xffffff, 0.3));

    this.boidInstancedMesh = null;
    this.boundsLine = null;

    window.addEventListener('resize', this.onWindowResize.bind(this));
  }

  onWindowResize()
  {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  updateVisualBounds(simulationSize)
  {
    if (this.boundsLine) {
      this.scene.remove(this.boundsLine);
    }

    const boxGeom = new THREE.BoxGeometry(simulationSize.x, simulationSize.y, simulationSize.z);
    const edges = new THREE.EdgesGeometry(boxGeom);
    this.boundsLine = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x444444 }));
    this.boundsLine.name = 'boid-bounds';
    this.boundsLine.position.set(simulationSize.x / 2, simulationSize.y / 2, simulationSize.z / 2);
    this.scene.add(this.boundsLine);

    if (this.controls) {
      this.controls.target.set(simulationSize.x / 2, simulationSize.y / 2, simulationSize.z / 2);

      const cx = simulationSize.x / 2;
      const cy = simulationSize.y / 2;
      const cz = simulationSize.z / 2;
      const d = simulationSize.x;
      this.camera.position.set(cx + d, cy + d, cz + d);
    }
  }

  createInstancedMesh(boidCount)
  {
    if (this.boidInstancedMesh) {
      this.scene.remove(this.boidInstancedMesh);
      this.boidInstancedMesh.geometry.dispose();
      this.boidInstancedMesh.material.dispose();
    }

    const body = new THREE.ConeGeometry(2, 6, 5).rotateX(Math.PI / 2);
    const wingLeft = new THREE.BoxGeometry(6, 0.1, 6).translate(3, 0, 0);
    const wingRight = new THREE.BoxGeometry(6, 0.1, 6).translate(-3, 0, 0);

    const bodyLen = body.attributes.position.count;
    const wingLen = wingLeft.attributes.position.count;

    body.setAttribute('isWing', new THREE.Float32BufferAttribute(new Float32Array(bodyLen).fill(0), 1));
    wingLeft.setAttribute('isWing', new THREE.Float32BufferAttribute(new Float32Array(wingLen).fill(1), 1));
    wingRight.setAttribute('isWing', new THREE.Float32BufferAttribute(new Float32Array(wingLen).fill(1), 1));

    const geometry = BufferGeometryUtils.mergeGeometries([body, wingLeft, wingRight], false);
    const material = new THREE.MeshPhongMaterial({ color: 0xFFFFFF });

    material.onBeforeCompile = (shader) =>
    {
      shader.uniforms.time = { value: 0 };

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <common>',
        `
          #include <common>
          varying vec3 vInstanceColor;
        `
      ).replace(
        '#include <color_fragment>',
        `
          #include <color_fragment>
          diffuseColor.rgb *= vInstanceColor;
        `
      );

      shader.vertexShader = `
      varying vec3 vInstanceColor;
      ${shader.vertexShader}
    `.replace(
        '#include <begin_vertex>',
        `
      #include <begin_vertex>
      vInstanceColor = instanceColor;
      `
      );

      shader.vertexShader = `
    attribute float isWing;
    uniform float time;

    // Helper to get a random float from an ID
    float hash(float n) {
        return fract(sin(n) * 43758.5453123);
    }

    ${shader.vertexShader}
  `.replace('#include <begin_vertex>',
        `
      #include <begin_vertex>
      
      if (isWing > 0.5) {
        // Create unique variations per instance
        float id = float(gl_InstanceID);
        float speedVariation = 0.5 + hash(id + 1.0) * 0.5; // Speed between 0.5x and 1.0x
        float phaseOffset = hash(id) * 6.28; // Phase offset between 0 and 2*PI
        
        float phase = (time * 8.0 * speedVariation) + phaseOffset;
        
        float distFromHinge = max(0.0, abs(position.x) - 1.0);
        float angle = sin(phase) * 0.5;
        
        transformed.y += distFromHinge * angle;
        transformed.z += distFromHinge * abs(angle) * 0.2;
      }
    `);

      material.userData.shaderUniforms = shader.uniforms;
    };

    this.boidInstancedMesh = new THREE.InstancedMesh(geometry, material, boidCount);

    const colorLeft = new THREE.Color(0x100904);
    const colorRight = new THREE.Color(0xAAAAAA);
    const color = new THREE.Color();
    for (let i = 0; i < boidCount; i++) {
      color.lerpColors(colorLeft, colorRight, Math.random());
      this.boidInstancedMesh.setColorAt(i, color);
    }

    this.boidInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.boidInstancedMesh);
  }

  updateInstances(matData)
  {
    if (this.boidInstancedMesh) {
      this.boidInstancedMesh.instanceMatrix.array.set(matData);
      this.boidInstancedMesh.instanceMatrix.needsUpdate = true;
    }
  }

  render(now)
  {
    if (this.controls) this.controls.update();

    if (this.boidInstancedMesh && this.boidInstancedMesh.material && this.boidInstancedMesh.material.userData && this.boidInstancedMesh.material.userData.shaderUniforms) {
      const su = this.boidInstancedMesh.material.userData.shaderUniforms;
      if (su.time) su.time.value = now * 0.001;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
