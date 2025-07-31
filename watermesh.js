import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

const scene = new THREE.Scene();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
renderer.setAnimationLoop( animate );

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xB1E1FF, 5);
scene.add(light);

scene.background = new THREE.Color(0x8B8000); // Sky blue


const resolution = 128;
const planeSize = 100;

const geometry = new THREE.PlaneGeometry(planeSize, planeSize, resolution - 1, resolution - 1);
geometry.rotateX(-Math.PI / 2); // make it horizontal (XZ plane)

const size = resolution * resolution;

const tensor = tf.randomUniform([resolution, resolution]); // 2D tensor

const data = tensor.dataSync(); // flatten (row-major)

//const data = new Float32Array(size);

// Fill with some data
for (let i = 0; i < size; i++) {
    data[i] = Math.random(); // or 0 to start flat
}

const heightTexture = new THREE.DataTexture(
    data,
    resolution,
    resolution,
    THREE.RedFormat,
    THREE.FloatType
);
heightTexture.needsUpdate = true;


const material = new THREE.ShaderMaterial({
    vertexShader: `
        uniform sampler2D heightMap;
        uniform float heightScale;
        varying vec2 vUv;

        void main() {
            vUv = uv;

            // Sample height from texture
            float height = texture2D(heightMap, uv).r;

            // Displace vertex in Y
            vec3 displaced = position + normal * height * heightScale;

            gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
        }
    `,
    fragmentShader: `
        void main() {
            gl_FragColor = vec4(0.61, 0.82, 0.93, 1.0);
        }
    `,
    uniforms: {
        heightMap: { value: heightTexture },
        heightScale: { value: 1.0 }
    },
    //wireframe: true // for debug
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);


function animate() {
    controls.update();
    renderer.render(scene, camera);
  }