import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import * as tf from '@tensorflow/tfjs';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(3, 3, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
renderer.setAnimationLoop( animate );

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// const cube = new THREE.Mesh(
//   new THREE.BoxGeometry(),
//   new THREE.MeshNormalMaterial()
// );
// scene.add(cube);

const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshNormalMaterial();

let amount = 100;
let mesh = new THREE.InstancedMesh( geometry, material, amount**2 );
mesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage ); // will be updated every frame
scene.add( mesh );

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const dummy = new THREE.Object3D();

const offset = ( amount - 1 ) / 2;
function animate() {
  let i = 0;
  const epochSeconds = Math.floor(Date.now() / 10);

  //cube.rotation.x += 0.01;
  for ( let x = 0; x < amount; x ++ ) 
  {
    for (let z = 0; z < amount; z ++)
    {
        dummy.position.set( offset - x, 0, offset - z );
        dummy.scale.set( 1, 1 + Math.sin(3 * epochSeconds + x + 7 * z), 1 );
        dummy.updateMatrix();
        mesh.setMatrixAt( i, dummy.matrix );
        i ++;
    }
  }
  mesh.instanceMatrix.needsUpdate = true;

  controls.update();
  renderer.render(scene, camera);

//   // Define two 512x512 matrices with random values
//     const a = tf.randomUniform([128, 128]);
//     const b = tf.randomUniform([128, 128]);

//     // Matrix multiplication
//     const c = tf.matMul(a, b);

//     // Optionally print the result (just first few values)
//     c.slice([4, 4], [1, 2]).print();
}

//animate();
