import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import * as tf from '@tensorflow/tfjs';

const res = await fetch('./data/sink.txt');
const text = await res.text();

const lines = text.trim().split('\n');

let xmin = 9999, ymin = 9999, zmin = 9999;
let xmax = -9999, ymax = -9999, zmax = -9999;

let parsed_lines = [];

for (const line of lines)
{
    if (line.startsWith('#'))
    {
        continue;
    }
    const [xStr, yStr, zStr, color] = line.trim().split(/\s+/);
    const x = parseInt(xStr, 10);
    const y = parseInt(yStr, 10);
    const z = parseInt(zStr, 10);

    parsed_lines.push([x,y,z,color]);
    
    if (color === '000000'){
        // 000000 determines base size
        xmin = Math.min(xmin, x)
        ymin = Math.min(ymin, y)
        zmin = Math.min(zmin, z)
        xmax = Math.max(xmax, x)
        ymax = Math.max(ymax, y)
        zmax = Math.max(zmax, z)
    }
}

let amountX = xmax - xmin + 1;
let amountY = ymax - ymin + 1;
const terrainData = new Array(amountX);
for (let i = 0; i < amountX; i++)
{
    terrainData[i] = new Array(amountY).fill(0);
}
for (const parsed of parsed_lines)
{
    const [x, y, z, color] = parsed;
    if (color == '8f563b')
    {
        terrainData[x - xmin][y - ymin] = Math.max(terrainData[x - xmin][y - ymin], z - zmin);
    }
}

const terrain = tf.tensor(terrainData);
terrain.print();

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

let terrainArray = terrain.arraySync();

let mesh = new THREE.InstancedMesh( geometry, material, amountX * amountY );
mesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage ); // will be updated every frame
scene.add( mesh );

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const dummy = new THREE.Object3D();

function animate() {
    //console.log(terrain);
  let i = 0;
  const epochSeconds = Math.floor(Date.now() / 10);

  //cube.rotation.x += 0.01;
  for ( let x = 0; x < amountX; x ++ ) 
  {
    for (let z = 0; z < amountY; z ++)
    {
        let h = terrainArray[x][z];
        dummy.position.set( x, h / 2, z );
        dummy.scale.set( 1, h / 2, 1 );
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
