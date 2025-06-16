import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import * as tf from '@tensorflow/tfjs';
import { applyPatchTo2DTensor } from './helper.js';


const res = await fetch('./data/channel_17.txt');
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

let M = xmax - xmin + 1;
let N = ymax - ymin + 1;

function makeArray(dx, dy)
{
    const arr = new Array(dx);
    for (let i = 0; i < dx; ++i)
    {
        arr[i] = new Array(dy).fill(0);
    }
    return arr;
}

const terrainData = makeArray(M, N);
for (const parsed of parsed_lines)
{
    const [x, y, z, color] = parsed;
    if (color == '8f563b')
    {
        terrainData[x - xmin][y - ymin] = Math.max(terrainData[x - xmin][y - ymin], z - zmax);
    }
}

const waterData = makeArray(M, N);
const sourceData = makeArray(M, N);
for (const parsed of parsed_lines)
{
    const [x, y, z, color] = parsed;
    if (color == '639bff' || color == 'fbf236')
    {
        waterData[x - xmin][y - ymin] = Math.max(waterData[x - xmin][y - ymin], z - terrainData[x - xmin][y - ymin]);
    }
    if (color == 'fbf236')
    {
        sourceData[x - xmin][y - ymin] += 1
    }
}

const terrain = tf.tensor(terrainData);
let water = tf.tensor(waterData);
const source = tf.tensor(sourceData);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(10, 15, 10);
const light = new THREE.HemisphereLight(0xB1E1FF, 0xB97A20, 5);
scene.add(light);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
renderer.setAnimationLoop( animate );

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const geometry = new THREE.BoxGeometry();
const terrainMaterial = new THREE.MeshPhongMaterial({
    color: 0x8f563b
});

let terrainMesh = new THREE.InstancedMesh( geometry, terrainMaterial, M * N );
terrainMesh.instanceMatrix.setUsage( THREE.StaticDrawUsage ); 
scene.add( terrainMesh );

const waterMaterial = new THREE.MeshPhongMaterial({
    color: 0x1e90ff
});
let waterMesh = new THREE.InstancedMesh(geometry, waterMaterial, M * N);
waterMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage ); // will be updated every frame
scene.add( waterMesh );

function drawTerrain()
{
    const dummy = new THREE.Object3D();
    let i = 0;
    for ( let x = 0; x < M; x ++ ) 
    {
        for (let z = 0; z < N; z ++)
        {
            let h = terrainData[x][z];
            dummy.position.set( x, h / 2.0, z );
            dummy.scale.set( 1, h, 1 );
            dummy.updateMatrix();
            terrainMesh.setMatrixAt( i, dummy.matrix );
            i ++;
        }
    }
    terrainMesh.instanceMatrix.needsUpdate = true;
}
drawTerrain();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const epsilon = 0.001;
let step = 0;
let flowX = tf.tensor(makeArray(M+1, N));
let flowY = tf.tensor(makeArray(M, N+1));
const dx = tf.scalar(1);
const dt = tf.scalar(0.1);
const sourceRate = tf.scalar(0.2);
const g = tf.scalar(0.1);

const dx2Bydt = dx.mul(dx).div(dt);

function simulate()
{
    // Source injection
    water = tf.add(water, tf.mul(dt, tf.mul(source, sourceRate)));

    // Flow change due to gravity
    const surface = tf.add(water, terrain);
    let flowIncX = surface.slice([0,0], [M-1,N]).sub(surface.slice([1,0], [M-1,N]));

    flowIncX = flowIncX.mul(g).mul(dx);
    flowX = applyPatchTo2DTensor(flowX, tf.add(flowX.slice([1,0], [M-1,N]), flowIncX), 1, 0);

    let flowIncY = surface.slice([0,0], [M, N-1]).sub(surface.slice([0,1], [M,N-1]));
    flowIncY = flowIncY.mul(g).mul(dx);
    flowY = applyPatchTo2DTensor(flowY, tf.add(flowY.slice([0,1], [M,N-1]), flowIncY), 0, 1);
    // Overdraft mitigation
    const totalOutflow = tf.add(tf.add(tf.maximum(0, tf.neg(flowX.slice([0,0], [M,N]))),
                       tf.maximum(0, flowX.slice([1,0], [M,N]))),
                       tf.add(tf.maximum(0, tf.neg(flowY.slice([0,0], [M,N]))),
                       tf.maximum(0, flowY.slice([0,1], [M,N]))));

    const noOutFlow = tf.equal(totalOutflow, 0).cast('float32');
    const hasOutFlow = tf.greater(totalOutflow, 0).cast('float32');
    //console.log(totalOutflow.arraySync(), noOutFlow.arraySync());
    let scale = noOutFlow.add(hasOutFlow.mul(water).mul(dx2Bydt).div(tf.maximum(totalOutflow, 0.001)));
    scale = tf.minimum(scale, 1);

    const scaledByRight = tf.less(flowX.slice([0,0], [M,N]), 0).cast('float32');
    const scaledByLeft = tf.greater(flowX.slice([1,0], [M,N]), 0).cast('float32');
    const scaledXL = flowX.slice([0,0], [M,N]).mul(scaledByRight).mul(scale);
    const scaledXR = flowX.slice([1,0], [M,N]).mul(scaledByLeft).mul(scale);
    flowX = applyPatchTo2DTensor(flowX, scaledXL, 0, 0);
    //console.log(scale.arraySync());
    flowX = applyPatchTo2DTensor(flowX, tf.zeros([1,N]), M, 0);
    flowX = applyPatchTo2DTensor(flowX, flowX.slice([1,0], [M,N]).add(scaledXR), 1, 0);

    const scaledByBottom = tf.less(flowY.slice([0,0], [M,N]), 0).cast('float32');
    const scaledByTop = tf.greater(flowY.slice([0,1], [M,N]), 0).cast('float32');
    const scaledYB = flowY.slice([0,0],[M,N]).mul(scaledByBottom).mul(scale);
    const scaledYT = flowY.slice([0,1],[M,N]).mul(scaledByTop).mul(scale);
    flowY = applyPatchTo2DTensor(flowY, scaledYB, 0, 0);
    flowY = applyPatchTo2DTensor(flowY, tf.zeros([M,1]), 0, N);
    flowY = applyPatchTo2DTensor(flowY, flowY.slice([0,1], [M,N]).add(scaledYT), 0, 1);

    let waterDelta = flowX.slice([0,0],[M,N]).add(flowY.slice([0,0],[M,M]))
                     .sub(flowX.slice([1,0],[M,N])).sub(flowY.slice([0,1],[M,N]));
    waterDelta = waterDelta.div(dx2Bydt);
    water = water.add(waterDelta);

    applyPatchTo2DTensor(water, tf.zeros([1,N]), 0, 0);
    applyPatchTo2DTensor(water, tf.zeros([1,N]), M-1, 0);
    applyPatchTo2DTensor(water, tf.zeros([M,1]), 0, 0);
    applyPatchTo2DTensor(water, tf.zeros([M,1]), 0, N-1);

    step++;
}

function drawWater()
{
    let depthArray = water.arraySync();

    const dummy = new THREE.Object3D();
    let i = 0;
    for ( let x = 0; x < M; x ++ ) 
    {
        for (let z = 0; z < N; z ++)
        {
            let h = depthArray[x][z];
            if (h < epsilon)
            {
                dummy.scale.set( 0, 0, 0 );
            }
            else
            {
                let _base = terrainData[x][z];
                dummy.position.set( x, _base + h / 2.0, z );
                dummy.scale.set( 1, h, 1 );
            }
            dummy.updateMatrix();
            waterMesh.setMatrixAt( i, dummy.matrix );

            i ++;
        }
    }
    waterMesh.instanceMatrix.needsUpdate = true;
}

function animate() {
  simulate();
  drawWater();

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
