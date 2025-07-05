import * as THREE from 'three';
import ThreeMeshUI from 'three-mesh-ui';
import * as tf from '@tensorflow/tfjs';

import FontJSON from './assets/Roboto-msdf.json';
import FontImage from './assets/Roboto-msdf.png';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { applyPatchTo2DTensor, makeArray, loadTerrain } from './helper.js';

let terrain = tf.tensor([]);
let water = tf.tensor([]);
let source = tf.tensor([]);
let flowX = tf.tensor([]);
let flowY = tf.tensor([]); 
let step = 0;
let M = 0;
let N = 0;
let terrainData = [];

const terrainList = ["channel_17", "channel_32", "sink"];
let terrainIndex = 0;

let loadTerrainResult = await loadTerrain(terrainList[terrainIndex]);
initTensors(loadTerrainResult);

const scene = new THREE.Scene();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
renderer.setAnimationLoop( animate );

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.HemisphereLight(0xB1E1FF, 0xB97A20, 5);
scene.add(light);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

function initTensors(loadTerrainResult)
{
    terrain.dispose();
    water.dispose();
    source.dispose();
    flowX.dispose();
    flowY.dispose();
    step = 0;

    let [_terrainData, waterData, sourceData] = loadTerrainResult;
    terrainData = _terrainData;
    M = terrainData.length;
    N = terrainData[0].length;

    terrain = tf.tensor(terrainData);
    water = tf.tensor(waterData);
    source = tf.tensor(sourceData);

    flowX = tf.tensor(makeArray(M+1, N));
    flowY = tf.tensor(makeArray(M, N+1));
}

const mapObjects = []

function setupScene()
{
    mapObjects.forEach(( obj ) => { scene.remove(obj);});
    while(mapObjects.length > 0) {
        mapObjects.pop();
    }

    const geometry = new THREE.BoxGeometry();
    const terrainMaterial = new THREE.MeshPhongMaterial({
        color: 0x8f563b
    });

    let terrainMesh = new THREE.InstancedMesh( geometry, terrainMaterial, M * N );
    terrainMesh.instanceMatrix.setUsage( THREE.StaticDrawUsage ); 
    scene.add( terrainMesh );

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

    const waterMaterial = new THREE.MeshPhongMaterial({
        color: 0x1e90ff
    });
    let waterMesh = new THREE.InstancedMesh(geometry, waterMaterial, M * N);
    waterMesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage ); // will be updated every frame
    scene.add( waterMesh );

    mapObjects.push(terrainMesh, waterMesh);
    return waterMesh;
}

let waterMesh = setupScene();

const epsilon = 0.001;
const dx = tf.scalar(1);
const dt_base = tf.scalar(0.1);
const dt_mult = 5;
const sourceRate = tf.scalar(0.2);
const g = tf.scalar(0.1);
const friction = 0.1;

const dt = tf.mul(dt_base, dt_mult);
const dx2Bydt = dx.mul(dx).div(dt);
const coeff_const = (1 - (1 - friction) ** dt_mult) / friction;

function simulate()
{
    tf.tidy(() => {
        // --- Source injection ---
        const injected = tf.mul(dt, tf.mul(source, sourceRate));
        const newWater1 = tf.add(water, injected);
        water.dispose();
        water = tf.keep(newWater1);

        // --- Flow change due to gravity (X) ---
        const surface = tf.add(water, terrain);

        const boost_x1 = tf.mul(tf.maximum(0, tf.neg(tf.sign(flowX.slice([1,0], [M-1,N])))),
            tf.div(tf.neg(flowX.slice([2,0], [M-1,N])), tf.maximum(water.slice([1,0], [M-1,N]), epsilon)));
        const boost_x2 = tf.mul(tf.maximum(0, tf.sign(flowX.slice([1,0], [M-1,N]))), 
            tf.div(flowX.slice([0,0], [M-1,N]), tf.maximum(water.slice([0,0], [M-1,N]), epsilon)));
        const boost_x = tf.exp(tf.minimum(1, tf.div(tf.maximum(boost_x1, boost_x2), dx)));

        let flowIncX = surface.slice([0, 0], [M - 1, N])
            .sub(surface.slice([1, 0], [M - 1, N]))
            .mul(g).mul(dx);
        const newFlowXCenter = tf.add(
            tf.mul(flowX.slice([1, 0], [M - 1, N]), (1 - friction) ** dt_mult),
            tf.mul(tf.mul(boost_x, coeff_const), flowIncX));
        const newFlowX1 = applyPatchTo2DTensor(flowX, newFlowXCenter, 1, 0);
        flowX.dispose();
        flowX = tf.keep(newFlowX1);

        // --- Flow change due to gravity (Y) ---
        const boost_y1 = tf.mul(tf.maximum(0, tf.neg(tf.sign(flowY.slice([0,1], [M,N-1])))),
            tf.div(tf.neg(flowY.slice([0,2], [M,N-1])), tf.maximum(water.slice([0,1], [M,N-1]), epsilon)));
        const boost_y2 = tf.mul(tf.maximum(0, tf.sign(flowY.slice([0,1], [M,N-1]))),
            tf.div(flowY.slice([0,0], [M,N-1]), tf.maximum(water.slice([0,0], [M,N-1]), epsilon)));
        const boost_y = tf.exp(tf.minimum(1, tf.div(tf.maximum(boost_y1, boost_y2), dx)));

        let flowIncY = surface.slice([0, 0], [M, N - 1])
            .sub(surface.slice([0, 1], [M, N - 1]))
            .mul(g).mul(dx);
        const newFlowYCenter = tf.add(
            tf.mul(flowY.slice([0, 1], [M, N - 1]), (1 - friction) ** dt_mult),
            tf.mul(tf.mul(boost_y, coeff_const), flowIncY));

        const newFlowY1 = applyPatchTo2DTensor(flowY, newFlowYCenter, 0, 1);
        flowY.dispose();
        flowY = tf.keep(newFlowY1);

        // --- Overdraft mitigation ---
        const totalOutflow = tf.add(
        tf.add(tf.maximum(0, tf.neg(flowX.slice([0, 0], [M, N]))),
                tf.maximum(0, flowX.slice([1, 0], [M, N]))),
        tf.add(tf.maximum(0, tf.neg(flowY.slice([0, 0], [M, N]))),
                tf.maximum(0, flowY.slice([0, 1], [M, N])))
        );

        const noOutFlow = tf.equal(totalOutflow, 0).cast('float32');
        const hasOutFlow = tf.greater(totalOutflow, 0).cast('float32');
        let scale = noOutFlow.add(hasOutFlow.mul(water).mul(dx2Bydt).div(tf.maximum(totalOutflow, 0.001)));
        scale = tf.minimum(scale, 1);

        // --- Scale flows ---
        const scaledByRight = tf.less(flowX.slice([0, 0], [M, N]), 0).cast('float32');
        const scaledByLeft  = tf.greater(flowX.slice([1, 0], [M, N]), 0).cast('float32');
        const scaledXL = flowX.slice([0, 0], [M, N]).mul(scaledByRight).mul(scale);
        const scaledXR = flowX.slice([1, 0], [M, N]).mul(scaledByLeft).mul(scale);

        let tempFlowX = applyPatchTo2DTensor(flowX, scaledXL, 0, 0);
        flowX.dispose();
        flowX = tf.keep(tempFlowX);

        tempFlowX = applyPatchTo2DTensor(flowX, tf.zeros([1, N]), M, 0);
        flowX.dispose();
        flowX = tf.keep(tempFlowX);

        tempFlowX = applyPatchTo2DTensor(flowX, flowX.slice([1, 0], [M, N]).add(scaledXR), 1, 0);
        flowX.dispose();
        flowX = tf.keep(tempFlowX);

        const scaledByBottom = tf.less(flowY.slice([0, 0], [M, N]), 0).cast('float32');
        const scaledByTop    = tf.greater(flowY.slice([0, 1], [M, N]), 0).cast('float32');
        const scaledYB = flowY.slice([0, 0], [M, N]).mul(scaledByBottom).mul(scale);
        const scaledYT = flowY.slice([0, 1], [M, N]).mul(scaledByTop).mul(scale);

        let tempFlowY = applyPatchTo2DTensor(flowY, scaledYB, 0, 0);
        flowY.dispose();
        flowY = tf.keep(tempFlowY);

        tempFlowY = applyPatchTo2DTensor(flowY, tf.zeros([M, 1]), 0, N);
        flowY.dispose();
        flowY = tf.keep(tempFlowY);

        tempFlowY = applyPatchTo2DTensor(flowY, flowY.slice([0, 1], [M, N]).add(scaledYT), 0, 1);
        flowY.dispose();
        flowY = tf.keep(tempFlowY);

        // --- Water update ---
        let waterDelta = flowX.slice([0, 0], [M, N])
                        .add(flowY.slice([0, 0], [M, N]))
                        .sub(flowX.slice([1, 0], [M, N]))
                        .sub(flowY.slice([0, 1], [M, N]));
        waterDelta = waterDelta.div(dx2Bydt);
        const newWater2 = water.add(waterDelta);
        water.dispose();
        water = tf.keep(newWater2);

        // --- Boundary cleanup ---
        let tempWater = applyPatchTo2DTensor(water, tf.zeros([1, N]), 0, 0);
        water.dispose();
        water = tf.keep(tempWater);

        tempWater = applyPatchTo2DTensor(water, tf.zeros([1, N]), M - 1, 0);
        water.dispose();
        water = tf.keep(tempWater);

        tempWater = applyPatchTo2DTensor(water, tf.zeros([M, 1]), 0, 0);
        water.dispose();
        water = tf.keep(tempWater);

        tempWater = applyPatchTo2DTensor(water, tf.zeros([M, 1]), 0, N - 1);
        water.dispose();
        water = tf.keep(tempWater);
    });
    // Step counter
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

    // console.log(step);
    // console.log(depthArray);
}

///////////////////
// UI contruction
///////////////////
const objsToRayCast = [];
let paused = false;
//const stepText = new ThreeMeshUI.Text( { content: "step: 0" } );

function makePanel() {
    while(objsToRayCast.length > 0) {
        objsToRayCast.pop();
    }

	// Container block, in which we put the two buttons.
	// We don't define width and height, it will be set automatically from the children's dimensions
	// Note that we set contentDirection: "row-reverse", in order to orient the buttons horizontally
    /*
	const container = new ThreeMeshUI.Block( {
		justifyContent: 'center',
		contentDirection: 'row-reverse',
		fontFamily: FontJSON,
		fontTexture: FontImage,
		fontSize: 1.4,
		padding: 0.2,
		borderRadius: 1.1
	} );

	container.position.set( 0, 11, 0 );
	container.rotation.x = -0.55;
	scene.add( container );

	// BUTTONS

	// We start by creating objects containing options that we will use with the two buttons,
	// in order to write less code.

	const buttonOptions = {
		width: 10.0,
		height: 2.0,
		justifyContent: 'center',
		offset: 0.05,
		margin: 0.02,
		borderRadius: 0.075
	};

	// Options for component.setupState().
	// It must contain a 'state' parameter, which you will refer to with component.setState( 'name-of-the-state' ).

	const hoveredStateAttributes = {
		state: 'hovered',
		attributes: {
			offset: 0.035,
			backgroundColor: new THREE.Color( 0x999999 ),
			backgroundOpacity: 1,
			fontColor: new THREE.Color( 0xffffff )
		},
	};

	const idleStateAttributes = {
		state: 'idle',
		attributes: {
			offset: 0.035,
			backgroundColor: new THREE.Color( 0x666666 ),
			backgroundOpacity: 0.3,
			fontColor: new THREE.Color( 0xffffff )
		},
	};

	// Buttons creation, with the options objects passed in parameters.

	const switchMapBtn = new ThreeMeshUI.Block( buttonOptions );
	const pauseUnpauseBtn = new ThreeMeshUI.Block( buttonOptions );

	// Add text to buttons
    const switchMapText = new ThreeMeshUI.Text( { content: "- " + terrainList[terrainIndex] + " +" } );
	switchMapBtn.add(switchMapText);

	pauseUnpauseBtn.add(stepText);

	// Create states for the buttons.
	// In the loop, we will call component.setState( 'state-name' ) when mouse hover or click

	const selectedAttributes = {
		offset: 0.02,
		backgroundColor: new THREE.Color( 0x777777 ),
		fontColor: new THREE.Color( 0x222222 )
	};

	switchMapBtn.setupState( {
		state: 'selected',
		attributes: selectedAttributes,
		onSet: async () => {
            terrainIndex = (terrainIndex + 1) % terrainList.length;
            loadTerrainResult = await loadTerrain(terrainList[terrainIndex]);
            switchMapText.set({ content: "- " + terrainList[terrainIndex] + " +"});
            initTensors(loadTerrainResult);
            waterMesh = setupScene();

            paused = false;
		}
	} );
	switchMapBtn.setupState( hoveredStateAttributes );
	switchMapBtn.setupState( idleStateAttributes );

	pauseUnpauseBtn.setupState( {
		state: 'selected',
		attributes: selectedAttributes,
		onSet: () => {
            paused = !paused;
		}
	} );
	pauseUnpauseBtn.setupState( hoveredStateAttributes );
	pauseUnpauseBtn.setupState( idleStateAttributes );

	container.add( switchMapBtn, pauseUnpauseBtn );
	objsToRayCast.push( switchMapBtn, pauseUnpauseBtn );
    */

    const pauseButton = document.getElementById("pause");
    pauseButton.addEventListener("click", () => {
            paused = !paused;
            pauseButton.textContent = paused ? "Resume" : "Pause ";
		});

    const switchSceneButton = document.getElementById("switchScene");
    switchSceneButton.addEventListener("click", async () => {
            terrainIndex = (terrainIndex + 1) % terrainList.length;
            loadTerrainResult = await loadTerrain(terrainList[terrainIndex]);
            switchSceneButton.textContent = "- " + terrainList[terrainIndex] + " +";
            initTensors(loadTerrainResult);
            waterMesh = setupScene();

            paused = false;
		});
    switchSceneButton.textContent = "- " + terrainList[0] + " +";
}

makePanel();
const stepLabel = document.getElementById("stepLabel");

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
mouse.x = mouse.y = null;

let selectState = false;

window.addEventListener( 'pointermove', ( event ) => {
	mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	mouse.y = -( event.clientY / window.innerHeight ) * 2 + 1;
} );

window.addEventListener( 'pointerdown', () => {
	selectState = true;
} );

window.addEventListener( 'pointerup', () => {
	selectState = false;
} );

window.addEventListener( 'touchstart', ( event ) => {
	selectState = true;
	mouse.x = ( event.touches[ 0 ].clientX / window.innerWidth ) * 2 - 1;
	mouse.y = -( event.touches[ 0 ].clientY / window.innerHeight ) * 2 + 1;
} );

window.addEventListener( 'touchend', () => {
	selectState = false;
	mouse.x = null;
	mouse.y = null;
} );

function raycast() {

	return objsToRayCast.reduce( ( closestIntersection, obj ) => {

		const intersection = raycaster.intersectObject( obj, true );

		if ( !intersection[ 0 ] ) return closestIntersection;

		if ( !closestIntersection || intersection[ 0 ].distance < closestIntersection.distance ) {

			intersection[ 0 ].object = obj;

			return intersection[ 0 ];

		}

		return closestIntersection;

	}, null );

}

function updateButtons() {

	// Find closest intersecting object

	let intersect;

	if ( renderer.xr.isPresenting ) {

		vrControl.setFromController( 0, raycaster.ray );

		intersect = raycast();

		// Position the little white dot at the end of the controller pointing ray
		if ( intersect ) vrControl.setPointerAt( 0, intersect.point );

	} else if ( mouse.x !== null && mouse.y !== null ) {

		raycaster.setFromCamera( mouse, camera );

		intersect = raycast();

	}

	// Update targeted button state (if any)

	if ( intersect && intersect.object.isUI ) {

		if ( selectState ) {

			// Component.setState internally call component.set with the options you defined in component.setupState
			intersect.object.setState( 'selected' );

		} else {

			// Component.setState internally call component.set with the options you defined in component.setupState
			intersect.object.setState( 'hovered' );

		}

	}

    // Update non-targeted buttons state

	objsToRayCast.forEach( ( obj ) => {

		if ( ( !intersect || obj !== intersect.object ) && obj.isUI ) {

			// Component.setState internally call component.set with the options you defined in component.setupState
			obj.setState( 'idle' );

		}

	} );

    // Update step counter
    stepLabel.textContent = "step " + step;
}
///////////////
/// Main Loop
///////////////

function animate() {
    const drawFreq = 10;
    if (step % drawFreq == 0)
    {
        drawWater();
    }
    if (!paused)
    {
        simulate();
    }
  
    controls.update();
    updateButtons();
    ThreeMeshUI.update();
    renderer.render(scene, camera);
  }