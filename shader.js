import * as THREE from 'three';

function loadTexture(url) {
    return new Promise((resolve, reject) => {
        const loader = new THREE.TextureLoader();
        loader.load(url, resolve, undefined, reject);
    });
}

export async function LoadMaterial()
{
    const terrainTex = await loadTexture('tex_stone.png');
    terrainTex.wrapS = THREE.RepeatWrapping;
    terrainTex.wrapT = THREE.RepeatWrapping;
    terrainTex.repeat.set(4, 4);  // Repeats 4 times horizontally and vertically
    terrainTex.needsUpdate = true;

    const waterTex = await loadTexture('tex_water.jpg');
    waterTex.wrapS = THREE.RepeatWrapping;
    waterTex.wrapT = THREE.RepeatWrapping;
    waterTex.repeat.set(4, 4);
    waterTex.needsUpdate = true;

    const waterMaterial = new THREE.ShaderMaterial({
        vertexShader: `
            uniform sampler2D heightMap;
            uniform sampler2D terrainMap;
            
            uniform int m;
            uniform int n;

            varying float waterDepth;
            varying float xGrid, yGrid;

            varying vec3 vertex_local;

            void main() {
                // Sample height from texture
                float height = texture2D(heightMap, uv).r;

                int ix = int(round(float(2 * n - 1) * uv.x));
                int iy = int(round(float(2 * m - 1) * uv.y));

                vec3 displaced = vec3((ix + 1) / 2, height, (iy + 1) / 2);

                gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);

                waterDepth = height - texture2D(terrainMap, uv).r;
                xGrid = float(ix / 2);
                yGrid = float(iy / 2);
                vertex_local = displaced;
            }
        `,
        fragmentShader:  `
            varying float waterDepth;
            varying float xGrid, yGrid;
            varying vec3 vertex_local;

            uniform sampler2D terrainDiffuseMap;
            uniform sampler2D waterDiffuseMap;

            uniform sampler2D vxMap;
            uniform sampler2D vyMap;
            uniform int m;
            uniform int n;
            uniform float uTime;

            void main() {
                bool facingX = abs(round(xGrid) - xGrid) != 0.0;
                bool facingY = abs(round(yGrid) - yGrid) != 0.0;
                bool isVertical = facingX || facingY;

                float fm = float(m);
                float fn = float(n);
                float vxSampleU = vertex_local.x / fn;
                float vxSampleV = vertex_local.z / (fm + 1.0) + 0.5 / (fm + 1.0);

                float vySampleU = vertex_local.x / (fn + 1.0) + 0.5 / (fn + 1.0);
                float vySampleV = vertex_local.z / fm;

                float vx = texture2D(vxMap, vec2(vxSampleU, vxSampleV)).r;
                float vy = texture2D(vyMap, vec2(vySampleU, vySampleV)).r;

                if (isVertical)
                {
                    if (facingX)
                    {
                        vec2 uv = vertex_local.zy;
                        if (abs(vy) > 0.0)
                        {
                            vec2 displacement = (uTime - floor(uTime)) * vec2(0.0, max(abs(vy), 1.0));
                            gl_FragColor = texture2D(waterDiffuseMap, uv + displacement);
                        }
                        else
                        {
                            gl_FragColor = texture2D(terrainDiffuseMap, uv);
                        }
                    }
                    else
                    {
                        vec2 uv = vertex_local.xy;
                        if (abs(vx) > 0.0)
                        {
                            vec2 displacement = (uTime - floor(uTime)) * vec2(0.0, max(abs(vx), 1.0));
                            gl_FragColor = texture2D(waterDiffuseMap, uv + displacement);
                        }
                        else
                        {
                            
                            gl_FragColor = texture2D(terrainDiffuseMap, uv);
                        }
                    }
                }
                else 
                if (waterDepth < 0.001)
                {
                    gl_FragColor = texture2D(terrainDiffuseMap, vertex_local.xz);
                }
                else
                {
                    vec2 displacement = (uTime - floor(uTime)) * vec2(-vy, -vx);

                    vec4 color = texture2D(waterDiffuseMap, vertex_local.xz + displacement);
                    color.xy = color.xy * mix(1.0, 0.7, waterDepth);
                    gl_FragColor = color;
                }
            }
        `,
        uniforms: {
            terrainDiffuseMap: {value: terrainTex},
            waterDiffuseMap: {value: waterTex},
            uTime: {value: 0.0}
        },
        side: THREE.DoubleSide,
        //wireframe: true // for debug
    });

    return waterMaterial;
}
