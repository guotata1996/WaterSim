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

    const waterTex = await loadTexture('tex_water_1.jpg');
    waterTex.wrapS = THREE.RepeatWrapping;
    waterTex.wrapT = THREE.RepeatWrapping;
    waterTex.repeat.set(4, 4);
    waterTex.needsUpdate = true;

    const waterTex2 = await loadTexture('tex_water_2.jpg');
    waterTex2.wrapS = THREE.RepeatWrapping;
    waterTex2.wrapT = THREE.RepeatWrapping;
    waterTex2.repeat.set(4, 4);
    waterTex2.needsUpdate = true;

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

                int px = min(n-1, max(1, (ix + 1) / 2));
                int py = min(m-1, max(1, (iy + 1) / 2));
                vec3 displaced = vec3(px, height, py); // hide those quads on boundary

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
            uniform sampler2D waterDiffuseMap1;
            uniform sampler2D waterDiffuseMap2;

            uniform sampler2D vxMap;
            uniform sampler2D vyMap;
            uniform int m;
            uniform int n;
            uniform float uTime;

            float samplevxMap(float localX, float localZ)
            {
                float fm = float(m);
                float fn = float(n);
                float u = localX / fn;
                float v = localZ / (fm + 1.0) + 0.5 / (fm + 1.0);
                return texture2D(vxMap, vec2(u, v)).r;
            }

            float samplevyMap(float localX, float localZ)
            {
                float fm = float(m);
                float fn = float(n);
                float u = localX / (fn + 1.0) + 0.5 / (fn + 1.0);
                float v = localZ / fm;
                return texture2D(vyMap, vec2(u, v)).r;
            }

            vec4 fadeWaterTextures(vec2 aUV, vec2 aVelocity)
            {
                const float HalfCycleTime = 1.0;
                float t1 = uTime / (2.0 * HalfCycleTime);
                float t2 = t1 + 0.5;
                vec2 displacement1 = (t1 - floor(t1)) * aVelocity;
                vec2 displacement2 = (t2 - floor(t2)) * aVelocity;
                vec4 color1 = texture2D(waterDiffuseMap1, aUV + displacement1);
                vec4 color2 = texture2D(waterDiffuseMap2, aUV + displacement2);

                float tInCycle = t1 - floor(t1);
                float weight1 = 2.0 * min(tInCycle, 1.0 - tInCycle);
                return mix(color1, color2, 1.0 - weight1);
            }

            void main() {
                bool facingX = abs(round(xGrid) - xGrid) != 0.0;
                bool facingY = abs(round(yGrid) - yGrid) != 0.0;
                bool isVertical = facingX || facingY;

                if (isVertical)
                {
                    if (facingX)
                    {
                        vec2 uv = vertex_local.zy;
                        float vy = samplevyMap(vertex_local.x, vertex_local.z);
                        if (abs(vy) > 0.0)
                        {
                            vec4 color = fadeWaterTextures(uv, vec2(0.0, max(abs(vy), 1.0)));
                            color.xy = color.xy * mix(1.0, 0.7, waterDepth);
                            gl_FragColor = color;
                        }
                        else
                        {
                            gl_FragColor = texture2D(terrainDiffuseMap, uv);
                        }
                    }
                    else
                    {
                        vec2 uv = vertex_local.xy;
                        float vx = samplevxMap(vertex_local.x, vertex_local.z);
                        if (abs(vx) > 0.0)
                        {
                            vec4 color = fadeWaterTextures(uv, vec2(0.0, max(abs(vx), 1.0)));
                            color.xy = color.xy * mix(1.0, 0.7, waterDepth);
                            gl_FragColor = color;
                        }
                        else
                        {
                            gl_FragColor = texture2D(terrainDiffuseMap, uv);
                        }
                    }
                }
                else if (waterDepth < 0.001)
                {
                    gl_FragColor = texture2D(terrainDiffuseMap, vertex_local.xz);
                }
                else
                {
                    float fm = float(m);
                    float fn = float(n);
                    float vxSampleUMinus = max(floor(vertex_local.x - 0.5) + 0.5, 0.0);
                    float vxSampleUPlus = min(vxSampleUMinus + 1.0, fn);
                    float vxSampleVMinus = max(floor(vertex_local.z), 0.0);
                    float vxSampleVPlus = min(vxSampleVMinus + 1.0, fm);
                    float vx00 = samplevxMap(vxSampleUMinus, vxSampleVMinus);
                    float vx01 = samplevxMap(vxSampleUMinus, vxSampleVPlus);
                    float vx10 = samplevxMap(vxSampleUPlus, vxSampleVMinus);
                    float vx11 = samplevxMap(vxSampleUPlus, vxSampleVPlus);
                    float vxSmooth = mix(
                        mix(vx00, vx01, vertex_local.z - vxSampleVMinus),
                        mix(vx10, vx11, vertex_local.z - vxSampleVMinus),
                        vertex_local.x - vxSampleUMinus);

                    float vySampleUMinus = max(floor(vertex_local.x), 0.0);
                    float vySampleUPlus = min(vySampleUMinus + 1.0, fn);
                    float vySampleVMinus = max(floor(vertex_local.z - 0.5) + 0.5, 0.0);
                    float vySampleVPlus = min(vySampleVMinus + 1.0, fm);
                    float vy00 = samplevyMap(vySampleUMinus, vySampleVMinus);
                    float vy01 = samplevyMap(vySampleUMinus, vySampleVPlus);
                    float vy10 = samplevyMap(vySampleUPlus, vySampleVMinus);
                    float vy11 = samplevyMap(vySampleUPlus, vySampleVPlus);
                    float vySmooth = mix(
                        mix(vy00, vy01, vertex_local.z - vySampleVMinus),
                        mix(vy10, vy11, vertex_local.z - vySampleVMinus),
                        vertex_local.x - vySampleUMinus);

                    vec2 v = vec2(-vySmooth, -vxSmooth);
                    const float ClampV = 0.5;
                    if (length(v) > ClampV)
                    {
                        v = v / length(v) * ClampV;
                    }

                    vec4 color = fadeWaterTextures(vertex_local.xz, v);
                    color.xy = color.xy * mix(1.0, 0.7, waterDepth);
                    gl_FragColor = color;
                }
            }
        `,
        uniforms: {
            terrainDiffuseMap: {value: terrainTex},
            waterDiffuseMap1: {value: waterTex},
            waterDiffuseMap2: {value: waterTex2},
            uTime: {value: 0.0}
        },
        side: THREE.DoubleSide,
        //wireframe: true // for debug
    });

    return waterMaterial;
}
