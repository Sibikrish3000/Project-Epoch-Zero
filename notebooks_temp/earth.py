import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72
from IPython.display import HTML, display
import json

class NasaEyesWebGL:
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.R_EARTH = 6378.137

    def generate_simulation_data(self):
        """Pre-calculates physics in Python to send to JS."""
        print("[-] Propagating SGP4 Physics for WebGL Engine...")
        
        # 1. Generate 24 hours of data at 30-second intervals
        # (Three.js will interpolate smoothly between these points)
        steps = 200 
        times = np.linspace(0, 200, steps) # 200 minutes roughly
        
        satA = Satrec.twoline2rv(*self.forecaster.tle_A, WGS72)
        satB = Satrec.twoline2rv(*self.forecaster.tle_B, WGS72)
        
        # Calculate Epoch Offset
        start_time = self.forecaster.start_time
        jd_epoch = satA.jdsatepoch + satA.jdsatepochF
        ts_epoch = pd.Timestamp(jd_epoch - 2440587.5, unit='D')
        offset_min = (start_time - ts_epoch).total_seconds() / 60.0
        
        pos_A = []
        pos_B = []
        
        for t in times:
            jd = jd_epoch + (t + offset_min) / 1440.0
            e, rA, v = satA.sgp4_array(np.array([jd]), np.array([0.0]))
            e, rB, v = satB.sgp4_array(np.array([jd]), np.array([0.0]))
            
            # Convert to Km (Standard Three.js scale: 1 unit = 1000 km)
            # This helps precision in WebGL
            scale = 0.001 
            pos_A.append([rA[0][0]*scale, rA[0][2]*scale, -rA[0][1]*scale]) # Swap Y/Z for WebGL coords
            pos_B.append([rB[0][0]*scale, rB[0][2]*scale, -rB[0][1]*scale])

        return {
            "times": times.tolist(),
            "trackA": pos_A,
            "trackB": pos_B,
            "start_time": str(start_time)
        }

    def render(self):
        data = self.generate_simulation_data()
        json_data = json.dumps(data)
        
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background-color: #000; font-family: 'Courier New', monospace; }}
                #hud {{
                    position: absolute; top: 20px; left: 20px; color: #00ffff; pointer-events: none;
                    text-shadow: 0px 0px 5px #00ffff;
                }}
                #range-box {{
                    font-size: 24px; font-weight: bold; border: 1px solid #00ffff; padding: 10px; background: rgba(0,0,0,0.5);
                }}
                #controls {{
                    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
                    display: flex; gap: 20px;
                }}
                button {{
                    background: transparent; border: 1px solid #00ffff; color: #00ffff; 
                    padding: 10px 20px; font-family: 'Courier New'; font-weight: bold; cursor: pointer;
                    transition: all 0.3s;
                }}
                button:hover {{ background: #00ffff; color: black; }}
            </style>
            <!-- Load Three.js from CDN -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        </head>
        <body>
            <div id="hud">
                <div id="range-box">INITIALIZING LINK...</div>
                <div id="status" style="margin-top:5px; font-size:12px;">SYSTEM: OPTICAL TRACKING ACTIVE</div>
            </div>
            
            <div id="controls">
                <button onclick="togglePlay()">⏯ PLAY/PAUSE</button>
                <button onclick="resetSim()">↺ RESET</button>
            </div>

            <script>
                // --- 1. SETUP DATA ---
                const simData = {json_data};
                const trackA = simData.trackA;
                const trackB = simData.trackB;
                const totalFrames = trackA.length;
                
                let currentFrame = 0;
                let isPlaying = true;
                const R_EARTH = 6.378; // Scaled R (1 unit = 1000km)

                // --- 2. SCENE SETUP ---
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);

                // --- 3. LIGHTING (Sunlight) ---
                const ambientLight = new THREE.AmbientLight(0x333333);
                scene.add(ambientLight);
                const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
                sunLight.position.set(50, 20, 30);
                scene.add(sunLight);

                // --- 4. EARTH (High Res Textures) ---
                const textureLoader = new THREE.TextureLoader();
                const earthGeo = new THREE.SphereGeometry(R_EARTH, 64, 64);
                
                // Textures from public URLs
                const earthMat = new THREE.MeshPhongMaterial({{
                    map: textureLoader.load('https://upload.wikimedia.org/wikipedia/commons/c/c3/Solarsystemscope_texture_8k_earth_daymap.jpg'),
                    bumpMap: textureLoader.load('https://upload.wikimedia.org/wikipedia/commons/c/c3/Solarsystemscope_texture_8k_earth_daymap.jpg'), // re-using for bump to save load time
                    bumpScale: 0.05,
                    specularMap: textureLoader.load('https://upload.wikimedia.org/wikipedia/commons/c/c3/Solarsystemscope_texture_8k_earth_daymap.jpg'), // smooth oceans
                    specular: new THREE.Color('grey')
                }});
                const earth = new THREE.Mesh(earthGeo, earthMat);
                scene.add(earth);

                // Atmosphere Glow
                const atmoGeo = new THREE.SphereGeometry(R_EARTH * 1.02, 64, 64);
                const atmoMat = new THREE.MeshBasicMaterial({{
                    color: 0x00aaff,
                    transparent: true,
                    opacity: 0.15,
                    side: THREE.BackSide
                }});
                const atmosphere = new THREE.Mesh(atmoGeo, atmoMat);
                scene.add(atmosphere);

                // Starfield
                const starGeo = new THREE.BufferGeometry();
                const starCount = 2000;
                const posArray = new Float32Array(starCount * 3);
                for(let i=0; i<starCount*3; i++) {{
                    posArray[i] = (Math.random() - 0.5) * 200; // Large spread
                }}
                starGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
                const starMat = new THREE.PointsMaterial({{size: 0.1, color: 0xffffff, transparent: true, opacity: 0.8}});
                const stars = new THREE.Points(starGeo, starMat);
                scene.add(stars);

                // --- 5. SATELLITES ---
                // Sat A (You) - Diamond
                const satAGeo = new THREE.OctahedronGeometry(0.15);
                const satAMat = new THREE.MeshBasicMaterial({{ color: 0x00ffff, wireframe: true }});
                const meshA = new THREE.Mesh(satAGeo, satAMat);
                scene.add(meshA);

                // Sat B (Target) - Cube
                const satBGeo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
                const satBMat = new THREE.MeshBasicMaterial({{ color: 0xffaa00 }});
                const meshB = new THREE.Mesh(satBGeo, satBMat);
                scene.add(meshB);

                // Trails (BufferGeometry for updates)
                const trailSize = 100;
                const trailB_Geo = new THREE.BufferGeometry();
                const trailB_Pos = new Float32Array(trailSize * 3).fill(0);
                trailB_Geo.setAttribute('position', new THREE.BufferAttribute(trailB_Pos, 3));
                const trailB = new THREE.Line(trailB_Geo, new THREE.LineBasicMaterial({{ color: 0xffaa00, transparent: true, opacity: 0.5 }}));
                scene.add(trailB);

                // --- 6. ORBITAL CONTROLS (Manual Camera) ---
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                
                // Initial Camera Position (Chase View)
                // We don't set it yet, it updates in loop

                // --- 7. ANIMATION LOOP ---
                function update() {{
                    if (isPlaying) {{
                        currentFrame += 0.2; // Slow speed factor
                        if (currentFrame >= totalFrames - 1) currentFrame = 0;
                    }}

                    // Interpolation indices
                    const idx = Math.floor(currentFrame);
                    const nextIdx = Math.min(idx + 1, totalFrames - 1);
                    const alpha = currentFrame - idx;

                    // Interpolate Position A
                    const pA1 = new THREE.Vector3(...trackA[idx]);
                    const pA2 = new THREE.Vector3(...trackA[nextIdx]);
                    const posA = new THREE.Vector3().lerpVectors(pA1, pA2, alpha);

                    // Interpolate Position B
                    const pB1 = new THREE.Vector3(...trackB[idx]);
                    const pB2 = new THREE.Vector3(...trackB[nextIdx]);
                    const posB = new THREE.Vector3().lerpVectors(pB1, pB2, alpha);

                    // Update Meshes
                    meshA.position.copy(posA);
                    meshB.position.copy(posB);

                    // Update Trail B
                    const positions = trailB.geometry.attributes.position.array;
                    // Shift values back
                    for (let i = 0; i < (trailSize - 1) * 3; i++) {{
                        positions[i] = positions[i + 3];
                    }}
                    // Add new point
                    positions[(trailSize - 1) * 3] = posB.x;
                    positions[(trailSize - 1) * 3 + 1] = posB.y;
                    positions[(trailSize - 1) * 3 + 2] = posB.z;
                    trailB.geometry.attributes.position.needsUpdate = true;

                    // --- CINEMATIC CAMERA LOGIC ---
                    // Camera sits slightly above/behind Sat A, looking at Earth center
                    // 1. Get vector from Earth Center (0,0,0) to Sat A
                    const radVec = posA.clone().normalize();
                    // 2. Offset camera "Up" (Radially) and slightly back
                    const camOffset = radVec.clone().multiplyScalar(1.0); // 1000km up
                    
                    // We simply lock controls target to Sat A, allowing user to rotate around it
                    controls.target.copy(posA);
                    
                    // If we want a fixed chase cam (comment out to allow manual rotation):
                    // camera.position.copy(posA).add(radVec.multiplyScalar(2.0)); 
                    
                    // Update HUD
                    const dist = posA.distanceTo(posB) * 1000; // Convert back to KM
                    const rangeBox = document.getElementById('range-box');
                    rangeBox.innerHTML = "RANGE: " + dist.toFixed(2) + " KM";
                    rangeBox.style.color = dist < 10 ? "#ff0000" : "#00ffff";
                    rangeBox.style.borderColor = dist < 10 ? "#ff0000" : "#00ffff";

                    controls.update();
                    renderer.render(scene, camera);
                    requestAnimationFrame(update);
                }}

                // Start
                camera.position.set(10, 10, 10); // Initial look
                update();

                // Functions
                window.togglePlay = function() {{ isPlaying = !isPlaying; }};
                window.resetSim = function() {{ currentFrame = 0; trailB_Pos.fill(0); }};
                
                // Handle Resize
                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """
        return display(HTML(html_code))

# Execution
# Ensure your 'forecaster' object from previous steps is ready
webgl_viz = NasaEyesWebGL(forecaster)
webgl_viz.render()