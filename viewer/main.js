// Studio viewer: white cyc, tiled grey floor, key light from above-left.
//
// The splats themselves are NOT lit by the scene light — their appearance is
// baked into the spherical harmonics at capture time (that's where the real
// paint reflections come from). The light exists to shade the floor and cast
// the contact shadow that grounds the car. Commercial splat configurators do
// exactly this.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

const params = new URLSearchParams(location.search);
const vehicle = params.get("v");
const msg = document.getElementById("msg");
const hud = document.getElementById("hud");

if (!vehicle) {
  listVehicles();
} else {
  start(vehicle).catch((err) => {
    console.error(err);
    msg.innerHTML = `Could not load <b>${vehicle}</b>.<br><small>${err.message}</small>`;
  });
}

async function listVehicles() {
  try {
    const { vehicles } = await (await fetch("/vehicles")).json();
    msg.innerHTML = vehicles.length
      ? "Pick a vehicle:<br><br>" +
        vehicles
          .map((v) => `<a href="?v=${encodeURIComponent(v.folder)}">${v.name}</a>`)
          .join(" &nbsp;·&nbsp; ")
      : "No vehicles yet — scan one from the capture page.";
  } catch {
    msg.textContent = "Server unreachable.";
  }
}

async function start(name) {
  const info = await (await fetch(`/vehicles/${encodeURIComponent(name)}`)).json();

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.getElementById("stage").appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  scene.fog = new THREE.Fog(0xffffff, 9, 22); // floor fades out instead of ending

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 100);
  camera.position.set(3.4, 2.0, 3.4);

  buildStudio(scene);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 0.55, 0);
  controls.minDistance = 1.2;
  controls.maxDistance = 14;
  controls.maxPolarAngle = Math.PI / 2 - 0.02; // never orbit under the floor
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.7;
  // grabbing should stop the turntable, not fight it
  controls.addEventListener("start", () => (controls.autoRotate = false));

  const viewer = new GaussianSplats3D.DropInViewer({
    // GPU-accelerated sort renders NOTHING — silently, no error — on this
    // laptop's Intel Arc iGPU through ANGLE/D3D11, and Chrome picks the
    // integrated GPU over the discrete one by default, so enabling it ships a
    // blank white studio to exactly the machine we develop on. CPU sort in a
    // worker is slower but correct everywhere; revisit only with a per-device
    // capability probe, never as a blanket default.
    gpuAcceleratedSort: false,
    sharedMemoryForWorkers: false, // no COOP/COEP headers on the plain LAN server
  });
  scene.add(viewer);

  await loadSplats(viewer, name, "splat");
  msg.hidden = true;
  hud.hidden = false;
  fillHud(info);
  wireControls(viewer, controls, name);

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function loadSplats(viewer, name, fmt) {
  // Splats arrive in the canonical frame (+z up, ground at z=0); three.js is
  // y-up, so the scene rotates them rather than the exporter rewriting files
  // that must stay standard for SuperSplat/Blender.
  return viewer.addSplatScene(
    `/vehicles/${encodeURIComponent(name)}/model?fmt=${fmt}`,
    {
      format: fmt === "splat" ? GaussianSplats3D.SceneFormat.Splat : GaussianSplats3D.SceneFormat.Ply,
      rotation: new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0), -Math.PI / 2
      ).toArray(),
      showLoadingUI: false,
      progressiveLoad: true, // low-detail shell first, detail streams in
    }
  );
}

function buildStudio(scene) {
  // Tiled grey floor. A canvas texture keeps the page self-contained (no CDN
  // image), and tiles read as a studio rather than an infinite void.
  const tile = document.createElement("canvas");
  tile.width = tile.height = 256;
  const ctx = tile.getContext("2d");
  ctx.fillStyle = "#d8dade";
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = "#c2c5ca";
  ctx.lineWidth = 4;
  ctx.strokeRect(0, 0, 256, 256);
  ctx.fillStyle = "#cfd2d7";
  ctx.fillRect(4, 4, 248, 248);

  const texture = new THREE.CanvasTexture(tile);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(24, 24);
  texture.anisotropy = 8;

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(48, 48),
    new THREE.MeshStandardMaterial({ map: texture, roughness: 0.85, metalness: 0.0 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  scene.add(new THREE.HemisphereLight(0xffffff, 0xd0d3d8, 0.75));

  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(-4.5, 7.5, 3.5); // above, angled
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 22;
  const s = 4.5;
  Object.assign(key.shadow.camera, { left: -s, right: s, top: s, bottom: -s });
  key.shadow.radius = 4;
  scene.add(key);

  // Splats don't cast shadows, so a soft blob grounds the car. Without it the
  // vehicle reads as floating above the floor rather than parked on it.
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1.25, 48),
    new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.18, depthWrite: false,
    })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.004; // just above the floor, avoids z-fighting
  shadow.scale.set(1.5, 1.0, 1);
  scene.add(shadow);
}

function fillHud(info) {
  document.getElementById("vname").textContent = info.name;
  const updated = info.updated_ts
    ? new Date(info.updated_ts * 1000).toLocaleString()
    : "—";
  document.getElementById("vsub").textContent =
    `${info.observations} observation${info.observations === 1 ? "" : "s"} · ${updated}`;
  document.getElementById("sSplats").textContent = info.splats.toLocaleString();
  document.getElementById("sObs").textContent =
    (info.observed_fraction * 100).toFixed(1) + "%";
  document.getElementById("sConf").textContent = info.mean_confidence.toFixed(2);
}

function wireControls(viewer, controls, name) {
  const prov = document.getElementById("cProv");
  const spin = document.getElementById("cSpin");

  prov.onchange = async () => {
    document.getElementById("legend").classList.toggle("on", prov.checked);
    prov.disabled = true;
    // The provenance twin is a separate export, so swap the loaded scene.
    await viewer.removeSplatScene(0);
    await loadSplats(viewer, name, prov.checked ? "provenance" : "splat");
    prov.disabled = false;
  };

  spin.onchange = () => (controls.autoRotate = spin.checked);
  controls.addEventListener("start", () => (spin.checked = false));
}
