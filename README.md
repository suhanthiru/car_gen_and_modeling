# cargen — Dynamic Vehicle 3D Reconstruction

One photo of a vehicle → a full interactive 3D model (3D Gaussian Splatting), with a generative
prior guessing unseen regions. Every later photo or walk-around video — from any phone on the
network — fuses real evidence into the same persistent model, replacing guesses with reality
while confirmed regions stay locked. Per-splat `provenance` / `confidence` metadata is the
arbitration mechanism.

## Layout

| Path | What |
|---|---|
| `cargen/core` | Splat cloud (SoA + fusion metadata), cameras/Sim(3), persistent `VehicleAsset` |
| `cargen/segmentation` | Vehicle masking — stub / rembg |
| `cargen/prior_generation` | Image→3D prior — stub sedan / TRELLIS / SF3D / Tripo API / custom slot; mesh→splats |
| `cargen/feature_matching` | ORB (real) / LightGlue (Milestone B) |
| `cargen/pose_estimation` | PnP + Sim(3) registration with confidence gating; video tracker |
| `cargen/video` | Motion-magnitude frame sampler |
| `cargen/fusion_engine` | Render→residual→dirty-flag→densify→update arbitration; renderers |
| `cargen/reid` | Duplicate detection embeddings — histogram (real) / DINOv2 |
| `cargen/export` | Standard 3DGS `.ply`, `.splat`, provenance overlay |
| `server/` | FastAPI: ingestion, per-vehicle queue, merge + events, named storage |
| `clients/capture` | Phone capture page (LAN) |
| `viewer/` | Browser splat viewer — studio floor, provenance overlay, turntable |
| `demo/` | Deterministic synthetic sedan fusion demo |

## Setup (CPU skeleton — works everywhere, no ML installs)

```powershell
pip install -e .[dev]
pytest                      # 170 tests, 80% coverage gate (currently 94%)
python demo\run_demo.py     # synthetic fusion regression
python -m server            # binds LAN; open the printed URL on your phone
```

Vehicle files land in `data\vehicles\<car-name>\` (named at capture time):

```
data/vehicles/bobs-civic/
├── manifest.json          identity, aliases, observation log, stats
├── cloud.npz              the splats
├── embeddings.npy         re-ID vectors
├── observations/          your raw uploads, kept for re-fusing later
└── exports/
    ├── model.ply          standard 3DGS — opens in SuperSplat / Blender
    ├── model.splat        what the viewer streams
    └── model_provenance.ply   red = guessed, green = confirmed
```

Viewer: `http://<laptop-ip>:8000/viewer/?v=<car-name>`. Capture page: `http://<laptop-ip>:8000/`
(phone must be on the same Wi-Fi; the server is not reachable from the internet).

## Environment

The ML stack lives in a venv **on D:**, not in the global Store Python. Two reasons,
both learned the hard way: C: had 17 GB free (torch+CUDA alone is 4.5 GB, and
Milestone B's toolchain is ~8 GB more), and the global env runs numpy 2.x / OpenCV 5,
which much of the 2024-era ML ecosystem won't accept. The venv also keeps those pins
away from your other projects.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install "rembg[cpu]"
```

Always run via `.\.venv\Scripts\python.exe` (or activate the venv) — the global
interpreter has no CUDA torch and no rembg.

Verify: `GET /health` reports the backends actually loaded, e.g.
`{"segmenter": "RembgSegmenter", "prior": "StubPriorGenerator", ...}`.

Then pick a prior backend (`CARGEN_PRIOR_BACKEND`) — each adapter's module
docstring has its full install recipe:

| Backend | Quality | VRAM | License | Notes |
|---|---|---|---|---|
| `trellis` | best | 16 GB rec. (8 GB only w/ `low_vram`, unverified) | MIT | **Native Gaussians — no mesh round-trip.** Painful Windows install. Intended default on the 12–16 GB box. |
| `sf3d` | good | ~7 GB | Stability Community | Easy install; laptop fallback. Mesh → surface-sampled. |
| `tripo` | best | none (cloud) | commercial API | Needs `TRIPO_API_KEY`; photo leaves your machine. Ships untested. |
| `custom` | — | — | — | Point `CARGEN_CUSTOM_PRIOR` at your own `module:callable`. |
| `stub` | procedural sedan | none | — | Default; no ML installs needed. |

Backends that emit Gaussians natively override `PriorGenerator.generate_splats`;
everything else inherits the default mesh→surface-sample path.

## Milestone B installs (real fusion)

Visual Studio Build Tools (C++), CUDA Toolkit matching the torch build, then:

```powershell
pip install gsplat lightglue
```

Backend selection, storage root, `auto_merge` (default **off** → pending-approval mode), and
LAN bind live in `server/config.py` (env-overridable, `CARGEN_*`).

## Why the prior isn't photorealistic (and what would be)

A single photo cannot produce a photorealistic model — the far side isn't in the pixels, so
it's a statistical guess about sedans-in-general. Photorealism needs **many real views,
jointly optimised**. Four things stand between the prior and that:

1. **View-dependent appearance** — `GaussianCloud.sh_rest` (SH bands 1-3) now exists and
   round-trips through `.ply`, but a prior leaves it zero: one photo carries no evidence
   about how a surface looks from elsewhere. Zero SH = matte by construction; no moving
   highlights. Only a multi-view optimisation can fill these bands.
2. **Joint optimisation** — `fusion_engine/optimize.py` is deliberately *localized* (one
   frame, dirty splats only). Right for incremental updates, but it never sees two views at
   once, so it can't be forced into multi-view consistency. Photorealism needs ~7k-30k
   iterations over all views (`consolidate.py`, not yet built).
3. **Adaptive densification** — real 3DGS splits/clones by view-space gradient during
   training. Ours only fills holes, so fine detail has no mechanism to appear.
4. **gsplat** — blocked on the CUDA install (see below).

Expect >30 dB PSNR on held-out views with 50+ good frames. Under ~20 views, worse than the
prior. Photorealism is bought with capture effort and compute, not a better prior.

## Known gotchas

- **Image-to-3D models need object-centric framing, or they return a blob.** SF3D is trained
  on square, centred, subject-filling images and its own `run.py` calls
  `resize_foreground(image, 0.85)` first. Feeding a raw photo (car at ~20% of a 4:3 frame,
  off-centre) is a severe distribution shift: measured, it inflated a sedan's W/L from 0.446
  to 0.524 and an estate's from 0.517 to 0.671. `SF3DPriorGenerator._frame_subject` calls
  SF3D's own function so the framing matches by construction. **Check this first on any new
  prior backend** — it dwarfs every other quality knob.
- **Capture advice that follows from the above:** a three-quarter view beats a side profile
  (27.5% vs 12.6% observed from one photo, similar proportion accuracy), and matte/light paint
  beats dark gloss — a glossy car mirrors its surroundings and the prior bakes that in as albedo.

- **Image-to-3D output is view-aligned, not canonical.** Measured on SF3D: the same car
  photographed at azimuth 0.0 vs 1.2 rad comes back with its length axis along -z vs -x,
  tilted by the camera's elevation. `canonicalize_orientation` (PCA) puts it back on its
  own axes — without it the car sits tilted in the viewer and two scans of one vehicle
  cannot be merged. PCA still can't tell front from back, so merges may be 180° out until
  the phase-2 render verifier lands.
- **texture_baker built with `USE_CUDA=0` is CPU-only**, but SF3D hands it CUDA tensors.
  `_install_cpu_baker_bridge` moves them across; it self-disables if you ever rebuild
  texture_baker with the CUDA Toolkit.

- **The `stub` segmenter is a crude rectangle**, so with it the prior's paint gets tinted by the
  background and the silhouette carries a small halo of densified splats. It exists to keep the
  CPU path installable, not to be good. Any real evaluation wants `CARGEN_SEGMENTER=rembg`.
- **Densification never invents depth in open space** (`densify_reach`): new splats may only grow
  within a few px of geometry that already exists, inheriting its depth. Removing that guard
  reintroduces a flat slab of splats across the frame whenever the mask is sloppy — a monocular
  view has no depth for a pixel with nothing behind it.

- **Viewer GPU sort**: `gpuAcceleratedSort` is forced **off** in `viewer/main.js`. With it on, the
  splats render nothing — silently, no error — on Intel Arc iGPUs via ANGLE/D3D11, and Chrome
  prefers the integrated GPU over a discrete one by default. Don't re-enable without a
  per-device capability probe.
- **Off-network access**: the server binds the LAN only. For scanning away from home, put
  Tailscale on the laptop and phone — same enclosed setup, no ports opened, no code change.
- **Demo `observed%` ceiling**: the procedural sedan is built from overlapping boxes, so ~34% of
  its sampled surface is interior faces no camera can reach. `demo/synthetic.py` visibility-culls
  them; real priors emit outer shells and don't have this problem.
