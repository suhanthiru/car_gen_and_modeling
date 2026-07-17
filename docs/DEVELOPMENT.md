# Development notes

## Module map

Each stage is a swappable module with a documented interface, so a stub can be replaced by a
real model without touching orchestration.

| Path | What |
|---|---|
| `cargen/core` | Splat cloud (SoA arrays + fusion metadata + SH bands), cameras/Sim(3), persistent `VehicleAsset` |
| `cargen/segmentation` | Vehicle masking — stub / rembg |
| `cargen/prior_generation` | Image→3D prior — stub / SF3D / TRELLIS / Tripo / custom; canonicalisation; mesh→splats |
| `cargen/feature_matching` | ORB (real) / LightGlue |
| `cargen/pose_estimation` | PnP + Sim(3) registration with confidence gating; video tracker |
| `cargen/video` | Motion-magnitude frame sampler |
| `cargen/fusion_engine` | Render→residual→dirty-flag→densify→optimise arbitration; CPU + gsplat renderers; localized optimizer |
| `cargen/reid` | Duplicate-detection embeddings — histogram (real) / DINOv2 |
| `cargen/export` | Standard 3DGS `.ply` (degree 0 or 3), `.splat`, provenance overlay |
| `server/` | FastAPI ingestion, per-vehicle queue, merge + events, named storage |
| `clients/capture` | Phone capture page (LAN) |
| `viewer/` | Browser splat viewer — studio floor, provenance overlay, turntable |
| `demo/` | Deterministic synthetic sedan fusion demo |

The test suite runs entirely on the CPU/stub paths (no GPU, no model weights) so it stays
fast and portable. Real backends are exercised by manual acceptance runs.

## Gotchas worth knowing before you touch the relevant code

### Image-to-3D output is view-aligned, not canonical
The same car photographed at azimuth 0.0 vs 1.2 rad comes back with its length axis pointing
in different directions, tilted by the camera's elevation. `canonicalize_orientation` (PCA)
recovers the object's own axes. Without it the car sits tilted in the viewer and two scans
of one vehicle cannot be merged. PCA still can't distinguish front from back, so a
reconstruction may face backwards and merges may be 180° out — the render-based re-ID
verifier (see [ROADMAP](ROADMAP.md)) is what fixes that.

### Densification never invents depth in open space
New splats may only grow within a few pixels of geometry that already exists, inheriting its
depth (`densify_reach`). A monocular pixel with nothing behind it has no recoverable depth;
removing this guard reintroduces a flat slab of splats across the frame whenever the
segmentation mask is sloppy.

### The `stub` segmenter is a crude rectangle
With it, the prior's paint gets tinted by the background and the silhouette carries a halo of
densified splats. It exists to keep the CPU path installable. Any real evaluation wants
`CARGEN_SEGMENTER=rembg`.

### SH bands are zero on any prior
`GaussianCloud.sh_rest` (view-dependent appearance) exists and round-trips through `.ply`,
but a single-image prior leaves it zero — one photo has no evidence of how a surface looks
from elsewhere. Only the multi-view consolidation pass fills these. Zero SH = matte by
construction. When writing `.ply`, the exporter omits the 45 `f_rest_*` floats entirely for
matte clouds (keeps prior files ~3.7× smaller); the `f_rest` layout is **channel-major**
(all 15 red coefficients, then green, then blue) — get it wrong and viewers still load the
file but colours smear as you orbit.

### texture_baker built with `USE_CUDA=0` is CPU-only
SF3D hands it CUDA tensors, so `_install_cpu_baker_bridge` moves them across per call. It
self-disables if you ever rebuild texture_baker with the CUDA Toolkit.

### Viewer GPU sort is forced off
`gpuAcceleratedSort: false` in `viewer/main.js`. With it on, the splats render nothing —
silently, no error — on Intel Arc iGPUs via ANGLE/D3D11, and Chrome prefers the integrated
GPU over a discrete one by default. Don't re-enable without a per-device capability probe.

### Restarting the server: kill by port, not by name
On Windows a stale server process can keep holding the port while a "restarted" one silently
fails to bind (`Errno 10048`) and serves old code — producing output that looks *suspiciously
identical* rather than wrong. If a fix appears to do nothing, confirm which process is
actually serving:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
```

and `GET /health` reports the backend classes actually loaded.

### CUDA Toolkit won't install
The toolkit installer bundles a GPU driver and aborts if your installed driver is *newer*
than the bundled one. Install components only, skipping the driver, from an **elevated**
shell (the cached installer path is from winget):

```powershell
& "$env:LOCALAPPDATA\Temp\WinGet\Nvidia.CUDA.12.1\cuda_12.1.0_531.14_windows.exe" `
  -s nvcc_12.1 cudart_12.1 cuda_profiler_api_12.1 thrust_12.1 `
  cublas_dev_12.1 curand_dev_12.1 visual_studio_integration_12.1
```

Version must match the PyTorch build (cu121 → 12.1); PyTorch rejects a CUDA *major* mismatch
when building extensions.

### Demo `observed%` ceiling
The procedural demo sedan is built from overlapping boxes, so ~34% of its sampled surface is
interior faces no camera can reach. `demo/synthetic.py` visibility-culls them so the number
converges sensibly; real priors emit outer shells and don't have this problem.

## Off-network access
The server binds the LAN only and is not reachable from the internet. To scan away from
home, put Tailscale on the laptop and phone — same enclosed setup, no ports opened, no code
change.
