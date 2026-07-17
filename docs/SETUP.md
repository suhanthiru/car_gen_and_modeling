# Setup — real reconstruction

The base install (`pip install -e ".[dev]"`) runs the tests, the synthetic demo, and the
server with a stand-in prior — no heavy dependencies. This doc covers installing the real ML
backends for actual 3D reconstruction.

## Environment: use a venv, ideally on a roomy drive

The ML stack is large and version-sensitive. Keep it in a virtual environment, separate from
your system Python:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install "rembg[cpu]"
```

Two reasons this matters, both learned the hard way on the reference machine:

- **Disk.** A CUDA build of PyTorch is ~4.5 GB installed, and the Milestone B toolchain adds
  ~8 GB more. If your system drive is tight, create the venv on a drive with room to spare —
  everything lands inside `.venv`.
- **Version pins.** The 2024-era 3D ML ecosystem expects `numpy<2` and OpenCV 4.x. A modern
  global Python often has numpy 2.x / OpenCV 5, which these backends reject. The venv keeps
  those pins away from your other projects.

Always run through `.\.venv\Scripts\python.exe` (or activate the venv). The global
interpreter has no CUDA PyTorch and no `rembg`.

Verify what's actually loaded at any time:

```
GET /health  →  {"segmenter": "RembgSegmenter", "prior": "SF3DPriorGenerator", ...}
```

## Choosing a prior backend

Set `CARGEN_PRIOR_BACKEND`. Each adapter's module docstring in
`cargen/prior_generation/` carries its exact install recipe.

| Backend | Quality | VRAM | License | Notes |
|---|---|---|---|---|
| `sf3d` | good | ~7 GB | Stability Community | What this project is tested on. Offline. Needs a C++ compiler (below). |
| `trellis` | best | 16 GB rec. | MIT | Native Gaussians, no mesh round-trip. Painful Windows install. |
| `tripo` | best | none (cloud) | commercial API | Needs `TRIPO_API_KEY`; photo leaves your machine. Ships untested. |
| `custom` | — | — | — | Point `CARGEN_CUSTOM_PRIOR` at your own `module:callable`. |
| `stub` | placeholder | none | — | Default. A procedural test shape, no ML installs. |

Backends that emit Gaussians natively override `PriorGenerator.generate_splats`; the rest go
through the default mesh → surface-sample path.

### SF3D specifics (the tested path)

SF3D ships two C++ extensions imported at module scope, so it cannot run without a compiler:

- **Visual Studio Build Tools** with the C++ workload (~4 GB).
- Build the extensions with `USE_CUDA=0` (plain C++, no CUDA Toolkit needed) **inside the
  MSVC developer shell** (`vcvars64.bat`), with `--no-build-isolation`. The module docstring
  in `cargen/prior_generation/sf3d_impl.py` has the exact commands.
- SF3D's weights are gated on Hugging Face: accept the license on the model page, then
  `huggingface-cli login`. The weights auto-download (~2 GB) on first run.

**Image framing is not optional.** SF3D is trained on square, centred, subject-filling
images and its own demo crops to that before inference. Feeding a raw photo (car small and
off-centre in a wide frame) is a severe distribution shift and produces a blob. cargen calls
SF3D's own `resize_foreground` so framing matches by construction — but if you add a *new*
backend, replicate this first: it dwarfs every other quality knob.

## Milestone B — real refinement (photorealism)

For the consolidation pass that turns a walk-around video into a photoreal model:

- **CUDA Toolkit** matching the PyTorch build (cu121 → CUDA 12.1). On Windows the toolkit
  installer bundles a driver and will abort if yours is newer — install *components only*,
  skipping the driver, from an elevated shell. See
  [`DEVELOPMENT.md`](DEVELOPMENT.md#cuda-toolkit-wont-install) for the exact invocation.
- Then, in the MSVC dev shell: `pip install gsplat lightglue`. `gsplat` JIT-compiles CUDA
  kernels on first import — the most fragile install in the stack.

## Configuration

Everything is environment-driven (`CARGEN_*`) via `server/config.py`: which backends to
load, storage root, `auto_merge` (default **off** → duplicates are flagged for approval),
and the network bind. Nothing is hard-coded to one machine, so moving to a bigger box is:
install, copy the `data/` folder, set the env vars.
