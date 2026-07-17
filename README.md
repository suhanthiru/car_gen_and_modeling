# cargen

**Turn one photo of a car into a 3D model — then let it learn the real thing over time.**

Take a single photo of a vehicle and cargen builds a complete, interactive 3D model of it:
the side you photographed *and* the front, roof, and far side you didn't. Those unseen parts
start as an educated guess. As you add more photos or a walk-around video — from any phone,
at any time — the model replaces its guesses with the real geometry and paint of *your*
specific car: the dent, the aftermarket wheels, the faded bumper.

It never forgets what it has already confirmed, and it never overwrites real detail with a
worse guess. That memory is the whole idea.

---

## The idea in one picture

```
   1 photo                a walk-around video               over time
      │                          │                              │
      ▼                          ▼                              ▼
┌───────────┐            ┌───────────────┐            ┌───────────────────┐
│ a whole   │            │ guesses get   │            │ a faithful digital │
│ car, most │  ───────▶  │ replaced by   │  ───────▶  │ twin of YOUR car,  │
│ of it a   │            │ the real car, │            │ confirmed panel    │
│ guess     │            │ panel by panel│            │ by panel           │
└───────────┘            └───────────────┘            └───────────────────┘
   ~15% real                 ~75% real                   approaching 1:1
```

Every point in the model is tagged **guessed** or **confirmed**, with a confidence score.
That tag is what lets new evidence overwrite guesses cheaply while protecting the detail
you've already captured — a blurry frame can't vandalise a clean one, and a part nobody has
photographed yet simply stays a guess until someone does.

The viewer has a toggle that paints this directly: **red = still a guess, green = confirmed
from a real photo.** You can watch your car turn green as you scan it.

---

## Why it's built this way

- **It's one persistent model per car, not a one-shot scan.** Photograph the same vehicle
  next week, next month, from a different phone — it fuses into the same model.
- **Video is first-class.** A slow walk-around confirms most of a car in one pass, because
  consecutive frames overlap and are easy to track.
- **Multiple cameras can contribute to one car.** Two people's photos of the same vehicle
  can merge into a single model — the system recognises they're the same car and combines
  the evidence. (Useful well beyond hobby scanning: a vehicle flagged from one photo, then
  confirmed and completed by street cameras that spot it, is the same mechanism.)
- **It runs on your own machine.** The server lives on your laptop and is reachable only by
  devices on your Wi-Fi — nothing is exposed to the internet. Your captures stay yours.

Under the hood it uses **3D Gaussian Splatting** (the same technique behind the glossy,
photoreal 3D captures you may have seen), a generative image-to-3D model for the initial
guess, and a fusion engine that arbitrates between guesses and evidence.

---

## Try it

You'll need Python 3.11. The base install has **no heavy ML dependencies** — it runs a
synthetic demo and the full server with a stand-in prior, so you can see the whole pipeline
work before installing anything large.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m pytest          # the test suite
.\.venv\Scripts\python.exe demo\run_demo.py   # watch a synthetic car go from guess to confirmed
.\.venv\Scripts\python.exe -m server          # start the server; open the printed URL on your phone
```

The server prints a link like `http://192.168.1.42:8000/`. Open it on a phone that's on the
same Wi-Fi, name your car, and take a photo. Then view the result at
`http://192.168.1.42:8000/viewer/?v=<your-car-name>`.

Each car is saved to a plainly-named folder on your machine:

```
data/vehicles/bobs-civic/
├── observations/          your original photos, kept so the model can improve later
└── exports/
    ├── model.ply          standard format — opens in SuperSplat, PlayCanvas, Blender
    ├── model.splat        what the web viewer streams
    └── model_provenance.ply   the red-vs-green "guessed vs confirmed" view
```

`model.ply` is a standard file — drag it into [SuperSplat](https://superspl.at/editor) in
your browser to inspect it outside the app.

### Getting a real 3D model (not the stand-in)

The base install uses a placeholder "prior" that produces a blocky test shape. For a real
reconstruction you install one image-to-3D backend. See
[`docs/SETUP.md`](docs/SETUP.md) for the full recipe — the short version is a CUDA build of
PyTorch, `rembg` for cutting the car out of the background, and one of these:

| Backend | Quality | Needs | Notes |
|---|---|---|---|
| **SF3D** | good | ~7 GB GPU, a C++ compiler | Runs fully offline. What this project is tested on. |
| **TRELLIS** | best | 16 GB GPU | Highest quality, offline. Fiddly to install on Windows. |
| **Tripo3D** | best | an API key | Cloud service — no GPU needed, but your photo leaves your machine. |

---

## How to get the best result

These aren't guesses — they're measured on this build:

- **Shoot a three-quarter view**, not a flat side-on shot. Standing at a front or rear
  corner so you see two sides at once roughly *doubles* how much of the car gets confirmed
  from a single photo.
- **A slow walk-around video beats any single photo.** This is how the model actually
  becomes your car rather than a generic one.
- **Matte or light-coloured paint reconstructs better than dark gloss.** A glossy car acts
  like a mirror, and a single photo can't tell "reflection of a tree" from "green paint".
- **Fill the frame and keep the car centred.** A tiny car in the corner of a wide photo
  confuses the model badly.
- **Even, diffuse light** (overcast, or open shade) beats harsh direct sun.

---

## Honest status

This is a working system with a deliberate boundary around what it can and can't do yet.

**What works today:** one photo → a complete, correctly-proportioned, textured 3D car;
segmentation that cleanly isolates the vehicle from a cluttered background; the guess-vs-
confirmed fusion logic; per-car persistent storage; the phone capture page and web viewer;
merging two scans of the same car.

**What's honest to expect:** a single photo gives you a *plausible* car, not a photograph.
The far side is inferred, glossy paint bakes in reflections, and the model can't yet tell a
car's front from its back (it may face the wrong way). None of this is a bug you can tune
away — **a single image simply doesn't contain the rest of the car.**

**What makes it photorealistic:** many real views of the actual car, optimised together —
i.e. a proper walk-around video plus a consolidation pass that's the next major piece of
work. With ~50+ good frames that reaches the quality of the 3D-capture demos you've seen
online. With only a handful of frames it won't. **Realism is bought with capture effort, not
with a cleverer guess** — which is exactly why the whole system is built to keep improving
one model over time instead of chasing perfection from a single shot.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what's built, what's next, and why.

---

## For developers

Architecture, the module map, install recipes, and the hard-won gotchas (image framing,
coordinate frames, the CUDA toolchain, viewer GPU quirks) live in
[`docs/SETUP.md`](docs/SETUP.md) and [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). The code is
organised as swappable modules — segmentation, prior generation, matching, pose, fusion,
export — each with a documented interface so a stub can be replaced by a real model without
touching the orchestration.

Configuration (which backends to load, storage location, whether to auto-merge duplicates,
network binding) is all environment-driven; `GET /health` reports exactly what's running.
