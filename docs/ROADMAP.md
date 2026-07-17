# Roadmap

## Built and working

- **Single photo → complete 3D car.** Segmentation (rembg), image-to-3D prior (SF3D),
  surface-aligned textured Gaussian splats, PCA canonicalisation, correct proportions.
- **The fusion engine** — the guessed-vs-confirmed arbitration: render the current model,
  diff against a new photo, flag disagreeing regions, replace guesses, protect confirmed
  detail, densify where there's evidence but no geometry. Provenance / confidence / view-count
  per splat.
- **Video ingestion** — motion-magnitude frame sampling, frame-to-frame tracking.
- **Persistent per-vehicle storage**, named folders, standard `.ply` / `.splat` exports.
- **Server** — LAN ingestion API, per-vehicle job queue, duplicate detection + merge with
  human approval, event log.
- **Clients** — phone capture page, browser splat viewer with the red/green provenance
  overlay and a turntable.
- **View-dependent appearance is plumbed** — `sh_rest` (SH bands 1–3) exists and round-trips
  through `.ply`; it's zero on priors and only filled by consolidation (below).

## What stands between here and photorealism

A single photo cannot be photorealistic — the far side of the car isn't in the pixels, so
it's a statistical guess. Photorealism needs **many real views, jointly optimised.** Four
concrete pieces:

1. **View-dependent appearance** *(data structure done)* — SH bands make a highlight move as
   you orbit. Present but zero until there's multi-view evidence to fill it.
2. **Joint optimisation (`consolidate.py`, next)** — today's fusion loop is deliberately
   *localized* (one frame, changed splats only), which is right for incremental updates but
   never sees two views at once. Photorealism needs ~7k–30k iterations over all frames
   together, so multi-view consistency forces the geometry to be correct. This is the next
   major piece.
3. **Accurate poses (COLMAP)** — joint optimisation needs sub-pixel-accurate camera
   positions. The current pose estimator is good enough to *locate* one new photo, not to
   solve all cameras at once. A COLMAP structure-from-motion step over the walk-around frames
   supplies that.
4. **gsplat** — the CUDA rasterizer that makes the optimisation loop possible. Gated on the
   CUDA install.

**The novelty worth naming:** standard 3D Gaussian Splatting starts from COLMAP's sparse
points and can only reconstruct what was photographed. cargen seeds the optimiser with the
generative prior instead — a *complete* starting shape, including the parts no camera has
seen — and consolidation refines the seen parts toward photoreal while the unseen parts keep
the prior's plausible guess. Provenance rides along: seen splats become confirmed, unseen
ones stay guessed.

Expected outcome: **>30 dB PSNR on held-out views with 50+ good frames** (the quality of
published 3DGS demos). Under ~20 views, or with poor poses, worse than the prior. Realism is
bought with capture effort and compute.

## After that

- **Render-based re-ID verifier** — render the model from a candidate camera's angle and
  compare *same-angle* to the incoming photo (in deep-feature space, masking to confirmed
  regions). This one component pays off three times: it resolves the front/back ambiguity,
  supplies the rotation needed to align two scans for merging, and enables recognising the
  same car across wildly different camera angles without angle-invariant embeddings. It's the
  keystone for trustworthy auto-merge and the multi-camera flow.
- **Multi-device ingest** — Raspberry Pi / fixed-camera workers posting to the same API
  (outbound-only, so they work behind any router), with device-tier evidence weighting so a
  low-quality camera can confirm coverage without overwriting a clean capture.
- **Phone app** — the capture page is already installable; wrap it for the app stores.
  Native AR (ARCore/ARKit) would give phone captures free, accurate poses — the best evidence
  tier — and largely remove the need for the COLMAP step on handheld captures.
