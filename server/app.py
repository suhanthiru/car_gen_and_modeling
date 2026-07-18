"""FastAPI ingestion server — the hub every capture device talks to.

Binds to the LAN only: phones on the same Wi-Fi reach it, the public internet
does not. Devices only ever make outbound calls to this one API, which is what
lets the same clients later point at a Tailscale address or a cloud host without
changing anything.

Endpoints:
    POST /observations                 upload a photo/video with a car name
    GET  /vehicles                     list vehicles
    GET  /vehicles/{name}              one vehicle's stats + observation log
    GET  /vehicles/{name}/model        the exported splat file
    GET  /events                       what happened (incl. merge notices)
    GET  /merges/pending               duplicates awaiting approval
    POST /merges/{id}/approve|reject   resolve one
    GET/POST /settings/auto-merge      the testing toggle
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from cargen import backends as cargen_backends
from cargen.core.asset import VehicleAsset
from cargen.pipeline import Pipeline
from server.config import CONFIG, PROJECT_ROOT, Config
from server.events import Event, EventLog
from server.merge import MergeManager
from server.queue import VehicleQueue
from server.store import VehicleStore

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def decode_image(payload: bytes, max_width: int) -> np.ndarray:
    """Uploaded bytes → RGB uint8, downscaled to a size fusion can chew on."""
    import cv2

    array = np.frombuffer(payload, np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    if bgr.shape[1] > max_width:
        scale = max_width / bgr.shape[1]
        bgr = cv2.resize(
            bgr, (max_width, int(round(bgr.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_app(config: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    config = config or CONFIG
    store = VehicleStore(config)
    events = EventLog(config.storage_root / "events.jsonl")
    queue = VehicleQueue()
    merges = MergeManager(store, events, config)
    app = FastAPI(title="cargen", version="0.1.0")

    # Built lazily: constructing the pipeline may load ML backends, which must
    # not happen at import time (tests, --help, etc.).
    state: dict = {"pipeline": pipeline}

    def get_pipeline() -> Pipeline:
        if state["pipeline"] is None:
            state["pipeline"] = Pipeline(prior_points=config.prior_points)
        return state["pipeline"]

    def resolve_or_404(key: str) -> str:
        folder = store.resolve(key)
        if folder is None:
            raise HTTPException(404, f"no vehicle named {key!r}")
        return folder

    # -- ingest --------------------------------------------------------------

    @app.post("/observations")
    async def post_observation(
        file: UploadFile = File(...),
        name: str = Form(...),
        device: str = Form("phone"),
    ):
        """Upload one capture of a vehicle. `name` is required and names the folder."""
        if not name.strip():
            raise HTTPException(400, "a vehicle name is required")
        payload = await file.read()
        if len(payload) > config.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"upload exceeds {config.max_upload_mb} MB")

        suffix = Path(file.filename or "capture").suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise HTTPException(
                400, f"unsupported file type {suffix!r}; "
                f"expected {sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES)}"
            )

        folder = store.resolve(name) or store.create_folder(name)
        is_new = not store.exists(folder)

        def work() -> dict:
            asset = store.load(folder) if store.exists(folder) else VehicleAsset(name=name)
            saved = store.save_upload(
                folder, f"{int(time.time())}-{file.filename or 'capture'}", payload
            )
            pipe = get_pipeline()
            if suffix in VIDEO_SUFFIXES:
                result = _ingest_video(pipe, asset, saved, device)
            else:
                image = decode_image(payload, config.max_image_width)
                result = pipe.ingest_photo(asset, image, device=device)
            exports = store.save(folder, result.asset)
            # the backend's own textured mesh, for comparison against our splats
            # and as the lightweight .glb asset
            if pipe.export_raw_prior(store.export_path(folder, "model.glb")):
                exports["glb"] = str(store.export_path(folder, "model.glb"))
            return {"summary": result.summary(), "exports": exports}

        try:
            outcome = queue.run_sync(folder, work)
        except Exception as exc:
            events.append(
                Event(kind="error", vehicle=folder, message=f"ingest failed: {exc}")
            )
            raise HTTPException(500, f"ingest failed: {exc}") from exc

        summary = outcome["summary"]
        events.append(
            Event(
                kind="observation",
                vehicle=folder,
                message=(
                    f"{'created' if is_new else 'updated'} '{folder}' from "
                    f"{file.filename} ({summary['frames_fused']} frames fused, "
                    f"{summary['observed_fraction']*100:.1f}% observed)"
                ),
                data=summary,
            )
        )
        merge_events = merges.scan(folder)
        return {
            "vehicle": store.summary(folder),
            "result": summary,
            "merges": [e.__dict__ for e in merge_events],
        }

    def _ingest_video(pipe: Pipeline, asset: VehicleAsset, path: Path, device: str):
        from cargen.video.frame_sampler import iter_video_frames

        return pipe.ingest_video(asset, iter_video_frames(str(path)), device=device)

    # -- read ----------------------------------------------------------------

    @app.get("/vehicles")
    def list_vehicles():
        return {"vehicles": [store.summary(f) for f in store.folders()]}

    @app.get("/vehicles/{key}")
    def get_vehicle(key: str):
        folder = resolve_or_404(key)
        manifest = store.manifest(folder) or {}
        return {
            **store.summary(folder),
            "observations_log": manifest.get("observations", []),
        }

    @app.get("/vehicles/{key}/model")
    def get_model(key: str, fmt: str = Query("splat", pattern="^(splat|ply|provenance)$")):
        folder = resolve_or_404(key)
        filename = {
            "splat": "model.splat",
            "ply": "model.ply",
            "provenance": "model_provenance.ply",
        }[fmt]
        path = store.export_path(folder, filename)
        if not path.exists():
            raise HTTPException(404, f"no {fmt} export for {folder!r} yet")
        return FileResponse(path, filename=f"{folder}-{filename}")

    @app.get("/events")
    def get_events(limit: int = 100):
        return {"events": events.all(limit)}

    # -- merges --------------------------------------------------------------

    @app.get("/merges/pending")
    def pending_merges():
        return {"pending": events.pending_merges()}

    @app.post("/merges/{event_id}/approve")
    def approve_merge(event_id: str):
        event = _pending_or_404(event_id)
        data = event["data"]
        primary, duplicate = data["primary"], data["duplicate"]
        if not (store.exists(primary) and store.exists(duplicate)):
            raise HTTPException(409, "one of these vehicles no longer exists")
        applied = queue.run_sync(
            primary,
            lambda: merges.apply(primary, duplicate, data["score"], pending_id=event_id),
            timeout=300,
        )
        return {"merged": applied.__dict__, "vehicle": store.summary(primary)}

    @app.post("/merges/{event_id}/reject")
    def reject_merge(event_id: str):
        event = _pending_or_404(event_id)
        data = event["data"]
        return {
            "rejected": merges.reject(event_id, data["primary"], data["duplicate"]).__dict__
        }

    def _pending_or_404(event_id: str) -> dict:
        event = events.find(event_id)
        if event is None or event["kind"] != "merge_pending":
            raise HTTPException(404, f"no pending merge {event_id!r}")
        if any(p["event_id"] == event_id for p in events.pending_merges()):
            return event
        raise HTTPException(409, f"merge {event_id!r} was already resolved")

    # -- settings ------------------------------------------------------------

    @app.get("/settings/auto-merge")
    def get_auto_merge():
        return {"auto_merge": config.auto_merge, "threshold": config.merge_threshold}

    @app.post("/settings/auto-merge")
    def set_auto_merge(enabled: bool = Form(...)):
        """The testing toggle: off = flag duplicates for approval, on = merge."""
        config.auto_merge = enabled
        return {"auto_merge": config.auto_merge}

    @app.get("/health")
    def health():
        # `backends` reports what is ACTUALLY loaded, not what config asked for.
        # Twice during development a stale server process kept the port and
        # silently served old code/stubs while everything looked fine; a running
        # server must be able to say what it is really running.
        pipe = state["pipeline"]
        backends = (
            {
                "segmenter": type(pipe.segmenter).__name__,
                "prior": type(pipe.prior_generator).__name__,
                "renderer": type(pipe.renderer).__name__,
                "embedder": type(pipe.embedder).__name__,
                "registrar": type(pipe.registrar).__name__,
                # None means refinement is the CPU colour-blend stand-in, not
                # the real optimizer — worth being able to see at a glance
                "optimizer": type(pipe.optimizer).__name__ if pipe.optimizer else None,
            }
            if pipe is not None
            else "not built yet (first upload builds it)"
        )
        return {
            "ok": True,
            "vehicles": len(store.folders()),
            "storage": str(config.storage_root),
            "auto_merge": config.auto_merge,
            # static probe, independent of pipe/backends being built yet —
            # answers "can this process even import gsplat" at a glance
            "gsplat_available": cargen_backends.gsplat_available(),
            "backends": backends,
            "queue": {
                "submitted": queue.stats.submitted,
                "completed": queue.stats.completed,
                "failed": queue.stats.failed,
                "active": queue.stats.active,
            },
        }

    # -- static clients ------------------------------------------------------

    capture_dir = PROJECT_ROOT / "clients" / "capture"
    viewer_dir = PROJECT_ROOT / "viewer"
    if viewer_dir.exists():
        app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    if capture_dir.exists():
        app.mount("/capture", StaticFiles(directory=capture_dir, html=True), name="capture")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return (capture_dir / "index.html").read_text(encoding="utf-8")

    app.state.store = store
    app.state.events = events
    app.state.queue = queue
    app.state.merges = merges
    app.state.config = config
    return app


app = create_app()
