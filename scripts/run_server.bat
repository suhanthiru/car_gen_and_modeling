@echo off
REM ====================================================================
REM  Start the cargen server with real backends: rembg + SF3D for the
REM  prior, gsplat (GPU) for rendering/fusion/re-ID. Needs the CUDA/MSVC
REM  env active because gsplat JIT-loads its cached CUDA extension on
REM  every import (see docs/DEVELOPMENT.md's gsplat section for why).
REM  Double-click this file, or run it from any terminal. Ctrl+C to stop.
REM ====================================================================
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
set PATH=%~dp0..\.venv\Scripts;%CUDA_HOME%\bin;%PATH%
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

set CARGEN_SEGMENTER=rembg
set CARGEN_PRIOR_BACKEND=sf3d
REM gsplat: ~36ms/render measured on a real 120k-splat cloud vs ~4000ms on
REM the CPU PointRenderer -- fast enough to make render-based re-ID usable.
set CARGEN_RENDERER=gsplat
REM Fallback registrar behind PnP: classical ORB features are too weak to
REM bridge a real viewpoint change on a low-texture car body (measured: 0
REM matches survive on 100% of a real 2nd-photo + 60-frame-video test).
REM Render-based re-ID scores a photo against the current model over a
REM pose sweep instead -- only viable now that it's GPU-backed.
set CARGEN_RENDER_REID=1

cd /d "%~dp0.."
".venv\Scripts\python.exe" -m server
pause
