@echo off
REM ====================================================================
REM  Start the cargen server with real backends (rembg + SF3D), CUDA env.
REM  Double-click this file, or run it from any terminal. Ctrl+C to stop.
REM ====================================================================
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

set CARGEN_SEGMENTER=rembg
set CARGEN_PRIOR_BACKEND=sf3d
set CARGEN_RENDERER=point

cd /d "%~dp0.."
".venv\Scripts\python.exe" -m server
pause
