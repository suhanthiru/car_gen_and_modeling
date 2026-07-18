@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set DISTUTILS_USE_SDK=1
set NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor
set PATH=D:\car_3d_model\car_gen_and_modeling\.venv\Scripts;%CUDA_HOME%\bin;%PATH%
cd /d D:\car_3d_model\car_gen_and_modeling
"D:\car_3d_model\car_gen_and_modeling\.venv\Scripts\python.exe" scripts\verify_gsplat.py > D:\car_3d_model\car_gen_and_modeling\scripts\_verify_gsplat_log.txt 2>&1
