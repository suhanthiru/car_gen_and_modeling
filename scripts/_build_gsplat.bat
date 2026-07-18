@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
"D:\car_3d_model\car_gen_and_modeling\.venv\Scripts\python.exe" -m pip install ninja packaging setuptools wheel jaxtyping rich
"D:\car_3d_model\car_gen_and_modeling\.venv\Scripts\python.exe" -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
