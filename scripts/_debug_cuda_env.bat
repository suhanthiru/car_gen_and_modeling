@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
echo CUDA_HOME is: %CUDA_HOME%
echo CUDA_PATH is: %CUDA_PATH%
"D:\car_3d_model\car_gen_and_modeling\.venv\Scripts\python.exe" -c "import os; print('python sees CUDA_HOME:', repr(os.environ.get('CUDA_HOME'))); from torch.utils.cpp_extension import _find_cuda_home; print('_find_cuda_home():', _find_cuda_home())"
