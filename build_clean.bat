@echo off
rem BUILD_CLEAN.BAT -- STANDARDIZED WU bude RVC CLEAN BUILD

rem Step 0: Delete ALL build artifacts
rmdir /s /q build\*.exe build\*.o build/s*.h build/spv 2>nul

rem Step 1: Regenerate SPIR-V headers from .comp
cd /c Users/eman5/wuburvc && \"tools\gen_spv.sh\" && cd build

rem Step 2: Rebuild CUDA object (ensure TMP/TEMP are set)
set TMP=C:\Users\eman5\wuburvc\build\tmp
set TEMP=%TMP%
CD /D %TMP%
call C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
nvcc -O3 -arch=sm_75 -Xcompiler /GS- -fmad=false -c src/wubu_rvc_cuda.cu -o build/wubu_rvc_cuda.o

rem Step 3: Link final executable
cd /c Users/eman5/wuburvc\build\ && link -o wubu_rvc_vk.exe build/wubu_rvc_cuda.o src/wubu_vk.c \"src/wubu_rvc_cli.c\" src/wubu_rvc_real.c \"tools/ab_vocals.sh\"
if errorlevel 1 exit /b 1

rem Step 4: Install to WuBuMedia\build\
mkdir -p \"C:\\Users\\eman5\\WuBuMedia\\build\"
copy wubu_rvc_vk.exe \"C:\\Users\\eman5\\WuBuMedia\\build\"

rem Step 5: Run benchmarks for validation
ecdex \"build\\bench_conv3.exe|192|370|512|7|3\" && ecdex \"build\\bench_conv3.exe|256|3700|256|11|3\"
echo Build completed successfully with fresh artifacts