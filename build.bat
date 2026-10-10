@echo off
REM Build one day's Meridian and leave the .exe beside its own source.
REM
REM   build.bat 01          builds code\snapshots\day01
REM   build.bat 01 run      builds it and runs --all
REM
REM THE SHADERS ARE COMPILED FIRST. Vulkan consumes SPIR-V, not GLSL, so glslc runs over every
REM .vert, .frag and .comp in the day's src and embed_spirv.py writes them into src\spirv.h as
REM arrays of 32-bit words. The executable then carries its own shaders and the shipped folder
REM is one file.
REM
REM THE OBJECT FILES GO TO D:\b, NOT NEXT TO THE SOURCE. cl.exe is not long-path aware and this
REM book sits 157 characters down the shelf; a few more from an object file name and the
REM compiler reports C1083 "cannot open source file" for a file that is plainly there.
REM
REM THE ONLY LIBRARY THAT IS NOT WINDOWS' OWN IS vulkan-1.lib, and it is a stub: it resolves to
REM C:\Windows\System32\vulkan-1.dll, which the graphics driver installed. Nothing from the SDK
REM ships with the program. Day 21 is about that.

setlocal
if "%~1"=="" (echo usage: build.bat NN [run] & exit /b 1)
set DAY=%~1
set HERE=%~dp0
set SRC=%HERE%snapshots\day%DAY%
if not exist "%SRC%\src" (echo no such day: %SRC% & exit /b 1)

set OBJ=D:\b\e552\day%DAY%
if not exist "%OBJ%" mkdir "%OBJ%" 2>nul

set VKSDK=D:\VulkanSDK\1.4.357.0
set SDK=10.0.26100.0
set KIT=C:\Program Files (x86)\Windows Kits\10

REM ---- shaders to SPIR-V, and into a header -------------------------------------------------
python "%HERE%embed_spirv.py" %DAY%
if errorlevel 1 (echo SHADER BUILD FAILED & exit /b 1)

if not defined VCToolsInstallDir (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
)

pushd "%SRC%"
REM /DUNICODE makes the Win32 resource macros expand to their wide form. Without it
REM IDC_ARROW is an LPSTR and LoadCursorW will not take it.
REM /DNOMINMAX stops windows.h defining min and max as MACROS.
cl /nologo /std:c++17 /EHsc /W3 /O2 /MT /DUNICODE /D_UNICODE /DNOMINMAX ^
   /Fo"%OBJ%\\" /Fd"%OBJ%\vc.pdb" ^
   /I"%VKSDK%\Include" ^
   /I"%KIT%\Include\%SDK%\um" /I"%KIT%\Include\%SDK%\shared" /I"%KIT%\Include\%SDK%\ucrt" ^
   /I"%SRC%\src" ^
   src\*.cpp ^
   /link /LIBPATH:"%VKSDK%\Lib" ^
   /LIBPATH:"%KIT%\Lib\%SDK%\um\x64" /LIBPATH:"%KIT%\Lib\%SDK%\ucrt\x64" ^
   vulkan-1.lib user32.lib gdi32.lib ole32.lib windowscodecs.lib ^
   /OUT:"%SRC%\meridian.exe"
set RC=%ERRORLEVEL%
popd

if not "%RC%"=="0" (echo BUILD FAILED day%DAY% & exit /b %RC%)
echo built: %SRC%\meridian.exe

if /I "%~2"=="run" (
  pushd "%SRC%"
  set VK_LAYER_PATH=%VKSDK%\Bin
  meridian.exe --all
  set RC=%ERRORLEVEL%
  popd
)
exit /b %RC%
