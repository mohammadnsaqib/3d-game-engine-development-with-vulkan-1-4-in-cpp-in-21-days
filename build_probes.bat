@echo off
REM Build one day's PROBES -- the small standalone programs whose output the chapter prints.
REM
REM   build_probes.bat 05           builds code\snapshots\day05\probes\*.cpp
REM
REM TWO KINDS OF PROBE, AND THEY ARE BUILT DIFFERENTLY.
REM   probes\*.cpp            must compile, link and run. Its stdout is the capture.
REM   probes\nocompile\*.cpp  must NOT compile. The compiler's refusal IS the capture, so a
REM                           successful build there is the failure. See that folder's README.
REM
REM THE SDK INCLUDE IS REACHED THROUGH A JUNCTION AT D:\b\sdk. Not for path length this time
REM but for the printed diagnostic: a refusal that names the header has to name it the same way
REM on every machine, or the captured text is about this computer rather than about the code.
setlocal
if "%~1"=="" (echo usage: build_probes.bat NN & exit /b 1)
set DAY=%~1
set HERE=%~dp0
set SRC=%HERE%snapshots\day%DAY%\probes
if not exist "%SRC%" (echo day%DAY% has no probes & exit /b 0)

set OBJ=D:\b\e552\day%DAY%\probes
if not exist "%OBJ%" mkdir "%OBJ%" 2>nul

set SDK=10.0.26100.0
set KIT=C:\Program Files (x86)\Windows Kits\10
if not exist "D:\b\sdk" mklink /J "D:\b\sdk" "%KIT%\Include\%SDK%\um" >nul

if not defined VCToolsInstallDir (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
)

pushd "%SRC%"
for %%f in (*.cpp) do (
  cl /nologo /std:c++17 /EHsc /W3 /O2 /MT /DUNICODE /D_UNICODE /DNOMINMAX ^
     /Fo"%OBJ%\\" /Fd"%OBJ%\vc.pdb" ^
     /I"D:\b\sdk" /I"%KIT%\Include\%SDK%\shared" /I"%KIT%\Include\%SDK%\ucrt" ^
     "%%f" ^
     /link /LIBPATH:"%KIT%\Lib\%SDK%\um\x64" /LIBPATH:"%KIT%\Lib\%SDK%\ucrt\x64" ^
     opengl32.lib user32.lib gdi32.lib ole32.lib windowscodecs.lib ^
     /OUT:"%SRC%\%%~nf.exe" || (echo PROBE BUILD FAILED %%f & popd & exit /b 1)
)
popd
echo built: %SRC%\*.exe
exit /b 0
