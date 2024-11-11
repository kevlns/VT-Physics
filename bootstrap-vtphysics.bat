@echo off
setlocal enabledelayedexpansion

echo "============================= VT-Physics Init Git Submodules ============================="
git submodule init
git submodule update

REM 设置 vcpkg 的安装路径
set VCPKG_PATH=%~dp0\Simulator\Thirdparty\vcpkg

REM 设置依赖库文件路径
set DEPENDENCIES_FILE=dependencies.txt

REM 检查 vcpkg 可执行文件是否存在
if not exist "%VCPKG_PATH%\vcpkg.exe" (
    echo Note: vcpkg.exe not found at %VCPKG_PATH%. Init vcpkg...
    call "%VCPKG_PATH%\bootstrap-vcpkg.bat"
)

REM 检查依赖库文件是否存在
if not exist "%DEPENDENCIES_FILE%" (
    echo ERROR: Dependencies file not found: %DEPENDENCIES_FILE%
    exit /b 1
)

echo "============================= VT-Physics Vcpkg Downloading Dependencies ============================="
REM 遍历依赖库文件中的每一行并安装依赖库
for /f "usebackq tokens=*" %%i in ("%DEPENDENCIES_FILE%") do (
    set LIB_NAME=%%i
    REM 跳过空行
    if not "!LIB_NAME!"=="" (
        echo Installing !LIB_NAME! ...
        "%VCPKG_PATH%\vcpkg.exe" install !LIB_NAME!
        
        REM 检查安装是否成功
        if errorlevel 1 (
            echo ERROR: Failed to install !LIB_NAME!.
            exit /b 1
        ) else (
            echo Successfully installed !LIB_NAME!
        )
    )
)

echo All dependencies installed successfully.
exit /b 0
