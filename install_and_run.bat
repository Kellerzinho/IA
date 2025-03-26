@echo off
echo ===== ETHARA MONITOR - INSTALADOR E EXECUTOR =====
echo.

REM Verifica se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python nao encontrado. Por favor, instale Python 3.8 ou superior.
    echo Visite: https://www.python.org/downloads/
    pause
    exit /b
)

:MENU
cls
echo ===== ETHARA MONITOR - MENU PRINCIPAL =====
echo.
echo 1. Instalar dependencias
echo 2. Configurar sistema
echo 3. Construir executavel
echo 4. Executar programa (via Python)
echo 5. Executar programa (via Executavel, se disponivel)
echo 6. Sair
echo.
set /p opcao="Escolha uma opcao (1-6): "

if "%opcao%"=="1" goto INSTALAR
if "%opcao%"=="2" goto CONFIGURAR
if "%opcao%"=="3" goto CONSTRUIR
if "%opcao%"=="4" goto EXECUTAR_PYTHON
if "%opcao%"=="5" goto EXECUTAR_EXE
if "%opcao%"=="6" goto SAIR
goto MENU

:INSTALAR
cls
echo Instalando dependencias...
python install_dependencies.py
echo.
pause
goto MENU

:CONFIGURAR
cls
echo Iniciando configuracao do sistema...
python config_manager.py
echo.
pause
goto MENU

:CONSTRUIR
cls
echo Construindo executavel...
python build_executable.py
echo.
pause
goto MENU

:EXECUTAR_PYTHON
cls
echo Executando via Python...
python people.py
goto MENU

:EXECUTAR_EXE
cls
if exist "dist\ethara_monitor.exe" (
    echo Executando o programa...
    start "" "dist\ethara_monitor.exe"
) else (
    echo Executavel nao encontrado. Por favor, construa o executavel primeiro (opcao 3).
    pause
)
goto MENU

:SAIR
echo Saindo...
exit /b 