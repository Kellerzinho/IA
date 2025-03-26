@echo off
echo Configurando inicializacao automatica do Ethara Monitor no Windows...

rem Determina o diretório onde este batch file está localizado
set SCRIPT_DIR=%~dp0

rem Cria script de inicialização
echo @echo off > "%SCRIPT_DIR%\start_ethara.bat"
echo echo Iniciando Ethara Monitor automaticamente... >> "%SCRIPT_DIR%\start_ethara.bat"
echo. >> "%SCRIPT_DIR%\start_ethara.bat"
echo rem Determina o diretorio onde este batch file esta localizado >> "%SCRIPT_DIR%\start_ethara.bat"
echo set SCRIPT_DIR=%%~dp0 >> "%SCRIPT_DIR%\start_ethara.bat"
echo. >> "%SCRIPT_DIR%\start_ethara.bat"
echo rem Verifica se o executavel existe >> "%SCRIPT_DIR%\start_ethara.bat"
echo if exist "%%SCRIPT_DIR%%\dist\ethara_monitor.exe" ( >> "%SCRIPT_DIR%\start_ethara.bat"
echo     start "" "%%SCRIPT_DIR%%\dist\ethara_monitor.exe" >> "%SCRIPT_DIR%\start_ethara.bat"
echo ) else ( >> "%SCRIPT_DIR%\start_ethara.bat"
echo     rem Verifica se Python esta instalado >> "%SCRIPT_DIR%\start_ethara.bat"
echo     python --version ^>nul 2^>^&1 >> "%SCRIPT_DIR%\start_ethara.bat"
echo     if %%errorlevel%% neq 0 ( >> "%SCRIPT_DIR%\start_ethara.bat"
echo         exit /b >> "%SCRIPT_DIR%\start_ethara.bat"
echo     ) >> "%SCRIPT_DIR%\start_ethara.bat"
echo     start "" python "%%SCRIPT_DIR%%\people.py" >> "%SCRIPT_DIR%\start_ethara.bat"
echo ) >> "%SCRIPT_DIR%\start_ethara.bat"

rem Cria um atalho no diretório de inicialização do Windows
set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_PATH=%STARTUP_FOLDER%\EtharaMonitor.lnk

rem Criar script VBS para a criação do atalho
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = "%SHORTCUT_PATH%" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "%SCRIPT_DIR%start_ethara.bat" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "Ethara Monitor - Sistema de Monitoramento Inteligente" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"

rem Executar o script VBS
cscript /nologo "%TEMP%\CreateShortcut.vbs"
del "%TEMP%\CreateShortcut.vbs"

echo.
echo Configuracao concluida! O Ethara Monitor sera iniciado automaticamente
echo na proxima vez que voce ligar o computador.
echo.
echo Para testar agora, execute:
echo %SCRIPT_DIR%start_ethara.bat
echo.
pause 