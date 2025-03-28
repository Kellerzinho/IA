@echo off
title Monitor de Mesas - Sistema de Detecção
color 0A

echo ====================================================================
echo                    MONITOR DE MESAS - INICIALIZACAO
echo ====================================================================
echo.
echo Sistema de monitoramento de ocupacao de mesas via RTSP/CUDA
echo.

REM Define a variável de ambiente para forçar RTSP via TCP
set OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp

REM Inicia o programa
echo.
echo INICIANDO MONITOR DE MESAS
echo ====================================================================
echo.
echo Pressione 'Q' na janela do monitor para encerrar o programa
echo.
echo Iniciando...
echo.

python people.py

echo.
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [ERRO] O programa encontrou um problema e foi encerrado.
    echo Verifique o arquivo table_tracker.log para mais detalhes.
) else (
    echo Programa encerrado com sucesso.
)

echo.
echo Pressione qualquer tecla para sair...
pause > nul 