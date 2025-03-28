@echo off
title Monitor de Mesas - Sistema de Detecção
color 0A

echo ====================================================================
echo                    MONITOR DE MESAS - INICIALIZACAO
echo ====================================================================
echo.
echo Sistema de monitoramento de ocupacao de mesas via RTSP/CUDA
echo.

REM Verifica se Python está instalado e configurado
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [ERRO] Python nao foi encontrado no sistema!
    echo Para instalar Python:
    echo 1. Visite https://www.python.org/downloads/
    echo 2. Baixe e instale a versao mais recente
    echo 3. Marque a opcao "Add Python to PATH" durante a instalacao
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)

REM Verifica se o arquivo de configuração existe
if not exist config.json (
    color 0E
    echo [AVISO] Arquivo config.json nao encontrado!
    echo.
    echo Criando arquivo de configuracao padrao...
    
    echo { > config.json
    echo   "camera": { >> config.json
    echo     "url": "rtsp://admin:senha@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0" >> config.json
    echo   }, >> config.json
    echo   "dashboard": { >> config.json
    echo     "url": "http://localhost:3300/api/monitor/update" >> config.json
    echo   }, >> config.json
    echo   "model": { >> config.json
    echo     "path": "roboflow.pt", >> config.json
    echo     "conf": 0.1, >> config.json
    echo     "mao_conf": 0.8, >> config.json
    echo     "pessoa_conf": 0.3, >> config.json
    echo     "iou": 0.4, >> config.json
    echo     "agnostic_nms": true >> config.json
    echo   }, >> config.json
    echo   "detection": { >> config.json
    echo     "show_people_boxes": true, >> config.json
    echo     "show_association_lines": true, >> config.json
    echo     "debug_mode": false >> config.json
    echo   }, >> config.json
    echo   "tracking_params": { >> config.json
    echo     "distance_threshold": 100, >> config.json
    echo     "alpha": 0.3, >> config.json
    echo     "confirm_time": 1.5, >> config.json
    echo     "max_missed_time": 5.0, >> config.json
    echo     "mao_confirm_time": 1.0, >> config.json
    echo     "pessoa_to_table_max_dist": 200, >> config.json
    echo     "peso_horizontal": 1.0, >> config.json
    echo     "peso_vertical": 0.8, >> config.json
    echo     "mao_horizontal_threshold": 150, >> config.json
    echo     "mao_vertical_threshold": 250, >> config.json
    echo     "atendimento_timeout": 30, >> config.json
    echo     "state_change_delay": 3 >> config.json
    echo   }, >> config.json
    echo   "table_capacities": { >> config.json
    echo     "0": 10, >> config.json
    echo     "1": 12, >> config.json
    echo     "2": 14, >> config.json
    echo     "3": 2, >> config.json
    echo     "4": 4, >> config.json
    echo     "5": 6, >> config.json
    echo     "6": 8 >> config.json
    echo   } >> config.json
    echo } >> config.json
    
    echo Config.json criado! Por favor, edite a URL da camera RTSP antes de continuar.
    echo.
    echo Deseja editar a configuracao agora? (S/N)
    choice /c SN /n
    if %ERRORLEVEL% EQU 1 (
        start notepad config.json
        echo.
        echo Apos salvar o arquivo, pressione qualquer tecla para continuar...
        pause > nul
    )
)

REM Verifica se o modelo YOLO existe
if not exist roboflow.pt (
    color 0E
    echo [AVISO] Modelo YOLO (roboflow.pt) nao encontrado!
    echo.
    echo Por favor, coloque o arquivo do modelo YOLO no mesmo diretorio
    echo deste script ou ajuste o caminho no arquivo config.json.
    echo.
    echo Pressione qualquer tecla para continuar assim mesmo...
    pause > nul
)

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