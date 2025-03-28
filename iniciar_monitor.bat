@echo off
title Monitor de Mesas - Sistema de Detecção
color 0A

echo [92m====================================================================
echo                    MONITOR DE MESAS - INICIALIZAÇÃO
echo ====================================================================[0m
echo.
echo [96m  Sistema de monitoramento de ocupação de mesas via RTSP/CUDA[0m
echo.

REM Verifica se Python está instalado e configurado
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [91m[ERRO] Python não foi encontrado no sistema![0m
    echo [93mPara instalar Python:
    echo 1. Visite https://www.python.org/downloads/
    echo 2. Baixe e instale a versão mais recente
    echo 3. Marque a opção "Add Python to PATH" durante a instalação[0m
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)

REM Verifica se o arquivo de configuração existe
if not exist config.json (
    color 0E
    echo [93m[AVISO] Arquivo config.json não encontrado![0m
    echo.
    echo [96mCriando arquivo de configuração padrão...[0m
    
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
    
    echo [92mConfig.json criado![0m Por favor, edite a URL da câmera RTSP antes de continuar.
    echo.
    echo [97mDeseja editar a configuração agora? (S/N)[0m
    choice /c SN /n
    if %ERRORLEVEL% EQU 1 (
        start notepad config.json
        echo.
        echo [96mApós salvar o arquivo, pressione qualquer tecla para continuar...[0m
        pause > nul
    )
)

REM Verifica se o modelo YOLO existe
if not exist roboflow.pt (
    color 0E
    echo [93m[AVISO] Modelo YOLO (roboflow.pt) não encontrado![0m
    echo.
    echo [96mPor favor, coloque o arquivo do modelo YOLO no mesmo diretório
    echo deste script ou ajuste o caminho no arquivo config.json.[0m
    echo.
    echo Pressione qualquer tecla para continuar assim mesmo...
    pause > nul
)

REM Verifica CUDA antes de iniciar
echo.
echo [96mVerificando disponibilidade de CUDA...[0m
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
if %ERRORLEVEL% NEQ 0 (
    color 0E
    echo [93m[AVISO] Não foi possível verificar a disponibilidade do CUDA.
    echo O sistema pode não funcionar corretamente sem GPU.[0m
    echo.
    echo [97mDeseja tentar executar mesmo assim? (S/N)[0m
    choice /c SN /n
    if %ERRORLEVEL% NEQ 1 (
        exit /b 1
    )
) else (
    python -c "import torch; cuda=torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0) if cuda else \"Não disponível\"}'); exit(0 if cuda else 1)"
    if %ERRORLEVEL% NEQ 0 (
        color 0E
        echo.
        echo [93m[AVISO] CUDA não está disponível no sistema!
        echo O sistema requer GPU com suporte a CUDA para funcionar corretamente.[0m
        echo.
        echo [97mDeseja tentar executar mesmo assim? (S/N)[0m
        choice /c SN /n
        if %ERRORLEVEL% NEQ 1 (
            exit /b 1
        )
    )
)

echo.
echo [92m====================================================================
echo                    INICIANDO MONITOR DE MESAS
echo ====================================================================[0m
echo.
echo [96mPressione [97m'Q'[96m na janela do monitor para encerrar o programa
echo Pressione [97m'D'[96m para ativar/desativar o modo de depuração
echo Pressione [97m'P'[96m para mostrar/ocultar caixas de pessoas
echo Pressione [97m'L'[96m para mostrar/ocultar linhas de associação[0m
echo.
echo [93mIniciando...[0m
echo.

REM Define a variável de ambiente para forçar RTSP via TCP
set OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp

REM Inicia o programa
python people.py

echo.
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo [91m[ERRO] O programa encontrou um problema e foi encerrado.
    echo Verifique o arquivo table_tracker.log para mais detalhes.[0m
) else (
    echo [92mPrograma encerrado com sucesso.[0m
)

echo.
echo [97mPressione qualquer tecla para sair...[0m
pause > nul 