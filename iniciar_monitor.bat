@echo off
title Monitor de Mesas
cls
echo =======================================================
echo    MONITOR DE MESAS - INICIANDO SISTEMA DE DETECCAO
echo =======================================================
echo.

:: Verifica se Python está instalado
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao foi encontrado no sistema.
    echo Instale o Python 3.8 ou superior e adicione-o ao PATH do sistema.
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)

:: Verifica se o modelo existe
if not exist people.pt (
    echo [ERRO] Modelo de deteccao (people.pt) nao encontrado!
    echo O sistema nao conseguira detectar pessoas e mesas sem o modelo.
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)

:: Verifica se o arquivo de configuração existe
if not exist config.json (
    echo [AVISO] Arquivo config.json nao encontrado.
    echo Criando arquivo de configuracao padrao...
    
    echo { > config.json
    echo     "camera": { >> config.json
    echo         "url": "rtsp://admin:senha@192.168.1.100:554/cam/realmonitor?channel=1^&subtype=0", >> config.json
    echo         "comment": "URL da camera RTSP. Modifique conforme necessario." >> config.json
    echo     }, >> config.json
    echo     "dashboard": { >> config.json
    echo         "url": "http://localhost:3310", >> config.json
    echo         "comment": "URL do dashboard para envio de notificacoes." >> config.json
    echo     }, >> config.json
    echo     "model": { >> config.json
    echo         "path": "roboflow.pt", >> config.json
    echo         "conf": 0.1, >> config.json
    echo         "mao_conf": 0.5, >> config.json
    echo         "pessoa_conf": 0.15, >> config.json
    echo         "iou": 0.4, >> config.json
    echo         "agnostic_nms": true >> config.json
    echo     }, >> config.json
    echo     "detection": { >> config.json
    echo         "show_people_boxes": true, >> config.json
    echo         "show_association_lines": true, >> config.json
    echo         "debug_mode": true >> config.json
    echo     }, >> config.json
    echo     "tracking_params": { >> config.json
    echo         "alpha": 0.4, >> config.json
    echo         "max_missed_time": 10, >> config.json
    echo         "confirm_time": 10, >> config.json
    echo         "state_change_delay": 5, >> config.json
    echo         "distance_threshold": 80, >> config.json
    echo         "mao_confirm_time": 2, >> config.json
    echo         "atendimento_timeout": 20, >> config.json
    echo         "mao_vertical_threshold": 250, >> config.json
    echo         "mao_horizontal_threshold": 150, >> config.json
    echo         "peso_horizontal": 2, >> config.json
    echo         "peso_vertical": 0.8, >> config.json
    echo         "pessoa_to_table_max_dist": 150 >> config.json
    echo     }, >> config.json
    echo     "table_capacities": { >> config.json
    echo         "0": 10, >> config.json
    echo         "1": 12, >> config.json
    echo         "2": 14, >> config.json
    echo         "3": 2, >> config.json
    echo         "4": 4, >> config.json
    echo         "5": 6, >> config.json
    echo         "6": 8 >> config.json
    echo     } >> config.json
    echo } >> config.json
    
    echo Arquivo de configuracao criado.
    echo.
    echo [IMPORTANTE] Edite o arquivo config.json para configurar a URL da camera.
    echo Pressione qualquer tecla para continuar...
    pause > nul
)

:: Verifica se as dependências estão instaladas
echo Verificando dependencias...

python -c "import torch" 2>nul
if %errorlevel% neq 0 (
    echo [AVISO] Algumas dependencias nao estao instaladas.
    echo Executando instalacao...
    python install_dependencies.py
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao instalar dependencias.
        echo.
        echo Pressione qualquer tecla para sair...
        pause > nul
        exit /b 1
    )
)

echo Dependencias verificadas com sucesso.
echo.

:: Executar script principal com CUDA_LAUNCH_BLOCKING para melhor depuração
echo Iniciando script principal...
echo.
echo Controles durante a execucao:
echo - Tecla Q: Encerra o programa
echo - Tecla D: Ativa/desativa modo debug
echo - Tecla P: Mostra/oculta caixas de pessoas
echo - Tecla L: Mostra/oculta linhas de associacao
echo.
echo [AVISO] Iniciando em modo de depuracao para diagnosticar problemas de deteccao de pessoas

set CUDA_LAUNCH_BLOCKING=1
python people.py

echo.
if %errorlevel% neq 0 (
    echo [ERRO] O programa encerrou com erro.
) else (
    echo Programa encerrado normalmente.
)

echo.
echo Pressione qualquer tecla para sair...
pause > nul 