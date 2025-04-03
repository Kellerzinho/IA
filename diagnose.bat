@echo off
title Diagnóstico do Modelo YOLO
cls
echo =======================================================
echo        DIAGNÓSTICO DO MODELO YOLO E CUDA
echo =======================================================
echo.

echo Verificando arquivos de modelo...
dir *.pt
echo.

echo Verificando se o Python e PyTorch estão funcionando...
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('Dispositivos GPU:', torch.cuda.device_count())"
echo.

echo Verificando se o módulo ultralytics está disponível...
python -c "try: import ultralytics; print('Ultralytics versão:', ultralytics.__version__); print('OK!'); except Exception as e: print('ERRO:', e)"
echo.

echo Criando script de teste para o modelo YOLO...
echo import sys > test_yolo.py
echo import os >> test_yolo.py
echo import torch >> test_yolo.py
echo print("Python Path:", sys.executable) >> test_yolo.py
echo print("Diretório atual:", os.getcwd()) >> test_yolo.py
echo print("Arquivos PT disponíveis:") >> test_yolo.py
echo for f in os.listdir("."): >> test_yolo.py
echo     if f.endswith(".pt"): print(f"  - {f}") >> test_yolo.py
echo. >> test_yolo.py
echo print("Verificando CUDA:") >> test_yolo.py
echo print("CUDA disponível:", torch.cuda.is_available()) >> test_yolo.py
echo if torch.cuda.is_available(): >> test_yolo.py
echo     print("  Device:", torch.cuda.get_device_name(0)) >> test_yolo.py
echo     print("  Memória:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB") >> test_yolo.py
echo. >> test_yolo.py
echo try: >> test_yolo.py
echo     from ultralytics import YOLO >> test_yolo.py
echo     print("Módulo YOLO importado com sucesso") >> test_yolo.py
echo     # Tenta carregar o modelo, primeiro tenta roboflow.pt, depois people.pt >> test_yolo.py
echo     model_path = "roboflow.pt" if os.path.exists("roboflow.pt") else "people.pt" if os.path.exists("people.pt") else None >> test_yolo.py
echo     if model_path: >> test_yolo.py
echo         print(f"Tentando carregar o modelo {model_path}...") >> test_yolo.py
echo         model = YOLO(model_path) >> test_yolo.py
echo         print(f"Modelo carregado com sucesso!") >> test_yolo.py
echo         print(f"Movendo modelo para GPU...") >> test_yolo.py
echo         model.to('cuda' if torch.cuda.is_available() else 'cpu') >> test_yolo.py
echo         print(f"Modelo pronto para uso!") >> test_yolo.py
echo     else: >> test_yolo.py
echo         print("ERRO: Nenhum arquivo de modelo (roboflow.pt ou people.pt) encontrado!") >> test_yolo.py
echo except Exception as e: >> test_yolo.py
echo     print("ERRO ao carregar modelo YOLO:") >> test_yolo.py
echo     print(e) >> test_yolo.py

echo Executando teste do modelo YOLO...
echo.
python test_yolo.py

echo.
echo Tentando executar people.py com argumentos de segurança...
python people.py --help 2>nul
if %errorlevel% neq 0 (
    echo O comando acima falhou, vamos tentar carregar apenas o modelo sem iniciar o sistema completo...
    echo.
    
    echo import os, torch, sys > test_load_only.py
    echo from ultralytics import YOLO >> test_load_only.py
    echo # Forçar uso de CUDA >> test_load_only.py
    echo os.environ["CUDA_VISIBLE_DEVICES"] = "0" >> test_load_only.py
    echo model_path = "roboflow.pt" if os.path.exists("roboflow.pt") else "people.pt" if os.path.exists("people.pt") else None >> test_load_only.py
    echo if not model_path: >> test_load_only.py
    echo     print("ERRO: Nenhum arquivo de modelo encontrado") >> test_load_only.py
    echo     sys.exit(1) >> test_load_only.py
    echo print(f"Carregando modelo {model_path}...") >> test_load_only.py
    echo model = YOLO(model_path) >> test_load_only.py
    echo print(f"Modelo carregado com sucesso!") >> test_load_only.py
    echo model.to('cuda' if torch.cuda.is_available() else 'cpu') >> test_load_only.py
    echo print(f"Tipo do modelo: {type(model)}") >> test_load_only.py
    echo print(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}") >> test_load_only.py
    
    python test_load_only.py
)

echo.
echo Diagnóstico concluído. Verifique as mensagens acima para identificar o problema.
echo.
echo Pressione qualquer tecla para sair...
pause > nul 