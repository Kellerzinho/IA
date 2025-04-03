@echo off
title Monitor de Mesas - Diagnóstico Mínimo
cls
echo =======================================================
echo          TESTE MÍNIMO DO MONITOR DE MESAS
echo =======================================================
echo.

echo Teste 1: Verificando Python...
python --version 2>&1
echo.

echo Teste 2: Verificando módulos básicos...
echo import sys, os, torch, cv2 > test_modules.py
echo print("Teste de importações bem-sucedido!") >> test_modules.py
echo print("Python path:", sys.executable) >> test_modules.py
echo if torch.cuda.is_available(): >> test_modules.py
echo     print("CUDA disponível:", torch.cuda.get_device_name(0)) >> test_modules.py
echo else: >> test_modules.py
echo     print("CUDA não disponível") >> test_modules.py

python test_modules.py
echo.

echo Teste 3: Verificando acesso ao people.py...
dir people.py
echo.

echo Teste 4: Tentando executar o people.py... 
echo ESTE TESTE PODE FALHAR - procure por erros na saída
echo.

set CUDA_LAUNCH_BLOCKING=1 
python people.py

echo.
echo Pressione qualquer tecla para encerrar...
pause > nul 