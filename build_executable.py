import subprocess
import sys
import os
import shutil
import platform

def build_executable():
    print("Iniciando a construção do executável...")
    
    # Verificar se CUDA está disponível
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠️ ERRO: CUDA não está disponível no sistema.")
            print("Este sistema requer GPU com suporte a CUDA para funcionar corretamente.")
            print("Por favor, verifique a instalação do CUDA e os drivers da GPU antes de continuar.")
            return False
        else:
            print(f"\n✅ CUDA disponível: {torch.cuda.get_device_name(0)}")
            print(f"Versão PyTorch: {torch.__version__}")
            print(f"Versão CUDA: {torch.version.cuda}")
    except ImportError:
        print("PyTorch não está instalado. Por favor, execute install_dependencies.py primeiro.")
        return False
    
    # Instalando dependências necessárias
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Verificando se todas as dependências estão instaladas
    dependencies = [
        "torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118",  # PyTorch com CUDA 11.8
        "ultralytics==8.0.196",
        "numpy>=1.23.5",
        "opencv-python>=4.7.0.72",
        "requests>=2.28.2"
    ]
    
    for dep in dependencies:
        subprocess.run([sys.executable, "-m", "pip", "install", dep])
    
    # Criando o diretório de build se não existir
    if not os.path.exists("build"):
        os.makedirs("build")
    
    # Determina a configuração baseada no sistema operacional
    is_windows = platform.system() == "Windows"
    separator = ";" if is_windows else ":"
    console_flag = "--noconsole" if is_windows else "--console"
    
    # Nome do executável (com extensão .exe no Windows)
    exe_name = "ethara_monitor.exe" if is_windows else "ethara_monitor"
    
    # Adiciona variáveis de ambiente para o PyInstaller incluir no executável
    env_vars = {
        "OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;tcp",
        "CUDA_VISIBLE_DEVICES": "0"
    }
    
    env_vars_str = " ".join([f'--env {k}="{v}"' for k, v in env_vars.items()])
    
    # Construindo o executável
    print(f"Construindo o executável para {platform.system()} com suporte a CUDA...")
    
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",
        console_flag,
        f"--add-data=roboflow.pt{separator}.",
        "--name=ethara_monitor",
        "--hidden-import=torch",
        "--hidden-import=torch.cuda",
        "--hidden-import=ultralytics",
        f"--path={os.path.dirname(torch.__file__)}",
        f"{env_vars_str}",
        "people.py"
    ]
    
    # Convertendo para string adequada ao sistema
    pyinstaller_cmd_str = " ".join(pyinstaller_cmd)
    
    # Executando o comando
    result = subprocess.run(pyinstaller_cmd_str, shell=True)
    
    if result.returncode != 0:
        print("Erro ao construir o executável.")
        return False
    
    # Movendo arquivos necessários para a pasta dist
    if os.path.exists("dist"):
        # Copiando modelo para a pasta dist
        if os.path.exists("roboflow.pt"):
            shutil.copy("roboflow.pt", "dist/")
        
        print(f"Executável construído com sucesso como '{exe_name}'!")
        print("O executável está localizado na pasta 'dist'")
        return True
    else:
        print("Erro ao construir o executável.")
        return False

if __name__ == "__main__":
    success = build_executable()
    if not success:
        print("\nFalha na construção do executável. Verifique os erros acima.") 