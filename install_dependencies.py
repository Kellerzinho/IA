import subprocess
import sys
import os
import platform

def install_dependencies():
    print("Instalando dependências do projeto Ethara Monitor...")
    
    # Verificar se CUDA está disponível
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠️ AVISO: CUDA não está disponível no sistema.")
            print("Este sistema requer GPU com suporte a CUDA para funcionar corretamente.")
            print("Por favor, verifique a instalação do CUDA e os drivers da GPU.")
            
            if platform.system() == "Windows":
                print("\nPara instalar CUDA no Windows:")
                print("1. Baixe e instale os drivers mais recentes da NVIDIA: https://www.nvidia.com/Download/index.aspx")
                print("2. Baixe e instale o CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            else:
                print("\nPara instalar CUDA no Linux:")
                print("1. Instale os drivers NVIDIA: sudo apt install nvidia-driver-{versão}")
                print("2. Instale o CUDA Toolkit: sudo apt install nvidia-cuda-toolkit")
            
            confirm = input("\nContinuar mesmo sem CUDA disponível? (s/n): ").lower().strip()
            if confirm != 's':
                return False
    except ImportError:
        print("PyTorch não está instalado. Será instalado a seguir.")
    
    # Lista de pacotes necessários - ajustada para resolver problemas de CUDA
    dependencies = [
        "torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118",  # PyTorch com CUDA 11.8
        "torchvision>=0.15.0 --extra-index-url https://download.pytorch.org/whl/cu118",  # Necessário para NMS
        "ultralytics==8.0.196",
        "numpy>=1.23.5",
        "opencv-python>=4.7.0.72",
        "requests>=2.28.2"
    ]
    
    # Criando arquivo requirements.txt
    with open("requirements.txt", "w") as f:
        for dep in dependencies:
            f.write(f"{dep}\n")
    
    print("Arquivo requirements.txt criado.")
    
    # Instalando dependências
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependências instaladas com sucesso!")
        
        # Verificação final de CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n✅ CUDA disponível: {torch.cuda.get_device_name(0)}")
                print(f"Versão PyTorch: {torch.__version__}")
                print(f"Versão CUDA: {torch.version.cuda}")
                return True
            else:
                print("\n⚠️ AVISO: PyTorch instalado, mas CUDA ainda não está disponível.")
                print("Pode ser necessário reiniciar o computador ou verificar a instalação do CUDA.")
                
                # Instrução adicional para verificação do CUDA
                print("\nPara verificar se o CUDA está instalado corretamente, execute:")
                if platform.system() == "Windows":
                    print("nvidia-smi")
                else:
                    print("nvidia-smi ou lspci | grep -i nvidia")
                
                # Sugestão para instalação manual se necessário
                print("\nSe precisar, você pode instalar manualmente com:")
                print("pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118")
                
                return False
        except ImportError:
            print("Erro ao verificar instalação do PyTorch.")
            return False
    except Exception as e:
        print(f"Erro ao instalar dependências: {str(e)}")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    if not success:
        print("\nAtenção: A instalação foi concluída, mas o suporte a CUDA pode não estar configurado corretamente.")
        print("O sistema requer GPU com CUDA para funcionar adequadamente.") 