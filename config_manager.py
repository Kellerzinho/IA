import json
import os
import sys

def create_default_config():
    """
    Cria um arquivo de configuração padrão se não existir
    """
    config = {
        "camera": {
            "url": "rtsp://rascal:R%40sc%401%23%23@127.0.0.1:46589/cam/realmonitor?channel=11&subtype=0",
            "comment": "URL da câmera RTSP. Modifique conforme necessário."
        },
        "dashboard": {
            "url": "http://localhost:5000/api/notify",
            "comment": "URL do dashboard para envio de notificações."
        },
        "model": {
            "path": "roboflow.pt",
            "conf": 0.1,
            "mao_conf": 0.8,
            "pessoa_conf": 0.3,
            "iou": 0.4,
            "agnostic_nms": True
        },
        "detection": {
            "show_people_boxes": True,
            "show_association_lines": True,
            "debug_mode": False
        },
        "tracking_params": {
            "alpha": 0.4,
            "max_missed_time": 10,
            "confirm_time": 10,
            "state_change_delay": 5,
            "distance_threshold": 80,
            "mao_confirm_time": 2,
            "atendimento_timeout": 20,
            "mao_vertical_threshold": 250,
            "mao_horizontal_threshold": 150,
            "peso_horizontal": 2,
            "peso_vertical": 0.8,
            "pessoa_to_table_max_dist": 100
        },
        "table_capacities": {
            "0": 10,
            "1": 12,
            "2": 14,
            "3": 2,
            "4": 4,
            "5": 6,
            "6": 8
        }
    }
    
    return config

def save_config(config, file_path='config.json'):
    """
    Salva a configuração em um arquivo JSON
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Configuração salva em {file_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar configuração: {e}")
        return False

def load_config(file_path='config.json'):
    """
    Carrega a configuração de um arquivo JSON
    """
    try:
        if not os.path.exists(file_path):
            print(f"Arquivo de configuração {file_path} não encontrado. Criando configuração padrão.")
            config = create_default_config()
            save_config(config, file_path)
            return config
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuração carregada de {file_path}")
        return config
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        config = create_default_config()
        print("Usando configuração padrão")
        return config

def update_camera_url(url, file_path='config.json'):
    """
    Atualiza a URL da câmera na configuração
    """
    config = load_config(file_path)
    config['camera']['url'] = url
    return save_config(config, file_path)

def update_dashboard_url(url, file_path='config.json'):
    """
    Atualiza a URL do dashboard na configuração
    """
    config = load_config(file_path)
    config['dashboard']['url'] = url
    return save_config(config, file_path)

def update_model_path(path, file_path='config.json'):
    """
    Atualiza o caminho do modelo na configuração
    """
    config = load_config(file_path)
    config['model']['path'] = path
    return save_config(config, file_path)

def interactive_config():
    """
    Interface interativa para configurar o sistema
    """
    config = load_config()
    
    print("\n===== CONFIGURAÇÃO DO ETHARA MONITOR =====\n")
    
    # Configuração da câmera
    print(f"[1] URL da câmera atual: {config['camera']['url']}")
    change = input("Deseja alterar a URL da câmera? (s/n): ").lower().strip()
    if change == 's':
        new_url = input("Nova URL da câmera: ").strip()
        config['camera']['url'] = new_url
    
    # Configuração do dashboard
    print(f"\n[2] URL do dashboard atual: {config['dashboard']['url']}")
    change = input("Deseja alterar a URL do dashboard? (s/n): ").lower().strip()
    if change == 's':
        new_url = input("Nova URL do dashboard: ").strip()
        config['dashboard']['url'] = new_url
    
    # Configuração do modelo
    print(f"\n[3] Caminho do modelo atual: {config['model']['path']}")
    change = input("Deseja alterar o caminho do modelo? (s/n): ").lower().strip()
    if change == 's':
        new_path = input("Novo caminho do modelo: ").strip()
        config['model']['path'] = new_path
    
    # Configuração dos parâmetros do modelo
    print("\n[4] Parâmetros de detecção:")
    print(f"  - Confiança geral: {config['model']['conf']}")
    print(f"  - Confiança para mãos: {config['model']['mao_conf']}")
    print(f"  - Confiança para pessoas: {config['model']['pessoa_conf']}")
    change = input("Deseja alterar os parâmetros de detecção? (s/n): ").lower().strip()
    if change == 's':
        try:
            config['model']['conf'] = float(input("Nova confiança geral (0.1-1.0): ").strip())
            config['model']['mao_conf'] = float(input("Nova confiança para mãos (0.1-1.0): ").strip())
            config['model']['pessoa_conf'] = float(input("Nova confiança para pessoas (0.1-1.0): ").strip())
        except ValueError:
            print("Valor inválido. Usando valores anteriores.")
    
    # Configuração de visualização
    print("\n[5] Opções de visualização:")
    print(f"  - Mostrar caixas de pessoas: {config['detection']['show_people_boxes']}")
    print(f"  - Mostrar linhas de associação: {config['detection']['show_association_lines']}")
    print(f"  - Modo debug: {config['detection']['debug_mode']}")
    change = input("Deseja alterar as opções de visualização? (s/n): ").lower().strip()
    if change == 's':
        resp = input("Mostrar caixas de pessoas (s/n): ").lower().strip()
        config['detection']['show_people_boxes'] = (resp == 's')
        
        resp = input("Mostrar linhas de associação (s/n): ").lower().strip()
        config['detection']['show_association_lines'] = (resp == 's')
        
        resp = input("Ativar modo debug (s/n): ").lower().strip()
        config['detection']['debug_mode'] = (resp == 's')
    
    # Salvar configuração
    save_config(config)
    print("\nConfiguração atualizada com sucesso!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            config = create_default_config()
            save_config(config)
        elif sys.argv[1] == "camera" and len(sys.argv) > 2:
            update_camera_url(sys.argv[2])
        elif sys.argv[1] == "dashboard" and len(sys.argv) > 2:
            update_dashboard_url(sys.argv[2])
        elif sys.argv[1] == "model" and len(sys.argv) > 2:
            update_model_path(sys.argv[2])
    else:
        interactive_config() 