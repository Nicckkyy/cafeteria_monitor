# -*- coding: utf-8 -*-

"""
Analisador de Ocupação de Refeitório com YOLOv8

Este script utiliza o modelo de inteligência artificial YOLOv8 para detectar
pessoas e cadeiras em tempo real através de uma câmera. Ele conta o número
de cadeiras ocupadas e vagas, exibindo o resultado na tela.

Autor: Seu Nome (inspirado por Gemini)
Curso: Desenvolvimento de Sistemas
"""

import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURAÇÕES ---
# Carrega o modelo YOLOv8 pré-treinado. 'yolov8n.pt' é o menor e mais rápido.
# Para maior precisão (e menor velocidade), pode-se usar 'yolov8s.pt' ou 'yolov8m.pt'.
MODEL_PATH = 'yolov8n.pt'

# Índice da câmera. 0 geralmente é a webcam padrão.
# Se tiver múltiplas câmeras, pode ser 1, 2, etc.
# Para usar um arquivo de vídeo, coloque o caminho: 'caminho/para/video.mp4'
SOURCE_INDEX = 0

# Classe de 'pessoa' e 'cadeira' no modelo COCO, que o YOLOv8 usa.
PERSON_CLASS_ID = 0
CHAIR_CLASS_ID = 56

# Cores para as caixas de detecção (formato BGR)
COLOR_EMPTY_CHAIR = (0, 255, 0)   # Verde
COLOR_OCCUPIED_CHAIR = (0, 0, 255) # Vermelho
COLOR_PERSON = (255, 0, 0)         # Azul

# Limiar de confiança para considerar uma detecção válida (de 0 a 1)
CONFIDENCE_THRESHOLD = 0.5

# --- FUNÇÃO AUXILIAR ---
def check_overlap(person_box, chair_box):
    """
    Verifica se a caixa de uma pessoa se sobrepõe significativamente à de uma cadeira.
    Isso é um indicador de que a cadeira está ocupada.
    
    Args:
        person_box (list): Coordenadas da caixa da pessoa [x1, y1, x2, y2].
        chair_box (list): Coordenadas da caixa da cadeira [x1, y1, x2, y2].

    Returns:
        bool: True se houver sobreposição, False caso contrário.
    """
    # Calcula a área de interseção
    x1_inter = max(person_box[0], chair_box[0])
    y1_inter = max(person_box[1], chair_box[1])
    x2_inter = min(person_box[2], chair_box[2])
    y2_inter = min(person_box[3], chair_box[3])

    # Se a interseção não for válida (largura ou altura negativas), não há sobreposição
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return False

    return True

# --- FUNÇÃO PRINCIPAL ---
def main():
    """
    Função principal que executa a captura de vídeo e a análise de ocupação.
    """
    # Inicializa o modelo YOLO
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        print("Verifique se o arquivo 'yolov8n.pt' está no mesmo diretório ou se a biblioteca 'ultralytics' está instalada.")
        return

    # Inicia a captura de vídeo
    cap = cv2.VideoCapture(SOURCE_INDEX)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo (índice {SOURCE_INDEX}).")
        print("Verifique se a câmera está conectada e funcionando.")
        return

    print("Análise em tempo real iniciada. Pressione 'q' para sair.")

    # Loop principal para processar cada frame do vídeo
    while True:
        # Lê um frame da câmera
        success, frame = cap.read()
        if not success:
            print("Não foi possível ler o frame da câmera. Encerrando.")
            break

        # Executa a detecção de objetos no frame
        results = model(frame, stream=True, verbose=False)

        # Listas para armazenar as caixas de detecção (bounding boxes)
        person_boxes = []
        chair_boxes = []

        # Processa os resultados da detecção
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Pega as coordenadas da caixa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Pega a confiança e a classe
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                # Filtra detecções com baixa confiança
                if conf > CONFIDENCE_THRESHOLD:
                    if cls == PERSON_CLASS_ID:
                        person_boxes.append([x1, y1, x2, y2])
                    elif cls == CHAIR_CLASS_ID:
                        chair_boxes.append([x1, y1, x2, y2])

        # Contadores para as cadeiras
        occupied_count = 0
        empty_count = 0
        
        # Lista para marcar cadeiras já contabilizadas como ocupadas
        occupied_chair_indices = set()

        # Lógica para determinar se uma cadeira está ocupada
        for p_box in person_boxes:
            for i, c_box in enumerate(chair_boxes):
                # Se a cadeira já foi marcada como ocupada por outra pessoa, pula
                if i in occupied_chair_indices:
                    continue
                
                # Verifica se a pessoa está "sentada" na cadeira
                if check_overlap(p_box, c_box):
                    occupied_count += 1
                    occupied_chair_indices.add(i)
                    break # Uma pessoa só pode ocupar uma cadeira

        # As cadeiras não marcadas como ocupadas são consideradas vagas
        empty_count = len(chair_boxes) - len(occupied_chair_indices)
        total_chairs = len(chair_boxes)

        # --- DESENHA AS INFORMAÇÕES NO FRAME ---

        # Desenha as caixas das cadeiras
        for i, c_box in enumerate(chair_boxes):
            x1, y1, x2, y2 = c_box
            if i in occupied_chair_indices:
                # Cadeira ocupada
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OCCUPIED_CHAIR, 2)
                cv2.putText(frame, 'Ocupada', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OCCUPIED_CHAIR, 2)
            else:
                # Cadeira vaga
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_EMPTY_CHAIR, 2)
                cv2.putText(frame, 'Vaga', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EMPTY_CHAIR, 2)
        
        # Desenha as caixas das pessoas
        for p_box in person_boxes:
            x1, y1, x2, y2 = p_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            # cv2.putText(frame, 'Pessoa', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PERSON, 2)

        # Cria um painel de informações no topo da tela
        info_panel = np.zeros((80, frame.shape[1], 3), dtype="uint8")
        info_panel[:] = (40, 40, 40) # Cor de fundo do painel (cinza escuro)

        text_total = f"Total de Cadeiras: {total_chairs}"
        text_occupied = f"Ocupadas: {occupied_count}"
        text_empty = f"Vagas: {empty_count}"

        cv2.putText(info_panel, text_total, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_panel, text_occupied, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OCCUPIED_CHAIR, 2)
        cv2.putText(info_panel, text_empty, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_EMPTY_CHAIR, 2)
        
        # Concatena o painel de informações com o frame do vídeo
        frame_with_panel = cv2.vconcat([info_panel, frame])

        # Mostra o resultado
        cv2.imshow("Analisador de Refeitorio - IA", frame_with_panel)

        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos ao finalizar
    cap.release()
    cv2.destroyAllWindows()
    print("Análise encerrada.")

if __name__ == "__main__":
    main()
