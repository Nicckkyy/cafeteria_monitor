# -*- coding: utf-8 -*-

"""
Analisador de Ocupação de Refeitório com YOLOv8 (v3.0)

Este script utiliza o modelo de inteligência artificial YOLOv8 para detectar
pessoas e cadeiras em tempo real. A v3.0 implementa uma lógica robusta que
infere a presença de cadeiras ocupadas mesmo quando estão ocultas (ocluídas)
por uma pessoa, baseando-se na proporção da caixa de detecção da pessoa.

Autor: Seu Nome (inspirado por Gemini)
Curso: Desenvolvimento de Sistemas
"""

import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURAÇÕES ---
# Carrega o modelo YOLOv8 pré-treinado. 'yolov8n.pt' é o menor e mais rápido.
MODEL_PATH = 'yolov8n.pt'

# Índice da câmera. 0 é a webcam padrão, 1 para a segunda, etc.
# Também pode ser o caminho de um arquivo de vídeo: 'caminho/video.mp4'
# ou um stream RTSP de uma câmera IP.
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

# Limiar de "Intersection over Union" (IoU) para considerar que uma pessoa está sentada.
SITTING_IOU_THRESHOLD = 0.2

# (NOVO) Limiar da proporção (altura/largura) da caixa da pessoa para inferir que ela está sentada.
# Pessoas em pé geralmente têm uma proporção > 1.8. Pessoas sentadas, < 1.7.
# Este valor pode ser ajustado para se adequar melhor ao ângulo da sua câmera.
SITTING_ASPECT_RATIO_THRESHOLD = 1.7


# --- FUNÇÃO AUXILIAR MELHORADA ---
def is_person_sitting(person_box, chair_box):
    """
    Verifica se uma pessoa está sentada em uma cadeira usando uma lógica mais robusta.

    Condições para ser verdadeiro:
    1. A sobreposição (IoU) entre a pessoa e a cadeira deve ser maior que SITTING_IOU_THRESHOLD.
    2. O centro horizontal da pessoa deve estar dentro dos limites horizontais da cadeira.

    Args:
        person_box (list): Coordenadas da caixa da pessoa [x1, y1, x2, y2].
        chair_box (list): Coordenadas da caixa da cadeira [x1, y1, x2, y2].

    Returns:
        bool: True se a pessoa parece estar sentada, False caso contrário.
    """
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    x1_inter = max(px1, cx1)
    y1_inter = max(py1, cy1)
    x2_inter = min(px2, cx2)
    y2_inter = min(py2, cy2)

    inter_width = x2_inter - x1_inter
    inter_height = y2_inter - y1_inter

    if inter_width <= 0 or inter_height <= 0:
        return False

    inter_area = inter_width * inter_height
    person_area = (px2 - px1) * (py2 - py1)
    chair_area = (cx2 - cx1) * (cy2 - cy1)
    union_area = person_area + chair_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    if iou < SITTING_IOU_THRESHOLD:
        return False

    person_center_x = (px1 + px2) / 2
    if not (cx1 < person_center_x < cx2):
        return False
        
    return True


# --- FUNÇÃO PRINCIPAL (LÓGICA DE CONTAGEM REFEITA) ---
def main():
    """
    Função principal que executa a captura de vídeo e a análise de ocupação.
    """
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        return

    cap = cv2.VideoCapture(SOURCE_INDEX)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo (índice {SOURCE_INDEX}).")
        return

    print("Análise em tempo real iniciada. Pressione 'q' para sair.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Não foi possível ler o frame da câmera. Encerrando.")
            break

        results = model(frame, stream=True, verbose=False)

        person_boxes = []
        chair_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])

                if conf > CONFIDENCE_THRESHOLD:
                    if cls == PERSON_CLASS_ID:
                        person_boxes.append([x1, y1, x2, y2])
                    elif cls == CHAIR_CLASS_ID:
                        chair_boxes.append([x1, y1, x2, y2])
        
        # --- LÓGICA DE CONTAGEM V3.0 ---
        person_assigned = [False] * len(person_boxes)
        chair_assigned = [False] * len(chair_boxes)
        
        # Passo 1: Associar pessoas a cadeiras VISÍVEIS
        for p_idx, p_box in enumerate(person_boxes):
            for c_idx, c_box in enumerate(chair_boxes):
                if chair_assigned[c_idx]: continue
                
                if is_person_sitting(p_box, c_box):
                    person_assigned[p_idx] = True
                    chair_assigned[c_idx] = True
                    break

        # Passo 2: Inferir cadeiras para pessoas não associadas (provavelmente sentadas em cadeiras ocultas)
        inferred_occupied_boxes = []
        for p_idx, p_box in enumerate(person_boxes):
            if not person_assigned[p_idx]:
                px1, py1, px2, py2 = p_box
                height = py2 - py1
                width = px2 - px1
                
                # Heurística: pessoas sentadas são mais "quadradas"
                if width > 0 and (height / float(width)) < SITTING_ASPECT_RATIO_THRESHOLD:
                    # Esta pessoa provavelmente está sentada em uma cadeira oculta
                    inferred_occupied_boxes.append(p_box)

        # Passo 3: Calcular os totais
        visible_occupied_indices = {i for i, assigned in enumerate(chair_assigned) if assigned}
        empty_indices = {i for i, assigned in enumerate(chair_assigned) if not assigned}

        occupied_count = len(visible_occupied_indices) + len(inferred_occupied_boxes)
        empty_count = len(empty_indices)
        total_chairs = occupied_count + empty_count
        
        # --- DESENHA AS INFORMAÇÕES NO FRAME ---
        # Desenha cadeiras VAGAS (verde)
        for i in empty_indices:
            x1, y1, x2, y2 = chair_boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_EMPTY_CHAIR, 2)
            cv2.putText(frame, 'Vaga', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EMPTY_CHAIR, 2)

        # Desenha cadeiras OCUPADAS VISÍVEIS (vermelho)
        for i in visible_occupied_indices:
            x1, y1, x2, y2 = chair_boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OCCUPIED_CHAIR, 2)
            cv2.putText(frame, 'Ocupada', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OCCUPIED_CHAIR, 2)
            
        # Desenha cadeiras OCUPADAS INFERIDAS (vermelho, na base da pessoa)
        for p_box in inferred_occupied_boxes:
            px1, _, px2, py2 = p_box
            # Desenha uma caixa na base da pessoa para representar a cadeira oculta
            cv2.rectangle(frame, (px1, py2 - 40), (px2, py2), COLOR_OCCUPIED_CHAIR, 2)
            cv2.putText(frame, 'Ocupada', (px1, py2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OCCUPIED_CHAIR, 2)

        # Desenha as pessoas (azul)
        for p_box in person_boxes:
            x1, y1, x2, y2 = p_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 1)

        # Painel de informações
        info_panel = np.zeros((80, frame.shape[1], 3), dtype="uint8")
        info_panel[:] = (40, 40, 40)
        text_total = f"Total de Cadeiras: {total_chairs}"
        text_occupied = f"Ocupadas: {occupied_count}"
        text_empty = f"Vagas: {empty_count}"
        cv2.putText(info_panel, text_total, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_panel, text_occupied, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OCCUPIED_CHAIR, 2)
        cv2.putText(info_panel, text_empty, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_EMPTY_CHAIR, 2)
        
        frame_with_panel = cv2.vconcat([info_panel, frame])
        cv2.imshow("Analisador de Refeitorio - IA (v3)", frame_with_panel)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Análise encerrada.")

if __name__ == "__main__":
    main()

