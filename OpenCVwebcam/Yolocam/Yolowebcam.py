# Importação das Bibliotecas
import numpy as np
import cv2
import time
from csv import DictWriter

# Definir a captura da câmera
camera = cv2.VideoCapture(0)

# Variáveis para altura e largura do frame
h, w = None, None

# Carregar nomes de classes que o YOLO reconhece
with open('Yolocam/YoloNames.names') as f:
    labels = [line.strip() for line in f]

# Carregar modelo e configuração do YOLO
network = cv2.dnn.readNetFromDarknet('Yolocam/yolov3.cfg', 'Yolocam/yolov3.weights')

# Obter nomes das camadas e das camadas de saída do modelo
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Parâmetros de detecção
probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Abrir arquivo CSV para salvar detecções
with open('teste.csv', 'w') as arquivo:
    cabecalho = ['Detectado', 'Acuracia']
    escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)
    escritor_csv.writeheader()

    while True:
        # Leitura do frame da câmera
        ret, frame = camera.read()
        if not ret:
            break

        if w is None or h is None:
            h, w = frame.shape[:2]

        # Pré-processamento da imagem para o YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        network.setInput(blob)

        # Medir tempo de inferência
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()
        print('Tempo gasto atual {:.5f} segundos'.format(end - start))

        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Processar cada detecção
        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        # Aplicar NMS para remover caixas sobrepostas
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)

                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colour_box_current, 2)

                escritor_csv.writerow(
                    {"Detectado": labels[int(class_numbers[i])], "Acuracia": '{:.4f}'.format(confidences[i])})
                print(labels[int(class_numbers[i])] + " - " + '{:.4f}'.format(confidences[i]))

        cv2.imshow('YOLO v3 WebCamera', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
