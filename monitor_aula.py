import cv2
import mediapipe as mp
import numpy as np
import time # Necesario para el temporizador de inactividad
fondo = cv2.imread("map.jpg")
fondo = cv2.resize(fondo, (400, 300))

# Inicializa MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializa el dibujante de MediaPipe
mp_drawing = mp.solutions.drawing_utils

# Captura de video desde la cámara (0 es la cámara por defecto)
cap = cv2.VideoCapture(0)

# --- Variables de control de tiempo ---
UMBRAL_ALERTA_SEGUNDOS = 5 # REDUCIDO PARA PRUEBAS (5 segundos para inactividad)
tiempo_inicio_inactivo = time.time()
ESTADO_PERSISTENTE = "INICIANDO" # Estado inicial, esperando detección
UMBRAL_MESA = 0.75 # Calibrar la altura 'y' de la muñeca para la zona de la mesa
                    # (ej. 0.75 significa que el 75% inferior de la pantalla es la mesa)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Procesar la imagen (BGR a RGB, no editable)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Realiza la detección
    results = holistic.process(image)

    # 2. Revertir para visualización (RGB a BGR, editable)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 3. Dibujar los puntos de referencia (Opcional, solo para desarrollo/debugging)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # --- Lógica de Detección de Escritura (Actualizada con zona de mesa) ---
    ESTADO_ACTUAL = "INACTIVO" # Estado detectado en este fotograma
    RIGHT_HAND = results.right_hand_landmarks

    if RIGHT_HAND:
        index_tip = np.array([RIGHT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                              RIGHT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y])
        thumb_tip = np.array([RIGHT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
                              RIGHT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].y])
        wrist_y = RIGHT_HAND.landmark[mp_holistic.HandLandmark.WRIST].y
        
        distance = np.linalg.norm(index_tip - thumb_tip)

        # Condición: agarre de lápiz Y muñeca en la zona de la mesa
        if (distance < 0.05) and (wrist_y > UMBRAL_MESA): # <--- Calibrar 0.05 y UMBRAL_MESA
            ESTADO_ACTUAL = "ESCRIBIENDO"
        else:
            ESTADO_ACTUAL = "NO_ESCRIBIENDO" # Renombrado para mayor claridad

    # --- Lógica de Persistencia y Temporizador ---
    tiempo_transcurrido = time.time() - tiempo_inicio_inactivo

    if ESTADO_ACTUAL == "ESCRIBIENDO":
        tiempo_inicio_inactivo = time.time() # Reinicia el temporizador
        ESTADO_PERSISTENTE = "ESCRIBIENDO"
    elif ESTADO_ACTUAL == "NO_ESCRIBIENDO":
        if tiempo_transcurrido >= UMBRAL_ALERTA_SEGUNDOS:
            ESTADO_PERSISTENTE = "ALERTA: INACTIVO PROLONGADO"
        # Si NO_ESCRIBIENDO pero aún no se cumple el umbral,
        # el ESTADO_PERSISTENTE se mantiene como estaba (ej. ESCRIBIENDO si fue lo último)
        # Esto previene el cambio instantáneo.

    # -------------------------------------------------------------------------
    #                     INTERFAZ SIMPLIFICADA DEL AULA
    # -------------------------------------------------------------------------

    # Crear un lienzo en blanco para la interfaz del aula
    # Puedes ajustar el tamaño (ej. 400x300)
    aula_map_width, aula_map_height = 400, 300
    aula_map = fondo.copy()
    #aula_map = np.zeros((aula_map_height, aula_map_width, 3), dtype=np.uint8) # Fondo negro

    # Definir la posición y tamaño de la "banca"
    banca_x1, banca_y1 = 50, 50
    banca_x2, banca_y2 = 350, 200 # Ejemplo, puedes ajustar

    # Definir el color de la banca
    banca_color = (0, 255, 0) # Verde por defecto
    texto_estado_banca = "ESTADO: ESCRIBIENDO"

    if ESTADO_PERSISTENTE == "ALERTA: INACTIVO PROLONGADO":
        banca_color = (0, 0, 255) # Rojo
        texto_estado_banca = f"ALERTA: Inactivo {int(tiempo_transcurrido)}s"
    elif ESTADO_PERSISTENTE == "NO_ESCRIBIENDO" and tiempo_transcurrido < UMBRAL_ALERTA_SEGUNDOS:
        banca_color = (0, 165, 255) # Naranja para una "pausa breve"
        texto_estado_banca = f"PAUSA: {int(tiempo_transcurrido)}s"
    elif ESTADO_PERSISTENTE == "INICIANDO":
        banca_color = (255, 255, 0) # Amarillo para "cargando"
        texto_estado_banca = "Cargando..."


    # Dibujar el rectángulo de la banca
    cv2.rectangle(aula_map, (banca_x1, banca_y1), (banca_x2, banca_y2), banca_color, -1) # -1 para rellenar
    cv2.rectangle(aula_map, (banca_x1, banca_y1), (banca_x2, banca_y2), (255, 255, 255), 2) # Borde blanco

    # Añadir texto dentro de la banca
    # Calculamos la posición para centrar el texto (aproximadamente)
    text_size = cv2.getTextSize(texto_estado_banca, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = banca_x1 + (banca_x2 - banca_x1 - text_size[0]) // 2
    text_y = banca_y1 + (banca_y2 - banca_y1 + text_size[1]) // 2
    cv2.putText(aula_map, texto_estado_banca, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Texto adicional en el mapa (ej. Nombre de la banca)
    cv2.putText(aula_map, "Banca 1 - TU PRUEBA", (10, aula_map_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


    # Mostrar la ventana de debugging (con los puntos de MediaPipe)
    cv2.imshow('Monitor de Postura IA (Debugging)', image)
    
    # Mostrar la ventana de la interfaz del aula
    cv2.imshow('Interfaz Aula (Profesor)', aula_map)


    # Salir con la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()