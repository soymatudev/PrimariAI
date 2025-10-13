import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializa el dibujante de MediaPipe (para la visualización)
mp_drawing = mp.solutions.drawing_utils

# Captura de video desde la cámara (0 es la cámara por defecto)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Procesar la imagen
    # Convierte BGR a RGB y haz que la imagen no sea editable para el procesamiento
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Realiza la detección
    results = holistic.process(image)

    # 2. Revertir para visualización
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 3. Dibujar los puntos de referencia (Opcional, solo para desarrollo/debugging)
    # Dibujar manos
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Dibujar pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # --- Función de Detección de Escritura ---
    ESTADO = "INACTIVO" # Variable de estado global o dentro del scope
    RIGHT_HAND = results.right_hand_landmarks
    LEFT_HAND = results.left_hand_landmarks

    if RIGHT_HAND:
        # 1. Extrae las coordenadas de los puntos clave (normalizadas 0-1)
        # Ejemplo: Punta del dedo índice (Landmark 8) y Punta del pulgar (Landmark 4)
        index_tip = np.array([RIGHT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                            RIGHT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y])
        thumb_tip = np.array([RIGHT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
                            RIGHT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].y])

        # 2. Calcular la distancia euclidiana entre la punta del índice y el pulgar
        distance = np.linalg.norm(index_tip - thumb_tip)

        # 3. Lógica de Agarre de Escritura (Calibración Inicial)
        # Si la distancia es pequeña (dedos juntos, como sosteniendo un lápiz)
        if distance < 0.05: # <-- ESTE VALOR DEBE CALIBRARSE EN EL AULA
            ESTADO = "ESCRIBIENDO"
        else:
            ESTADO = "MANO DERECHA ABIERTA/INACTIVA"

    
    elif LEFT_HAND:
        # 1. Extrae las coordenadas de los puntos clave (normalizadas 0-1)
        index_tip = np.array([LEFT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                            LEFT_HAND.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y])
        thumb_tip = np.array([LEFT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].x,
                            LEFT_HAND.landmark[mp_holistic.HandLandmark.THUMB_TIP].y])

        # 2. Calcular la distancia euclidiana entre la punta del índice y el pulgar
        distance = np.linalg.norm(index_tip - thumb_tip)

        # 3. Lógica de Agarre de Escritura (Calibración Inicial)
        if distance < 0.05:
            ESTADO = "ESCRIBIENDO"
        else:
            ESTADO = "MANO IZQUIERDA ABIERTA/INACTIVA"
    

    # Muestra el estado en la pantalla
    cv2.putText(image, f"ESTADO: {ESTADO}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if ESTADO == "ESCRIBIENDO" else (0, 0, 255), 2, cv2.LINE_AA)

    # MOSTRAR LA IMAGEN FINAL (CON EL TEXTO DIBUJADO) ✅
    cv2.imshow('Monitor de Postura IA', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()