import cv2
import mediapipe as mp
import numpy as np
import time

# Inicialización (sin cambios)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,  # Usar el modelo más ligero
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True
)

mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Variables de Control (sin cambios)
UMBRAL_ALERTA_SEGUNDOS = 5 
BANCAS = {
    "Banca Izquierda": {"X_MAX": 0.5, "activo_time": time.time(), "estado": "INICIANDO"},
    "Banca Derecha":   {"X_MIN": 0.5, "activo_time": time.time(), "estado": "INICIANDO"},
}
UMBRAL_MESA = 0.75
UMBRAL_AGARRE = 0.05

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- OPTIMIZACIÓN DE RENDIMIENTO ---
    ancho_reducido = 640
    alto_reducido = 480 
    frame = cv2.resize(frame, (ancho_reducido, alto_reducido)) # Redimensionar

    # 1. Procesar la imagen para MediaPipe (BGR a RGB, no editable)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Realiza la detección
    results = pose.process(image)

    # 2. Revertir para visualización (Editable y de vuelta a BGR)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 3. Dibujar los puntos de referencia (DEBUGGING)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # <--- Activado

    # --- Lógica de Detección y Asignación de Bancas (Tu código) ---

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Obtener coordenadas clave
        right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        
        # Aproximación simplificada de escritura
        is_writing_approx = wrist_y > UMBRAL_MESA and elbow_y > UMBRAL_MESA 
        
        # Asignación a Bancas
        banca_a_actualizar = None
        if right_shoulder_x < BANCAS["Banca Izquierda"]["X_MAX"]:
            banca_a_actualizar = "Banca Izquierda"
        elif right_shoulder_x >= BANCAS["Banca Derecha"]["X_MIN"]:
            banca_a_actualizar = "Banca Derecha"

        # Actualizar Estados
        if banca_a_actualizar:
            tiempo_actual = time.time()
            banca = BANCAS[banca_a_actualizar]
            tiempo_transcurrido = tiempo_actual - banca["activo_time"]

            if is_writing_approx:
                banca["activo_time"] = tiempo_actual
                banca["estado"] = "ESCRIBIENDO"
            else:
                if tiempo_transcurrido >= UMBRAL_ALERTA_SEGUNDOS:
                    banca["estado"] = "ALERTA: INACTIVO PROLONGADO"
                elif banca["estado"] == "ESCRIBIENDO":
                    banca["estado"] = "PAUSA CORTA"
            
            BANCAS[banca_a_actualizar] = banca 

    # -------------------------------------------------------------------------
    #                     INTERFAZ SIMPLIFICADA DEL AULA
    # -------------------------------------------------------------------------

    aula_map_width, aula_map_height = 800, 300 
    aula_map = np.zeros((aula_map_height, aula_map_width, 3), dtype=np.uint8)

    mapa_bancas_ui = {
        "Banca Izquierda": {"coords": (50, 50, 350, 250), "nombre": "ASIENTO 1"},
        "Banca Derecha":   {"coords": (450, 50, 750, 250), "nombre": "ASIENTO 2"},
    }

    for nombre_banca, datos_banca in BANCAS.items():
        x1, y1, x2, y2 = mapa_bancas_ui[nombre_banca]["coords"]
        estado = datos_banca["estado"]
        
        color = (0, 255, 0)
        texto = "ESCRIBIENDO"
        
        if estado == "ALERTA: INACTIVO PROLONGADO":
            color = (0, 0, 255)
            tiempo_transcurrido = time.time() - datos_banca["activo_time"]
            texto = f"ALERTA: {int(tiempo_transcurrido)}s"
        elif estado == "PAUSA CORTA":
            color = (0, 165, 255)
            tiempo_transcurrido = time.time() - datos_banca["activo_time"]
            texto = f"PAUSA: {int(tiempo_transcurrido)}s"
        elif estado == "INICIANDO":
            color = (255, 255, 0)
            texto = "Cargando..."

        cv2.rectangle(aula_map, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(aula_map, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        cv2.putText(aula_map, mapa_bancas_ui[nombre_banca]["nombre"], (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(aula_map, texto, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostrar ambas ventanas
    cv2.imshow('Monitor de Postura IA (Debugging)', image)
    cv2.imshow('Interfaz Aula (Profesor)', aula_map) # <--- Activado

    # Salir con la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()