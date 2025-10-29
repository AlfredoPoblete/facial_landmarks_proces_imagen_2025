"""
Detector de landmarks faciales usando OpenCV y dlib.
Versión alternativa más compatible.
"""
import cv2
import numpy as np
import dlib
from .config import LANDMARK_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar landmarks faciales usando OpenCV.
    """
    
    def __init__(self):
        """Inicializa el detector de OpenCV."""
        # Cargar el detector de rostros de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Intentar cargar el predictor de landmarks de dlib si está disponible
        try:
            # URL del predictor preentrenado
            import urllib.request
            import os
            
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                # Descargar el predictor
                url = "https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
                urllib.request.urlretrieve(url, predictor_path)
            
            self.predictor = dlib.shape_predictor(predictor_path)
            self.use_dlib = True
        except:
            print("No se pudo cargar dlib, usando solo OpenCV básico")
            self.predictor = None
            self.use_dlib = False
    
    def detect(self, image, visualization_style="points"):
        """
        Detecta landmarks faciales en la imagen.

        Args:
            image (numpy.ndarray): Imagen en formato BGR (OpenCV)
            visualization_style (str): Estilo de visualización

        Returns:
            tuple: (imagen_procesada, landmarks, info)
        """
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Crear copia para dibujar
        imagen_con_puntos = image.copy()
        
        info = {
            "rostros_detectados": 0,
            "total_landmarks": 0,
            "deteccion_exitosa": False
        }
        
        if len(faces) > 0:
            info["rostros_detectados"] = len(faces)
            
            # Tomar el primer rostro detectado
            (x, y, w, h) = faces[0]
            
            # Convertir a rectángulo dlib si está disponible dlib
            if self.use_dlib and self.predictor:
                rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = self.predictor(gray, rect)
                
                info["total_landmarks"] = 68
                info["deteccion_exitosa"] = True
                
                # Dibujar landmarks según el estilo
                if visualization_style == "points":
                    # Solo puntos
                    for n in range(68):
                        point = landmarks.part(n)
                        cv2.circle(
                            imagen_con_puntos,
                            (point.x, point.y),
                            LANDMARK_RADIUS,
                            LANDMARK_COLOR,
                            LANDMARK_THICKNESS
                        )
                
                elif visualization_style == "contours":
                    # Contornos principales
                    # Ojos
                    for n in range(36, 42):  # Ojo izquierdo
                        point = landmarks.part(n)
                        cv2.circle(imagen_con_puntos, (point.x, point.y), 2, (0, 255, 0), -1)
                    
                    for n in range(42, 48):  # Ojo derecho
                        point = landmarks.part(n)
                        cv2.circle(imagen_con_puntos, (point.x, point.y), 2, (0, 255, 0), -1)
                    
                    # Boca
                    for n in range(48, 68):  # Boca
                        point = landmarks.part(n)
                        cv2.circle(imagen_con_puntos, (point.x, point.y), 2, (0, 255, 0), -1)
                
                elif visualization_style == "mesh":
                    # Conectar puntos principales
                    # Contorno facial
                    for n in range(0, 17):
                        p1 = landmarks.part(n)
                        p2 = landmarks.part((n + 1) % 17)
                        cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    
                    # Cejas
                    for n in range(17, 22):
                        p1 = landmarks.part(n)
                        p2 = landmarks.part(n + 1)
                        cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    
                    for n in range(22, 27):
                        p1 = landmarks.part(n)
                        p2 = landmarks.part(n + 1)
                        cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    
                    # Nariz
                    for n in range(27, 31):
                        p1 = landmarks.part(n)
                        p2 = landmarks.part(n + 1)
                        cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    
                    # Boca externa
                    for n in range(48, 60):
                        p1 = landmarks.part(n)
                        p2 = landmarks.part(n + 1)
                        cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    p1 = landmarks.part(60)
                    p2 = landmarks.part(48)
                    cv2.line(imagen_con_puntos, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 255), 1)
                    
                    # Puntos
                    for n in range(68):
                        point = landmarks.part(n)
                        cv2.circle(imagen_con_puntos, (point.x, point.y), 1, (0, 255, 0), -1)
                
                elif visualization_style == "heatmap":
                    # Heatmap simplificado
                    heatmap = np.zeros_like(image)
                    for n in range(68):
                        point = landmarks.part(n)
                        cv2.circle(heatmap, (point.x, point.y), 10, (0, 0, 255), -1)
                    
                    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
                    imagen_con_puntos = cv2.addWeighted(imagen_con_puntos, 0.7, heatmap, 0.3, 0)
                
                # Crear objeto compatible con las funciones de utils
                class SimpleLandmarks:
                    def __init__(self, landmarks):
                        self.landmark = []
                        for i in range(68):
                            point = landmarks.part(i)
                            # Crear objeto similar a MediaPipe
                            class Landmark:
                                def __init__(self, x, y):
                                    self.x = x / image.shape[1]  # Normalizar
                                    self.y = y / image.shape[0]  # Normalizar
                            
                            self.landmark.append(Landmark(point.x, point.y))
                
                landmarks_obj = SimpleLandmarks(landmarks)
                
                return imagen_con_puntos, landmarks_obj, info
            
            else:
                # Solo mostrar el rectángulo del rostro si no hay dlib
                info["total_landmarks"] = 0
                info["deteccion_exitosa"] = True
                
                cv2.rectangle(imagen_con_puntos, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(imagen_con_puntos, "Rostro detectado", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Crear objeto de landmarks básico
                class SimpleLandmarks:
                    def __init__(self, face_rect):
                        self.landmark = []
                        # Crear algunos landmarks básicos basados en el rectángulo
                        for i in range(68):
                            class Landmark:
                                def __init__(self, x_norm, y_norm):
                                    self.x = x_norm
                                    self.y = y_norm
                            
                            # Distribución básica de landmarks en el rostro
                            row = i // 17
                            col = i % 17
                            x_norm = (face_rect[0] + (col/16) * face_rect[2]) / image.shape[1]
                            y_norm = (face_rect[1] + (row/3) * face_rect[3]) / image.shape[0]
                            self.landmark.append(Landmark(x_norm, y_norm))
                
                landmarks_obj = SimpleLandmarks((x, y, w, h))
                return imagen_con_puntos, landmarks_obj, info
        
        # No se detectó rostro
        return imagen_con_puntos, None, info
    
    def close(self):
        """Libera recursos del detector."""
        pass