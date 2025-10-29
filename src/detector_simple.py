"""
Detector de landmarks faciales usando solo OpenCV.
Versión simplificada sin dependencias externas.
"""
import cv2
import numpy as np
from .config import LANDMARK_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar rostros usando solo OpenCV.
    """
    
    def __init__(self):
        """Inicializa el detector de OpenCV."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def detect(self, image, visualization_style="points"):
        """
        Detecta rostros y características faciales en la imagen.

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
            
            # Detectar ojos, nariz y boca dentro del rostro
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = imagen_con_puntos[y:y+h, x:x+w]
            
            # Detectar ojos
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Detectar nariz
            nose = self.nose_cascade.detectMultiScale(roi_gray)
            
            info["total_landmarks"] = len(eyes) * 2 + len(nose) * 2  # Puntos aproximados
            info["deteccion_exitosa"] = True
            
            if visualization_style == "points":
                # Dibujar rectángulo del rostro
                cv2.rectangle(imagen_con_puntos, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Dibujar puntos de ojos
                for (ex, ey, ew, eh) in eyes:
                    center_x = x + ex + ew//2
                    center_y = y + ey + eh//2
                    cv2.circle(imagen_con_puntos, (center_x, center_y), 3, (0, 0, 255), -1)
                
                # Dibujar punto de nariz
                for (nx, ny, nw, nh) in nose:
                    center_x = x + nx + nw//2
                    center_y = y + ny + nh//2
                    cv2.circle(imagen_con_puntos, (center_x, center_y), 2, (255, 0, 0), -1)
                    
            elif visualization_style == "contours":
                # Solo contornos principales
                cv2.rectangle(imagen_con_puntos, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Círculos más grandes para ojos
                for (ex, ey, ew, eh) in eyes:
                    center_x = x + ex + ew//2
                    center_y = y + ey + eh//2
                    cv2.circle(imagen_con_puntos, (center_x, center_y), ew//2, (0, 0, 255), 2)
                
            elif visualization_style == "mesh":
                # Conectar puntos principales
                cv2.rectangle(imagen_con_puntos, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Conectar ojos si hay más de uno
                if len(eyes) >= 2:
                    eye1_center = (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
                    eye2_center = (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
                    cv2.line(imagen_con_puntos, eye1_center, eye2_center, (255, 0, 255), 2)
                
                # Dibujar puntos
                for (ex, ey, ew, eh) in eyes:
                    center_x = x + ex + ew//2
                    center_y = y + ey + eh//2
                    cv2.circle(imagen_con_puntos, (center_x, center_y), 3, (0, 0, 255), -1)
                    
            elif visualization_style == "heatmap":
                # Heatmap simplificado
                heatmap = np.zeros_like(image)
                
                # Crear gradiente de calor sobre el rostro
                face_region = heatmap[y:y+h, x:x+w]
                face_region[:] = (0, 0, 50)  # Base roja
                
                # Añadir intensidad en el centro
                cv2.circle(heatmap, (x + w//2, y + h//2), w//3, (0, 0, 100), -1)
                cv2.circle(heatmap, (x + w//2, y + h//2), w//4, (0, 0, 150), -1)
                
                # Superponer heatmap
                imagen_con_puntos = cv2.addWeighted(imagen_con_puntos, 0.7, heatmap, 0.3, 0)
            
            # Crear objeto de landmarks compatible
            class SimpleLandmarks:
                def __init__(self, faces, eyes, image_shape):
                    self.landmark = []
                    alto, ancho = image_shape[:2]
                    
                    if len(faces) > 0:
                        # Crear landmarks basados en la detección
                        face = faces[0]
                        fx, fy, fw, fh = face
                        
                        # Normalizar coordenadas del rostro
                        for i in range(68):  # Simular 68 landmarks
                            class Landmark:
                                def __init__(self, x_norm, y_norm):
                                    self.x = x_norm
                                    self.y = y_norm
                            
                            # Distribución aproximada de landmarks en el rostro
                            row = i // 17
                            col = i % 17
                            x_norm = (fx + (col/16) * fw) / ancho
                            y_norm = (fy + (row/3) * fh) / alto
                            self.landmark.append(Landmark(x_norm, y_norm))
            
            landmarks_obj = SimpleLandmarks(faces, eyes, image.shape)
            return imagen_con_puntos, landmarks_obj, info
        
        # No se detectó rostro
        cv2.putText(imagen_con_puntos, "No face detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return imagen_con_puntos, None, info
    
    def close(self):
        """Libera recursos del detector."""
        pass