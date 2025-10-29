# src/config.py
"""
Configuración del detector de landmarks faciales.
"""

# Configuración de visualización
LANDMARK_COLOR = (0, 255, 0)  # Verde en BGR
LANDMARK_RADIUS = 2
LANDMARK_THICKNESS = -1  # Relleno

# Cantidad de landmarks esperados
TOTAL_LANDMARKS = 68  # dlib usa 68 landmarks

# Parámetros del modelo OpenCV
MIN_FACE_SIZE = (100, 100)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5