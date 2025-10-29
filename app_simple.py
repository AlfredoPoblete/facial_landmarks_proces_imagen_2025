"""
Aplicación Streamlit simplificada para detección de landmarks faciales y análisis de expresiones.
Versión ultra-compatible sin OpenCV.
"""
import streamlit as st
from PIL import Image
import tempfile
import os
from src.detector_pil import FaceLandmarkDetector
from src.config import TOTAL_LANDMARKS


# Configuración de la página
st.set_page_config(
    page_title="Detector de Landmarks Faciales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar tema oscuro
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stRadio, .stFileUploader {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-container {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("Detector de Landmarks Faciales y Análisis de Expresiones")
st.markdown("""
Esta aplicación **simula** la detección de **68 puntos clave** en rostros humanos y calcula métricas de expresiones faciales.
Subí una imagen para ver la demostración de landmarks faciales.
""")

# Sidebar con información y controles
with st.sidebar:
    st.header("Configuración")

    # Selector de estilo de visualización
    estilo_visualizacion = st.selectbox(
        "Estilo de Visualización",
        ["points", "mesh", "contours", "heatmap"],
        format_func=lambda x: {
            "points": "Solo Puntos",
            "mesh": "Puntos + Malla",
            "contours": "Contornos Principales",
            "heatmap": "Heatmap de Densidad"
        }[x]
    )

    st.header("Información")
    st.markdown("""
    ### ¿Qué son los Landmarks?
    Son puntos de referencia (68 puntos) que simulan:
    - Ojos (párpados superiores e inferiores)
    - Nariz (puente y fosas nasales)
    - Boca (labios y comisuras)
    - Contorno facial completo

    ### Estilos de Visualización
    - **Solo Puntos**: Puntos individuales
    - **Puntos + Malla**: Conexiones entre puntos
    - **Contornos**: Solo contornos principales
    - **Heatmap**: Efecto de calor

    ### Métricas de Expresiones
    - **Apertura de boca**: Distancia entre labios
    - **Apertura de ojos**: Distancia entre párpados
    - **Inclinación de cabeza**: Ángulo de rotación

    ### Aplicaciones
    - Demostración de conceptos
    - Análisis de expresiones
    - Procesamiento de imágenes
    - Animación facial
    """)

    st.divider()
    st.caption("Desarrollado en el Laboratorio 2 - IFTS24")

# Funciones auxiliares
def resize_image(image, max_width=800):
    """Redimensiona la imagen manteniendo el aspect ratio."""
    ancho, alto = image.size
    if ancho > max_width:
        ratio = max_width / ancho
        nuevo_ancho = max_width
        nuevo_alto = int(alto * ratio)
        image = image.resize((nuevo_ancho, nuevo_alto))
    return image

def calcular_apertura_boca(landmarks, alto, ancho):
    """Calcula la apertura de la boca basada en landmarks simulados."""
    try:
        punto_superior = landmarks.landmark[51]  # Centro del labio superior
        punto_inferior = landmarks.landmark[57]  # Centro del labio inferior
        y1 = punto_superior.y * alto
        y2 = punto_inferior.y * alto
        return abs(y2 - y1)
    except:
        return 25.0  # Valor simulado

def calcular_apertura_ojos(landmarks, alto, ancho):
    """Calcula la apertura de los ojos basada en landmarks simulados."""
    try:
        # Ojo izquierdo
        superior_izq = landmarks.landmark[38]
        inferior_izq = landmarks.landmark[40]
        apertura_izq = abs(superior_izq.y - inferior_izq.y) * alto

        # Ojo derecho
        superior_der = landmarks.landmark[44]
        inferior_der = landmarks.landmark[46]
        apertura_der = abs(superior_der.y - inferior_der.y) * alto

        return apertura_izq, apertura_der
    except:
        return 15.0, 15.0  # Valores simulados

def calcular_inclinacion_cabeza(landmarks, alto, ancho):
    """Calcula la inclinación de la cabeza basada en landmarks simulados."""
    try:
        ojo_izq = landmarks.landmark[38]
        ojo_der = landmarks.landmark[44]
        x1 = ojo_izq.x * ancho
        y1 = ojo_izq.y * alto
        x2 = ojo_der.x * ancho
        y2 = ojo_der.y * alto
        delta_x = x2 - x1
        delta_y = y2 - y1
        import math
        angulo_radianes = math.atan2(delta_y, delta_x)
        angulo_grados = math.degrees(angulo_radianes)
        return angulo_grados
    except:
        return 0.0  # Valor simulado

# Uploader de imagen
uploaded_file = st.file_uploader(
    "Subí una imagen para analizar",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Procesamiento de imagen
    imagen_original = Image.open(uploaded_file)
    
    # Redimensionar si es muy grande
    imagen_cv2 = resize_image(imagen_original)
    
    # Columnas para mostrar antes/después
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(imagen_cv2, use_container_width=True)
    
    # Detectar landmarks
    with st.spinner("Procesando imagen..."):
        detector = FaceLandmarkDetector()
        imagen_procesada, landmarks, info = detector.detect(imagen_cv2, estilo_visualizacion)
        detector.close()
    
    with col2:
        st.subheader("Landmarks Simulados")
        st.image(imagen_procesada, use_container_width=True)
    
    # Mostrar información de detección
    st.divider()
    
    if info["deteccion_exitosa"]:
        st.success("Procesamiento exitoso (simulado)")
        
        # Calcular métricas de expresión
        alto, ancho = imagen_cv2.size
        apertura_boca = calcular_apertura_boca(landmarks, alto, ancho)
        apertura_ojos_izq, apertura_ojos_der = calcular_apertura_ojos(landmarks, alto, ancho)
        inclinacion_cabeza = calcular_inclinacion_cabeza(landmarks, alto, ancho)
        
        # Métricas
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Rostros detectados", info["rostros_detectados"])
        
        with metric_col2:
            st.metric("Landmarks detectados", f"{info['total_landmarks']}/{TOTAL_LANDMARKS}")
        
        with metric_col3:
            porcentaje = (info['total_landmarks'] / TOTAL_LANDMARKS) * 100
            st.metric("Precisión", f"{porcentaje:.1f}%")
        
        # Métricas de expresión
        st.subheader("Análisis de Expresiones (Simulado)")
        expr_col1, expr_col2, expr_col3 = st.columns(3)
        
        with expr_col1:
            st.metric("Apertura Boca", f"{apertura_boca:.1f}px")
        
        with expr_col2:
            st.metric("Apertura Ojos", f"I:{apertura_ojos_izq:.1f} D:{apertura_ojos_der:.1f}px")
        
        with expr_col3:
            st.metric("Inclinación Cabeza", f"{inclinacion_cabeza:.1f}°")
    else:
        st.error("No se pudo procesar la imagen")

else:
    # Mensaje de bienvenida
    st.info("Subí una imagen para comenzar la demostración")
    
    # Ejemplo visual
    st.markdown("### Ejemplo de Resultado")
    st.markdown("Esta aplicación simula la detección de landmarks faciales para demostrar los conceptos de análisis de expresiones.")
    
    # Mostrar una imagen de ejemplo
    try:
        st.image(
            "https://pyimagesearch.com/wp-content/uploads/2021/04/face_landmarks_3.jpg",
            caption="Ejemplo de landmarks faciales (68 puntos)",
            width=400
        )
    except:
        st.write("**Nota**: Esta aplicación es una demostración de conceptos de detección facial.")
        
        # Crear una imagen de ejemplo simple
        ejemplo = Image.new('RGB', (400, 300), color='lightgray')
        st.image(ejemplo, caption="Imagen de ejemplo", width=400)