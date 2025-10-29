# Detector de Landmarks Faciales

Aplicación web para detectar landmarks faciales y analizar expresiones usando OpenCV y Streamlit.

## Características

- Detección de rostros y características faciales (68 puntos aproximados)
- Análisis de expresiones faciales (apertura de boca, ojos, inclinación de cabeza)
- Interfaz web interactiva
- Múltiples estilos de visualización
- Compatible con Python 3.13

## Tecnologías

- **OpenCV**: Detección facial y procesamiento de imágenes
- **Streamlit**: Framework web
- **Pillow**: Manipulación de imágenes
- **NumPy**: Procesamiento numérico
- **Python 3.13+**

## Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/AlfredoPoblete/facial_landmarks_proces_imagen_2025.git
cd facial_landmarks_proces_imagen_2025

# Crear entorno virtual (opcional)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py

# La aplicación estará disponible en http://localhost:8501
```

## Despliegue

La aplicación está desplegada en Streamlit Cloud: [https://facial-landmark.streamlit.app/](https://facial-landmark.streamlit.app/)