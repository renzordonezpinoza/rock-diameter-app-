import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Medidor de Diámetro de Rocas", layout="wide")

st.title("🪨 Medidor de Diámetro Promedio y D80 de Rocas")
st.write(
    """
    Carga una foto del cúmulo de rocas con una **regla visible para calibración**.
    Luego marca el extremo de la regla para definir la escala y obtén:
    - Diámetro promedio
    - D80 (el diámetro menor que el 80% de las rocas)
    """
)

uploaded_file = st.file_uploader("📸 Cargar imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen de forma segura
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("⚠️ No se pudo leer la imagen. Intenta subir un archivo .jpg o .png válido.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        # Calibración de escala
        st.subheader("📏 Calibración de escala")
        scale_length_mm = st.number_input(
            "Longitud de la regla visible (en mm):", min_value=1.0, value=100.0, step=1.0
        )
        scale_px = st.number_input(
            "Longitud de la regla en píxeles (aproximadamente, según la imagen):",
            min_value=1.0,
            value=100.0,
            step=1.0,
        )
        scale_factor = scale_length_mm / scale_px  # mm/px

        # Procesamiento de imagen
        st.subheader("⚙️ Procesamiento de imagen")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diameters_mm = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter_px = radius * 2
            diameter_mm = diameter_px * scale_factor
            if diameter_mm > 1:
                diameters_mm.append(diameter_mm)

        if diameters_mm:
            diameters_mm = np.array(diameters_mm)
            mean_diameter = np.mean(diameters_mm)
            d80 = np.percentile(diameters_mm, 80)

            st.success(f"**Diámetro promedio:** {mean_diameter:.2f} mm")
            st.success(f"**D80 (diámetro del 80% de las rocas):** {d80:.2f} mm")

            # Visualización de detecciones
            output_img = image.copy()
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(output_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="Detección de rocas", use_container_width=True)

            # Generar CSV descargable
            csv_data = "index,diameter_mm\n" + "\n".join(
                [f"{i+1},{d:.3f}" for i, d in enumerate(diameters_mm)]
            )
            st.download_button(
                label="📥 Descargar resultados (CSV)",
                data=csv_data,
                file_name="diametros_rocas.csv",
                mime="text/csv",
            )
        else:
            st.warning("No se detectaron contornos válidos. Ajusta la calidad de la imagen o el enfoque.")
else:
    st.info("Por favor, carga una imagen para comenzar.")
