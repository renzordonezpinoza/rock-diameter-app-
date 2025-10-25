import cv2
import numpy as np
import streamlit as st
import math

st.set_page_config(page_title="Rock Diameter Measurement", layout="wide")
st.title("🪨 Rock Diameter Measurement App (con D80)")

uploaded = st.file_uploader("📸 Sube una foto del cúmulo de rocas (con una regla visible)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

    st.markdown("### ⚙️ Calibración")
    st.write("1️⃣ Mide en píxeles la distancia entre los extremos de la regla (puedes usar un programa de medición o contar en la imagen).")
    pixel_dist = st.number_input("Distancia medida en la imagen (px)", min_value=1.0)
    real_mm = st.number_input("Longitud real de esa distancia (mm)", min_value=1.0, value=100.0)

    if st.button("🔍 Detectar rocas"):
        if pixel_dist > 0:
            px_per_mm = pixel_dist / real_mm
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(blur[th==255]) < np.mean(blur[th==0]):
                th = cv2.bitwise_not(th)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

            cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            diameters_mm = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 50:
                    continue
                equi_diam = 2.0 * math.sqrt(area / math.pi)
                diameters_mm.append(equi_diam / px_per_mm)
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0,0,255), 2)

            if diameters_mm:
                diameters_mm.sort()
                n = len(diameters_mm)
                avg = np.mean(diameters_mm)
                med = np.median(diameters_mm)
                D80 = np.percentile(diameters_mm, 80)

                st.success(f"Rocas detectadas: {n}")
                st.info(f"Promedio: {avg:.2f} mm | Mediana: {med:.2f} mm | **D80: {D80:.2f} mm**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Rocas detectadas", use_container_width=True)

                csv_data = "index,diameter_mm\n" + "\n".join([f"{i+1},{d:.3f}" for i, d in enumerate(diameters_mm)])
                st.download_button(
                    label="📥 Descargar datos (CSV)",
                    data=csv_data,
                    file_name="rock_diameters.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No se detectaron rocas. Verifica iluminación o calibración.")
