import streamlit as st
import numpy as np
from NDVIfunctions import (
    load_hsd_local,
    display_heatmap,
    calculate_ndvi,
    generate_wavelengths,
    get_red_nir_band_indices,
)

st.title("🌿 Hyperspectral Viewer: NDVI from .hsd File")

# Upload .hsd file
uploaded_hsd = st.file_uploader("📁 Upload a .hsd file", type=["hsd"])

if uploaded_hsd:
    # Load HSI cube
    hsi_cube, Y, X = load_hsd_local(uploaded_hsd)
    height, width, num_bands = hsi_cube.shape
    st.success(f"Loaded hyperspectral image of shape: {hsi_cube.shape} (H × W × Bands)")

    st.subheader("🔧 Wavelength & Band Selection for NDVI")

    ndvi_mode = st.radio(
        "How would you like to select bands for NDVI?",
        ["Manual Selection", "Auto Selection (Start, Step, #Bands)"]
    )

    if ndvi_mode == "Manual Selection":
        # Manual Band Selection
        red_idx = st.number_input("Enter Red Band Index", min_value=0, max_value=num_bands-1, value=20)
        nir_idx = st.number_input("Enter NIR Band Index", min_value=0, max_value=num_bands-1, value=40)
        wavelengths = np.arange(num_bands)  # Just for consistent behavior
        red_wl = red_idx  # Optional: Map index to wavelength if known
        nir_wl = nir_idx

    else:
        # Auto Band Selection
        start_nm = st.number_input("Starting Wavelength (nm)", value=400.0, step=1.0)
        step_nm = st.number_input("Wavelength Interval (nm)", value=5.0, step=0.1)
        entered_num_bands = st.number_input(
            "Number of Bands", value=num_bands, min_value=1, max_value=num_bands, step=1
        )
        wavelengths = generate_wavelengths(int(entered_num_bands), start_nm, step_nm)

        # Auto-select best red and NIR bands
        red_idx, nir_idx = get_red_nir_band_indices(wavelengths)
        red_wl, nir_wl = wavelengths[red_idx], wavelengths[nir_idx]
        st.success(f"Auto-selected Red: {red_wl:.1f} nm (band {red_idx}), NIR: {nir_wl:.1f} nm (band {nir_idx})")

    # Visualize selected bands
    st.subheader("🎨 Band Visualizations")
    display_heatmap(hsi_cube[:, :, red_idx], title=f"Red Band - {red_wl} nm")
    display_heatmap(hsi_cube[:, :, nir_idx], title=f"NIR Band - {nir_wl} nm")

    # NDVI Computation
    st.subheader("📈 NDVI Computation")
    ndvi = calculate_ndvi(hsi_cube, nir_idx, red_idx)
    display_heatmap(ndvi, title="NDVI Map (NIR - Red) / (NIR + Red)")