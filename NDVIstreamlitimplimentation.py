import streamlit as st
from NDVIfunctions import load_hsd_local  # <- Update this if function is different
from NDVIfunctions import display_heatmap
from NDVIfunctions import calculate_ndvi

st.title("Hyperspectral Viewer: Answer Prototype (.hsd)")

uploaded_hsd = st.file_uploader("Upload .hsd file", type=["hsd"])

if uploaded_hsd:
    # Read file into buffer

    # Load HSI cube and metadata
    hsi_cube, Y, X = load_hsd_local(uploaded_hsd)  # Make sure this function returns (cube, metadata)

    st.success(f"Loaded image: shape = {hsi_cube.shape}")
    total_bands = hsi_cube.shape[2]

    # Band selection sliders
    band1 = st.slider("Select Red Band Index", 0, total_bands - 1, 30)
    band2 = st.slider("Select NIR Band Index", 0, total_bands - 1, 60)

    # Band visualizations
    st.subheader("Band Visualizations")
    display_heatmap(hsi_cube[:, :, band1], title=f"Band {band1}")
    display_heatmap(hsi_cube[:, :, band2], title=f"Band {band2}")

    # NDVI
    st.subheader("NDVI Computation")
    ndvi = calculate_ndvi(hsi_cube, band2, band1)
    display_heatmap(ndvi, title="NDVI")