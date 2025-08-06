import io
import numpy as np

def write_HSD_to_buffer(file_path: str, data: np.ndarray):
    """
    Writes hyperspectral data to a binary file.

    Parameters:
    -----------
    file_path : str
        Path to save the output binary file.
    data : np.ndarray
        Hyperspectral data in the shape (Y, Z, X). This will be transposed to (Y, X, Z)
        before saving if needed.
    """
    # Transpose data if needed (optional, based on how you read it later)
    data = data.transpose(0, 2, 1)
    
    try:
        with open(file_path, 'wb') as file:
            file.write(data.tobytes())
        print(f"File successfully saved at: {file_path}")
    
    except Exception as e:
        print(f"Error saving file: {e}")



def read_HSD_from_buffer(buffer: bytes, band: int = 141) -> np.ndarray:
    """
    Reads hyperspectral data from a binary buffer and returns the image cube.

    Parameters:
    -----------
    buffer : bytes
        Binary content of the .hsd or .dat file.
    band : int
        Number of spectral bands (default = 141).

    Returns:
    --------
    np.ndarray
        Hyperspectral image cube with shape (Y, Z, X) transposed to (Y, X, Z).
    int
        Y dimension (height).
    int
        X dimension (width).
    """
    file_size = len(buffer)  # Get file size

    # Detect camera type based on file size
    if file_size == 370623040:
        HSData, header, Y, X = read_HSC180X(buffer)
    elif file_size == 87630400:
        HSData, header, Y, X = read_HSC170X_old(buffer)
    elif file_size == 44315200:
        HSData, Y, X = read_HSC170X_new(buffer)
    elif file_size == 585755200:
        HSData, header, Y, X = read_HSC180X_CL(buffer)
    elif file_size == 14805000:
        HSData, Y, X = read_custom(buffer)
    else:
        raise ValueError(f"Unsupported file size: {file_size} bytes")

    return HSData.transpose(0, 2, 1), Y, X


def read_custom(buffer):
    """
    Reads custom hyperspectral data format (uint8) from buffer.

    Parameters:
    -----------
    buffer : bytes
        Binary content containing header and raw hyperspectral data.

    Returns:
    --------
    np.ndarray
        Hyperspectral data with shape (Y, Z, X).
    int
        Y dimension.
    int
        X dimension.
    """
    X = 350
    Y = 300
    Z = 141
    RAW_len = X * Y * Z
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint8)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, Y, X


def read_HSC180X_CL(buffer):
    """
    Reads data from HSC180X_CL format (uint16).

    Parameters:
    -----------
    buffer : bytes
        Binary content containing header and data.

    Returns:
    --------
    np.ndarray
        Hyperspectral data with shape (Y, Z, X).
    bytes
        Header.
    int
        Y dimension.
    int
        X dimension.
    """
    X = 1920
    Y = 1080
    Z = 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, header, Y, X


def read_HSC180X(buffer):
    """
    Reads data from HSC180X format (uint16).

    Parameters:
    -----------
    buffer : bytes
        Binary content containing header and data.

    Returns:
    --------
    np.ndarray
        Hyperspectral data with shape (Y, Z, X).
    bytes
        Header.
    int
        Y dimension.
    int
        X dimension.
    """
    X = 1280
    Y = 1024
    Z = 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, header, Y, X


def read_HSC170X_old(buffer):
    """
    Reads data from older HSC170X format (uint16, converted to uint8).

    Parameters:
    -----------
    buffer : bytes
        Binary content containing header and data.

    Returns:
    --------
    np.ndarray
        Hyperspectral data with shape (Y, Z, X), converted to uint8.
    int
        Y dimension.
    int
        X dimension.
    """
    X = 640
    Y = 480
    Z = 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint16)
    dat = dat.astype(np.uint8)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, Y, X


def read_HSC170X_new(buffer):
    """
    Reads data from newer HSC170X format (uint8).

    Parameters:
    -----------
    buffer : bytes
        Binary content containing header and data.

    Returns:
    --------
    np.ndarray
        Hyperspectral data with shape (Y, Z, X).
    int
        Y dimension.
    int
        X dimension.
    """
    X = 640
    Y = 480
    Z = 141
    RAW_len = X * Y * Z
    header = buffer[:len(buffer) - RAW_len]
    header_size = len(header)
    print("Header Size:", header_size, "bytes")
    dat = np.frombuffer(buffer[header_size:], dtype=np.uint8)
    HSData = np.reshape(dat, (Y, Z, X))
    return HSData, Y, X

def load_hsd_local(file_obj):
    """
    Loads a hyperspectral .hsd or .dat file from a file-like object into a numpy array.

    Parameters:
    -----------
    file_obj : file-like object
        File-like object returned by Streamlit's st.file_uploader.

    Returns:
    --------
    np.ndarray
        Hyperspectral data array with shape (Y, X, Z).
    int
        Y dimension (height).
    int
        X dimension (width).
    """
    buffer = file_obj.read()
    HSData, Y, X = read_HSD_from_buffer(buffer)  # Ensure this function is defined elsewhere
    return HSData, Y, X

import matplotlib.pyplot as plt
import streamlit as st

def display_heatmap(image_band, title="Band"):
    """
    Display a single 2D image band as a heatmap in Streamlit using matplotlib.

    Parameters:
        image_band (numpy.ndarray): 2D array representing a single spectral band.
        title (str): Title for the heatmap plot. Default is "Band".

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.imshow(image_band, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

import numpy as np

def calculate_ndvi(hsi_cube, nir_band_idx, red_band_idx):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from a hyperspectral cube.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters:
        hsi_cube (numpy.ndarray): 3D hyperspectral image cube (rows x cols x bands).
        nir_band_idx (int): Index of the Near-Infrared (NIR) band.
        red_band_idx (int): Index of the Red band.

    Returns:
        numpy.ndarray: 2D NDVI image with values ranging from -1 to 1.
    """
    nir = hsi_cube[:, :, nir_band_idx].astype(np.float32)
    red = hsi_cube[:, :, red_band_idx].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)  # avoid division by zero
    return ndvi