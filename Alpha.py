import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Introductory text
st.title("Alpha Angle")
st.write("This application shows a simplistic model that allows determining the hip flexion angle alpha from the seated position based on the beta angle, which is the angle between the neck and the x-axis in the xOz plane in the frontal view.")

# Slideshow images
st.subheader("Representative diagrams of the flexion situation and the variation of angles")

# Load the images
images = ["alpha1.png", "alpha2.png", "alpha3.png", "alpha4.png", "alpha5.png", "alpha6.png"]  # Replace with your own images
image_index = 0

if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

def resize_image(image, size=(400, 400)):
    return image.resize(size, Image.Resampling.LANCZOS)

def next_image():
    st.session_state.image_index = (st.session_state.image_index + 1) % len(images)

def previous_image():
    st.session_state.image_index = (st.session_state.image_index - 1) % len(images)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous"):
        previous_image()
with col2:
    image = Image.open(images[st.session_state.image_index])
    resized_image = resize_image(image)
    st.image(resized_image)
with col3:
    if st.button("Next"):
        next_image()

# Parameter input
st.subheader("Parameter Input")
C = st.number_input("Neck length in cm", min_value=0.0, max_value=100.0, value=6.0, step=0.5)
A = st.number_input("Femoral length in cm", min_value=0.0, max_value=100.0, value=42.0, step=0.5)
tf_deg = st.number_input("Femoral torsion in degrees", min_value=0.0, max_value=360.0, value=25.0, step=0.5)
CCD_deg = st.number_input("Cervico-diaphyseal angle (CCD) in degrees", min_value=0.0, max_value=360.0, value=120.0, step=0.5)
HKS_deg = st.number_input("HKS in degrees", min_value=0.0, max_value=360.0, value=7.0, step=0.5)
gamma_deg = st.number_input("Abduction/adduction angle (gamma) in degrees", min_value=-180.0, max_value=180.0, value=0.0, step=0.5)
angle_lim_deg = st.number_input("Limit angle in degrees before probable Hip Dislocation", min_value=0.0, max_value=360.0, value=10.0, step=0.5)
psi_deg = st.number_input("Xi angle in degrees", min_value=0.0, max_value=360.0, value=30.0, step=0.5)

# Chart display
st.subheader("Chart Based on Parameters")

def generate_chart(C, A, tf_deg, CCD_deg, HKS_deg, gamma_deg, angle_lim_deg, psi_deg):
    # Conversion of angles to radians
    a = np.radians(CCD_deg)
    b = np.radians(180 - CCD_deg - HKS_deg)
    c = np.radians(HKS_deg)
    tf = np.radians(tf_deg)
    gamma = np.radians(gamma_deg)

    # Length of the femur in cm
    B = A * np.sin(b) / np.sin(a)

    # Coordinates of vectors u and w
    w = np.array([0, B, 0])
    u = np.array([np.sin(np.pi - a + gamma) * C, -np.cos(np.pi - a + gamma) * C, C * np.sin(tf)])

    def rotation_matrix_x(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

    def rotate_vector(vector, rotation_matrix):
        return np.dot(rotation_matrix, vector)

    def calculate_signed_angle_with_x_in_xOz(vector):
        vector_xOz = np.array([vector[0], 0, vector[2]])
        x_axis = np.array([1, 0, 0])
        vector_xOz_norm = np.linalg.norm(vector_xOz)
        x_axis_norm = np.linalg.norm(x_axis)
        dot_product = np.dot(vector_xOz, x_axis)
        cos_angle = dot_product / (vector_xOz_norm * x_axis_norm)
        angle_rad = np.arccos(cos_angle)
        if vector[2] < 0:
            angle_rad = -angle_rad
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    theta_degrees = np.linspace(0, 120, 120)
    signed_angles_with_x_in_xOz = []

    for theta_deg in theta_degrees:
        theta = np.radians(theta_deg)
        R = rotation_matrix_x(theta)
        u_prime = rotate_vector(u, R)
        angle_with_x = calculate_signed_angle_with_x_in_xOz(u_prime)
        signed_angles_with_x_in_xOz.append(angle_with_x)

    plt.figure(figsize=(10, 6))
    plt.plot(theta_degrees, signed_angles_with_x_in_xOz, label="Angle with x-axis in xOz plane")
    plt.xlabel("Rotation angle alpha (degrees)")
    plt.ylabel("Beta angle (degrees)")
    plt.title("Evolution of the angle with x-axis in xOz plane as a function of θ")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 130, 10))
    plt.yticks(np.arange(-50, 55, 5))

    st.pyplot(plt)

    # Calculate and display the beta angle
    beta = angle_lim_deg - psi_deg
    st.write(f"The beta angle must not exceed: {beta}°")

if st.button("Generate Chart"):
    generate_chart(C, A, tf_deg, CCD_deg, HKS_deg, gamma_deg, angle_lim_deg, psi_deg)
st.write("Look at the corresponding alpha angle on the graph above.")
