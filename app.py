import os
os.environ["DDE_BACKEND"] = "pytorch"

# Import deepxde only after setting — and immediately force backend
import deepxde as dde
from deepxde.backend import set_default_backend

set_default_backend("pytorch")   # ← very explicit override (recent versions)
import torch
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import linalg, eye, kron, csr_matrix

st.set_page_config(page_title="Nuclear PINN", layout="wide")
st.title("☢️ Nuclear PINN")

# --- SIDEBAR ---
D_val = st.sidebar.slider("Diffusion Coefficient (D)", 0.5, 2.0, 1.0)
Sa_val = st.sidebar.slider("Absorption (Σa)", 0.01, 0.5, 0.1)
S_val = 1.0
N = 40

# --- 1. NUMERICAL SOLVER (FDM) ---
def solve_fdm(D, Sa, S, n_points):
    h = 1.0 / (n_points - 1)
    dim = n_points**2
    main_diag = (4 * D / h**2 + Sa) * np.ones(dim)
    off_diag = (-D / h**2) * np.ones(dim)
    
    I = eye(n_points)
    T = csr_matrix(np.diag(main_diag[:n_points]) + np.diag(off_diag[:n_points-1], 1) + np.diag(off_diag[:n_points-1], -1))
    A = kron(I, T) + kron(np.diag(np.ones(n_points-1), 1) + np.diag(np.ones(n_points-1), -1), np.diag(off_diag[:n_points]))
    b = np.full(dim, S)
    phi = linalg.spsolve(A, b)
    return phi.reshape((n_points, n_points))

# --- 2. LOAD PYTORCH MODEL ---
@st.cache_resource
def load_pytorch_model():
    net = dde.nn.FNN([4] + [64] * 4 + [1], "tanh", "Glorot uniform")
    
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    param_space = dde.geometry.Rectangle([0.5, 0.01], [2.0, 0.5])
    full_domain = dde.geometry.Rectangle(
        np.concatenate((geom.xmin, param_space.xmin)), 
        np.concatenate((geom.xmax, param_space.xmax))
    )
    data = dde.data.PDE(full_domain, lambda x,y: 0, [], num_domain=1)
    model = dde.Model(data, net)
    
    try:
        state_dict = torch.load("./models/nuclear_pinn_model.pt", map_location=torch.device('cpu'))
        model.net.load_state_dict(state_dict)
        model.net.eval()
        
        # Critical: Compile to set self.outputs as callable (PyTorch backend setup)
        model.compile("adam", lr=0.001)  # Dummy params; no real training here
        
        st.success("PyTorch PINN loaded and compiled ✓")
    except FileNotFoundError:
        st.error("Model file 'nuclear_pinn_model.pt' not found! Ensure it's saved from Colab with torch.save(model.net.state_dict(), 'nuclear_pinn_model.pt').")
        st.stop()
    except Exception as e:
        st.error(f"Loading failed: {e}")
        st.stop()
    
    return model

# --- MAIN APP LOGIC ---
with st.spinner("Computing Physics..."):
    # FDM
    phi_fdm = solve_fdm(D_val, Sa_val, S_val, N)
    
    # PINN
    model = load_pytorch_model()
    
    # Prepare Input
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    D_arr = np.full((N*N, 1), D_val)
    Sa_arr = np.full((N*N, 1), Sa_val)
    # PyTorch expects float32
    test_pts = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], D_arr, Sa_arr)).astype(np.float32)
    
    # Predict
    phi_pinn = model.predict(test_pts).reshape(N, N)

# --- PLOTTING ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Numerical Truth (FDM)")
    st.plotly_chart(go.Figure(data=[go.Surface(z=phi_fdm, colorscale='Blues')]), use_container_width=True)

with col2:
    st.subheader("AI Prediction (PyTorch)")
    st.plotly_chart(go.Figure(data=[go.Surface(z=phi_pinn, colorscale='Reds')]), use_container_width=True)

# Error
st.divider()
error = np.abs(phi_fdm - phi_pinn)
st.subheader(f"Mean Error: {np.mean(error):.5f}")
st.plotly_chart(go.Figure(data=[go.Heatmap(z=error, colorscale='Viridis')]), use_container_width=True)
