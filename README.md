# Physics-Informed Neural Network (PINN) for Neutron Diffusion

This project uses a PINN to solve the 2D Steady-State Neutron Diffusion Equation. 
Unlike traditional numerical solvers, this model is **parametric**, allowing real-time 
exploration of how physical constants affect reactor flux.

## The Physics
The model solves:
$-D \nabla^2 \phi + \Sigma_a \phi = S$

## Features
- **Parametric Inference:** Adjust $D$ and $\Sigma_a$ in real-time.
- **Validation:** Compare PINN results against a Finite Difference Method (FDM) solver.
- **Interactive Visualization:** 3D surface plots via Plotly.

## How to Run
1. Clone the repo: `git clone https://github.com/your-username/repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
