# -*- coding: utf-8 -*-
"""
Lorenz System Simulator, Analyzer, and Visualizer (PEP 8 Compliant)

This script simulates the Lorenz system of differential equations, a classic
model derived from atmospheric convection studies, famous for its butterfly-shaped
chaotic attractor.

It performs the following actions:
1.  **Defines Lorenz System:** Implements the system of three ODEs.
2.  **Solves ODEs:** Integrates the Lorenz equations over time using SciPy's
    odeint for multiple nearby initial conditions.
3.  **Calculates LLE:** Estimates the Largest Lyapunov Exponent (LLE) to quantify
    the system's sensitivity to initial conditions (chaos).
4.  **Generates Static Plots:** Creates a multi-panel figure (Matplotlib) showing
    the 3D attractor and its 2D projections.
5.  **Generates Animation:** Produces an MP4 video animation of the attractor's
    trajectory (requires FFmpeg or another supported writer like ImageMagick for GIF).
6.  **Generates Interactive HTML:** Creates an interactive 3D plot using Plotly
    for exploration in a web browser.
7.  **Saves Results:** Stores LLE results and outputs in a dedicated directory.
8.  **Opens Output:** Attempts to automatically open the generated HTML file
    using the system's default application.

Dependencies:
    - Python 3.x
    - numpy
    - scipy
    - matplotlib
    - plotly
    - ffmpeg (External command-line tool, recommended for MP4 animation)
    - OR imagemagick (External command-line tool, for GIF animation)

Usage:
  Run from the command line: python <script_name>.py
  Outputs saved in 'lorenz_output' directory.
"""

# Standard library imports
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import warnings
import webbrowser
from typing import List, Tuple, Optional, Callable, Sequence

# Third-party imports
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # Explicitly needed for projection='3d'
from scipy.integrate import odeint

# --- Backend Configuration ---
# Use 'Agg' backend for non-interactive environments. Suppress related warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        matplotlib.use("Agg")
    except ImportError:
        print("[!] Warning: Failed to set Matplotlib backend to 'Agg'.")

# =====================================
# Configuration Constants
# =====================================

# --- Output Settings ---
OUTPUT_DIR_NAME: str = "lorenz_output"
VIDEO_ANIMATION_FILENAME: str = "lorenz_animation.mp4" # MP4 is generally preferred
# ANIMATION_FILENAME = "lorenz_animation.gif" # GIF alternative
INTERACTIVE_HTML_FILENAME: str = "lorenz_interactive.html" # Added for Plotly
STATIC_PLOT_FILENAME: str = "lorenz_static_plots.png"
LLE_RESULTS_FILENAME: str = "lorenz_lle_results.txt"

# --- Lorenz System Parameters (Classic Chaotic Regime) ---
LORENZ_PARAM_SIGMA: float = 10.0 # Prandtl number
LORENZ_PARAM_RHO: float = 28.0   # Rayleigh number
LORENZ_PARAM_BETA: float = 8.0 / 3.0 # Geometric factor

# --- Initial States (Near one of the unstable fixed points) ---
INITIAL_STATES: List[List[float]] = [
    [0.0, 1.0, 1.05],    # Primary trajectory
    [0.01, 1.0, 1.05],   # Slightly perturbed in x
    [0.0, 1.01, 1.05],   # Slightly perturbed in y
]

# --- Simulation Time Spans ---
VIS_T_END: float = 40.0        # End time for visualization trajectories
VIS_NUM_STEPS: int = 15000     # Number of steps for visualization
LLE_T_END: float = 150.0       # End time for LLE simulation (needs longer time)
LLE_NUM_STEPS: int = 30000     # Number of steps for LLE simulation

# --- LLE Calculation Settings ---
LLE_EPSILON: float = 1e-9      # Initial separation for LLE

# --- Animation Settings ---
# Check if preferred writer ('ffmpeg' for MP4) is available
DEFAULT_ANIMATION_WRITER: str = "ffmpeg" # For MP4
# DEFAULT_ANIMATION_WRITER: str = "imagemagick" # For GIF
if not writers.is_available(DEFAULT_ANIMATION_WRITER):
    print(f"[!] Warning: Matplotlib writer '{DEFAULT_ANIMATION_WRITER}' not found.")
    print("    Animation saving might fail.")
    print("    Consider installing FFmpeg (for MP4) or ImageMagick (for GIF).")
    # Fallback or disable:
    ANIMATION_WRITER: Optional[str] = None # Disable animation saving attempt
else:
    ANIMATION_WRITER: Optional[str] = DEFAULT_ANIMATION_WRITER

ANIMATION_FPS: int = 30        # Frames per second
ANIMATION_DPI: int = 100       # Video frame resolution
FRAME_STEP: int = 10           # Use every Nth point for animation frame
TAIL_LENGTH: int = 300         # Tail length in number of *points* (adjust if needed)
ANIMATION_INTERVAL: int = 25   # Delay between frames in milliseconds

# --- Visualization Settings ---
STATIC_PLOT_DPI: int = 150     # Static plot resolution

# Type alias for state vector
State = np.ndarray
# Type alias for system parameters tuple (sigma, rho, beta)
LorenzParams = Tuple[float, float, float]

# =====================================
# Function Definitions
# =====================================

def lorenz_system(state: State, t: float, sigma: float, rho: float, beta: float) -> State:
    """
    Define the differential equations for the Lorenz system.

    Args:
        state: Current state vector [x, y, z].
        t: Current time t (required by odeint).
        sigma: Lorenz system parameter σ (Prandtl number).
        rho: Lorenz system parameter ρ (Rayleigh number).
        beta: Lorenz system parameter β (Geometric factor).

    Returns:
        np.ndarray: Array containing the derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# --- solve_visual_trajectories (Generic ODE solver) ---
def solve_visual_trajectories(
    initial_states: List[List[float]],
    t_span: np.ndarray,
    system_func: Callable, # Generic callable type hint
    sys_params: Tuple,      # Parameters tuple for system_func
) -> List[np.ndarray]:
    """
    Solves the ODE system for visualization for multiple initial states.

    Args:
        initial_states: A list of initial state vectors [[x0, y0, z0], ...].
        t_span: Array of time points for the solution.
        system_func: The function defining the ODE system (e.g., lorenz_system).
        sys_params: Tuple of system parameters required by system_func.

    Returns:
        A list of numpy arrays, each holding a trajectory [x(t), y(t), z(t)].
    """
    print("[*] Solving trajectories for visualization...")
    solutions = []
    num_traj = len(initial_states)
    for i, init_state in enumerate(initial_states):
        print(f"  [*] Integrating trajectory {i+1}/{num_traj}...")
        try:
            solution = odeint(system_func, init_state, t_span, args=sys_params)
            solutions.append(solution)
        except Exception as e:
            print(f"  [!] Error integrating trajectory {i+1}: {e}")
            print("      Skipping this trajectory for visualization.")
    if solutions:
        print("  [+] Visualization integration complete.")
    else:
        print("  [!] No trajectories were successfully integrated for visualization.")
    return solutions

# --- calculate_lle (Generic LLE calculator) ---
def calculate_lle(
    initial_state: List[float],
    t_span_lle: np.ndarray,
    system_func: Callable, # Generic callable type hint
    sys_params: Tuple,      # Parameters tuple for system_func
    epsilon: float,
) -> Tuple[Optional[float], str, float]:
    """
    Estimates the Largest Lyapunov Exponent (LLE) for a given dynamical system.

    Args:
        initial_state: The starting state vector [x0, y0, z0].
        t_span_lle: Array of time points for the LLE calculation.
        system_func: The function defining the ODE system.
        sys_params: Tuple of system parameters required by system_func.
        epsilon: The initial small separation distance between trajectories.

    Returns:
        A tuple containing:
        - Estimated LLE (float) or None if calculation failed.
        - Interpretation string of the LLE result.
        - Actual simulation time used (float).
    """
    print("\n[*] Calculating Largest Lyapunov Exponent (LLE)...")
    if len(t_span_lle) < 2:
        print("  [!] Error: LLE time span needs at least two points.")
        return None, "LLE Calculation Failed (invalid time span).", 0.0

    dt_lle = t_span_lle[1] - t_span_lle[0]
    total_steps_lle = len(t_span_lle)
    lle_sum = 0.0
    state0 = np.array(initial_state, dtype=float)
    perturb_vector = np.zeros_like(state0)
    perturb_vector[0] = epsilon # Perturb only the first component
    state1 = state0 + perturb_vector
    current_t = 0.0

    print("  [*] Running reference and perturbed simulations...")
    try:
        for i in range(total_steps_lle - 1):
            t_step_span = [current_t, current_t + dt_lle]
            sol0 = odeint(system_func, state0, t_step_span, args=sys_params)
            sol1 = odeint(system_func, state1, t_step_span, args=sys_params)

            if sol0.shape[0] < 2 or sol1.shape[0] < 2:
                 print(f"\n  [!] Error: ODE integration failed at LLE step {i}.")
                 return None, "LLE Calculation Failed (integration error).", current_t

            state0 = sol0[1]
            state1 = sol1[1]
            diff_vector = state1 - state0
            distance = np.linalg.norm(diff_vector)

            if distance > 1e-15: # Avoid log(0) and division issues
                lle_sum += np.log(distance / epsilon)
                state1 = state0 + (diff_vector / distance) * epsilon # Rescale
            else:
                print(f"  [!] Warning: LLE trajectories separation near zero at step {i}. Re-perturbing.")
                state1 = state0 + perturb_vector

            current_t += dt_lle
            if (i + 1) % (max(1, total_steps_lle // 20)) == 0: # Update progress ~20 times
                progress = 100 * (i + 1) / total_steps_lle
                print(f"    LLE Progress: {progress:.1f}%", end='\r')

        print("\n  [+] LLE Calculation loop finished.")

    except Exception as e:
        print(f"\n  [!] An error occurred during LLE calculation: {e}")
        return None, f"LLE Calculation Failed (Runtime Error: {e}).", current_t

    lle_estimate: Optional[float] = None
    lle_interpretation: str = "LLE Calculation Failed (Unknown error after loop)."

    if current_t > 1e-9:
        lle_estimate = lle_sum / current_t
        print(f"  [+] LLE Calculation Complete. Total time simulated: {current_t:.2f} units.")
        result_text = f"Estimated LLE: {lle_estimate:.4f}\n"
        if lle_estimate > 0.01:
            result_text += "  Interpretation: Positive LLE suggests chaotic behavior (sensitive dependence on initial conditions)."
        elif lle_estimate < -0.01:
            result_text += "  Interpretation: Negative LLE suggests convergence to a stable fixed point or limit cycle."
        else:
            result_text += "  Interpretation: LLE near zero suggests quasi-periodic or periodic behavior, or potentially a bifurcation point."
        lle_interpretation = result_text
        print(f"\n{lle_interpretation}")
    else:
        lle_interpretation = "LLE Calculation Failed (simulation time near zero)."
        print(f"  [!] {lle_interpretation}")

    return lle_estimate, lle_interpretation, current_t

# --- save_lle_results (Adapted for Lorenz parameters) ---
def save_lle_results(
    filepath: pathlib.Path,
    params: LorenzParams, # Takes the (sigma, rho, beta) tuple
    sim_time: float,
    target_time: float,
    num_steps: int,
    epsilon: float,
    interpretation: str,
) -> None:
    """
    Saves LLE calculation results and Lorenz parameters to a text file.

    Args:
        filepath: Path object for the output text file.
        params: Tuple of Lorenz parameters (sigma, rho, beta).
        sim_time: Actual simulation time achieved.
        target_time: Target simulation time for LLE.
        num_steps: Number of integration steps used.
        epsilon: Initial separation distance used.
        interpretation: String summarizing the LLE result.
    """
    print(f"[*] Saving LLE results to: {filepath}")
    sigma, rho, beta = params
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("=" * 30 + "\n")
            f.write(" Lorenz System Analysis Results\n")
            f.write("=" * 30 + "\n\n")
            f.write("System Parameters:\n")
            # Use LaTeX for Greek letters
            f.write(f"  Sigma (σ) = {sigma:.4f}\n")
            f.write(f"  Rho   (ρ) = {rho:.4f}\n")
            f.write(f"  Beta  (β) = {beta:.4f} (or {beta*3:.0f}/3)\n\n") # Show fraction too
            f.write("LLE Calculation Settings:\n")
            f.write(f"  Target Integration Time: {target_time:.2f} units\n")
            f.write(f"  Actual Integration Time: {sim_time:.2f} units\n")
            f.write(f"  Number of Steps:         {num_steps - 1}\n")
            f.write(f"  Initial Separation (ε):  {epsilon:.2e}\n\n")
            f.write("Lyapunov Exponent Results:\n")
            f.write(f"{interpretation}\n")
        print(f"  [+] LLE results saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not write LLE results to file: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred while saving LLE results: {e}")

# --- generate_save_static_plots (Adapted for Lorenz) ---
def generate_save_static_plots(
    filepath: pathlib.Path,
    solutions: List[np.ndarray],
    t_span: np.ndarray,
    initial_states: List[List[float]],
    params: LorenzParams, # Takes the (sigma, rho, beta) tuple
    lle_est: Optional[float],
    dpi: int,
) -> None:
    """
    Generates and saves static plots (3D and 2D projections) of the Lorenz attractor.

    Args:
        filepath: Path object for the output PNG file.
        solutions: List of trajectory arrays.
        t_span: Time array corresponding to the solutions.
        initial_states: List of starting points used.
        params: Tuple of Lorenz parameters (sigma, rho, beta).
        lle_est: Estimated LLE value (or None).
        dpi: Dots per inch for the saved image.
    """
    if not solutions:
        print("[!] Skipping static plot generation: No solution data available.")
        return

    print(f"\n[*] Generating and saving static plots to: {filepath.name}")
    sigma, rho, beta = params
    fig_static = plt.figure(figsize=(14, 10), dpi=dpi)

    # --- Title (Using LaTeX for Greek letters) ---
    title_lle_part = f" | LLE ≈ {lle_est:.4f}" if lle_est is not None else ""
    # Note: Need raw string (r"...") or double backslashes (\\) for LaTeX in f-string
    title = (
        rf"Lorenz System ($\sigma$={sigma:.1f}, $\rho$={rho:.1f}, $\beta$={beta:.2f})"
        f"{title_lle_part}"
    )
    fig_static.suptitle(title, fontsize=16)

    # --- 3D Plot ---
    ax3d = fig_static.add_subplot(2, 2, 1, projection="3d")
    num_trajectories = len(solutions)
    all_x = np.concatenate([s[:, 0] for s in solutions])
    all_y = np.concatenate([s[:, 1] for s in solutions])
    all_z = np.concatenate([s[:, 2] for s in solutions])

    cmap_3d = 'coolwarm' # Classic Lorenz colormap
    linewidth_3d = 0.6
    alpha_3d = 0.8

    for i, solution in enumerate(solutions):
        x_sol, y_sol, z_sol = solution[:, 0], solution[:, 1], solution[:, 2]
        points = np.array([x_sol, y_sol, z_sol]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_span.min(), t_span.max())
        lc = Line3DCollection(
            segments, cmap=cmap_3d, norm=norm, linewidth=linewidth_3d, alpha=alpha_3d
        )
        lc.set_array(t_span[:-1]) # Color segments by time
        ax3d.add_collection(lc)

    ax3d.set_xlabel("X Axis")
    ax3d.set_ylabel("Y Axis")
    ax3d.set_zlabel("Z Axis")
    ax3d.set_title(f"3D View ({num_trajectories} Trajectories)")
    ax3d.grid(True, linestyle='--', alpha=0.5)

    starts = np.array(initial_states)
    ax3d.scatter(
        starts[:, 0], starts[:, 1], starts[:, 2],
        color='black', s=70, marker='o', edgecolor='white',
        label='Start Points', depthshade=False, zorder=10
    )
    ax3d.legend(loc='upper left')

    pad = 0.05 # Padding factor for axis limits
    ax3d.set_xlim(all_x.min()*(1-pad), all_x.max()*(1+pad))
    ax3d.set_ylim(all_y.min()*(1-pad), all_y.max()*(1+pad))
    ax3d.set_zlim(all_z.min()*(1-pad), all_z.max()*(1+pad))

    # Adjust view angle for Lorenz 'butterfly'
    ax3d.view_init(elev=25, azim=-135)

    # --- 2D Projections (using the first trajectory) ---
    x_p, y_p, z_p = solutions[0][:, 0], solutions[0][:, 1], solutions[0][:, 2]
    scatter_args = {'c': t_span, 'cmap': 'viridis', 's': 0.5, 'alpha': 0.7} # Different cmap for projections

    ax_xy = fig_static.add_subplot(2, 2, 2)
    ax_xy.scatter(x_p, y_p, **scatter_args); ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y"); ax_xy.set_title("X-Y Projection"); ax_xy.grid(True, linestyle=':', alpha=0.6); ax_xy.set_aspect('equal', adjustable='box')

    ax_xz = fig_static.add_subplot(2, 2, 3)
    ax_xz.scatter(x_p, z_p, **scatter_args); ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z"); ax_xz.set_title("X-Z Projection"); ax_xz.grid(True, linestyle=':', alpha=0.6); ax_xz.set_aspect('equal', adjustable='box')

    ax_yz = fig_static.add_subplot(2, 2, 4)
    ax_yz.scatter(y_p, z_p, **scatter_args); ax_yz.set_xlabel("Y"); ax_yz.set_ylabel("Z"); ax_yz.set_title("Y-Z Projection"); ax_yz.grid(True, linestyle=':', alpha=0.6); ax_yz.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    try:
        fig_static.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  [+] Static plots saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not save static plots: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred while saving static plots: {e}")
    finally:
        plt.close(fig_static) # Ensure figure is closed


# --- generate_save_animation (Adapted for Lorenz) ---
def generate_save_animation(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    frame_step: int,
    tail_len_points: int, # Renamed for clarity: tail length in data points
    interval_ms: int,
    writer_name: Optional[str],
    fps: int,
    dpi: int,
) -> bool:
    """
    Generates and saves an animation of the Lorenz attractor trajectory.

    Args:
        filepath: Path object for the output animation file (MP4/GIF).
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        frame_step: Step size through data for each animation frame.
        tail_len_points: Length of trajectory tail in number of *data points*.
        interval_ms: Delay between frames in milliseconds.
        writer_name: Name of the Matplotlib animation writer (e.g., 'ffmpeg').
                     If None, saving is skipped.
        fps: Frames per second for the output video.
        dpi: Dots per inch for the video frames.

    Returns:
        True if animation was saved successfully, False otherwise.
    """
    if writer_name is None:
        print("\n[*] Skipping animation generation: No valid writer specified.")
        return False
    if solution_data.size == 0:
         print("\n[*] Skipping animation generation: No solution data available.")
         return False

    print(f"\n[*] Generating animation (using '{writer_name}' writer)...")
    x_anim, y_anim, z_anim = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]
    num_points = len(t_span)
    num_frames = num_points // frame_step

    if num_frames < 2:
        print("  [!] Error: Not enough data points for animation.")
        return False

    fig_anim = plt.figure(figsize=(10, 8), dpi=dpi)
    ax_anim = fig_anim.add_subplot(111, projection='3d')

    line_anim, = ax_anim.plot([], [], [], lw=1.5, color='blue') # Use blue for line
    point_anim, = ax_anim.plot([], [], [], 'o', color='red', markersize=6, zorder=10) # Red marker

    ax_anim.set_xlim(x_anim.min(), x_anim.max())
    ax_anim.set_ylim(y_anim.min(), y_anim.max())
    ax_anim.set_zlim(z_anim.min(), z_anim.max())
    ax_anim.set_xlabel("X Axis")
    ax_anim.set_ylabel("Y Axis")
    ax_anim.set_zlabel("Z Axis")
    ax_anim.set_title("Lorenz Attractor Animation")
    ax_anim.grid(True, linestyle=':', alpha=0.4)
    ax_anim.view_init(elev=25, azim=-135) # Consistent view angle

    def update(frame_idx_scaled: int) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
        """Updates the line and point data for each frame."""
        current_data_idx = frame_idx_scaled * frame_step
        # Tail start index calculation based on *data points*
        tail_start_data_idx = max(0, current_data_idx - tail_len_points)

        # Update tail line data (take every point in the tail range)
        line_anim.set_data(x_anim[tail_start_data_idx : current_data_idx+1],
                           y_anim[tail_start_data_idx : current_data_idx+1])
        line_anim.set_3d_properties(z_anim[tail_start_data_idx : current_data_idx+1])

        # Update current position marker data
        point_anim.set_data(x_anim[current_data_idx : current_data_idx+1],
                            y_anim[current_data_idx : current_data_idx+1])
        point_anim.set_3d_properties(z_anim[current_data_idx : current_data_idx+1])
        return line_anim, point_anim

    print(f"  [*] Creating animation with {num_frames} frames...")
    try:
        anim = FuncAnimation(
            fig_anim, update, frames=num_frames,
            interval=interval_ms, blit=True, repeat=False
            )
    except Exception as e:
         print(f"  [!] Error initializing FuncAnimation: {e}")
         plt.close(fig_anim)
         return False

    animation_saved = False
    print(f"  [*] Saving animation to: {filepath.name} (this may take time)...")
    try:
        def progress_update(i: int, n: int):
            if n > 0 and ((i + 1) % 20 == 0 or (i + 1) == n): # Update every 20 frames
                print(f"    Saving frame {i+1}/{n} ({100*(i+1)/n:.1f}%)", end='\r')

        anim.save(filepath, writer=writer_name, fps=fps, dpi=dpi, progress_callback=progress_update)
        print("\n  [+] Animation saved successfully.")
        animation_saved = True
    except FileNotFoundError:
        print(f"\n  [!] Error: Animation writer '{writer_name}' command not found or failed.")
        print(f"      Ensure '{writer_name}' (e.g., FFmpeg/ImageMagick) is installed and in PATH.")
    except Exception as e:
        print(f"\n  [!] Error occurred while saving animation: {e}")
    finally:
        print(" " * 80, end='\r') # Clear progress line
        plt.close(fig_anim)

    return animation_saved


# --- generate_save_interactive_html (Added for Lorenz) ---
def generate_save_interactive_html(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    params: LorenzParams, # Takes the (sigma, rho, beta) tuple
) -> bool:
    """
    Generates and saves an interactive 3D plot of the Lorenz attractor using Plotly.

    Args:
        filepath: Path object for the output HTML file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        params: Tuple of Lorenz parameters (sigma, rho, beta).

    Returns:
        True if HTML was saved successfully, False otherwise.
    """
    if solution_data.size == 0:
         print("\n[*] Skipping interactive HTML generation: No solution data available.")
         return False

    print(f"\n[*] Generating interactive HTML plot to: {filepath.name}")
    sigma, rho, beta = params
    x_trace, y_trace, z_trace = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]

    try:
        trace = go.Scatter3d(
            x=x_trace, y=y_trace, z=z_trace, mode='lines',
            line=dict(color=t_span, colorscale='Viridis', width=3, # Viridis often good for time
                      colorbar=dict(title='Time')),
            name='Lorenz Trajectory'
        )

        layout = go.Layout(
            title=dict(
                # Use HTML entities for Greek letters in Plotly title
                text=(f"Interactive Lorenz Attractor<br>"
                      f"(σ={sigma:.1f}, ρ={rho:.1f}, β={beta:.2f})"),
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
                xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                # Adjust aspect ratio and camera for Lorenz
                aspectratio=dict(x=1, y=1, z=0.7), # Similar Z compression to static plot view
                camera_eye=dict(x=1.5, y=-1.5, z=1.0) # Adjust for good initial view
            ),
            margin=dict(l=10, r=10, b=10, t=50),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(filepath, include_plotlyjs='cdn') # Use CDN

        print(f"  [+] Interactive HTML saved successfully to {filepath.name}")
        return True

    except ImportError:
        print("  [!] Error: Plotly library is not installed (pip install plotly).")
        return False
    except Exception as e:
        print(f"  [!] An unexpected error occurred generating interactive HTML: {e}")
        return False


# --- open_file_cross_platform (Standard version) ---
def open_file_cross_platform(filepath: pathlib.Path) -> None:
    """
    Attempts to open the specified file using the default system application.
    Prioritizes 'termux-open' if available, otherwise uses 'webbrowser'.

    Args:
        filepath: Path object of the file to open.
    """
    print(f"\n[*] Attempting to open file: {filepath.resolve()}")
    file_uri = filepath.resolve().as_uri() # Use file:// URI

    termux_open_cmd = shutil.which('termux-open')
    termux_used = False

    if termux_open_cmd:
        print(f"  [*] 'termux-open' command found. Attempting use...")
        command = [termux_open_cmd, str(filepath.resolve())]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
            print(f"  [+] 'termux-open' executed successfully.")
            if result.stdout: print(f"    Output: {result.stdout.strip()}")
            termux_used = True
        except FileNotFoundError:
            print(f"  [!] Error: '{termux_open_cmd}' not found during execution.")
        except subprocess.CalledProcessError as e:
            print(f"  [!] Error: 'termux-open' failed (code {e.returncode}).")
            if e.stderr: print(f"    Stderr: {e.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("  [!] Error: 'termux-open' command timed out.")
        except Exception as e:
            print(f"  [!] Unexpected error executing 'termux-open': {e}")

    if not termux_used:
        if termux_open_cmd:
             print("  [*] 'termux-open' failed, falling back to standard webbrowser...")
        else:
             print("  [*] 'termux-open' not found. Using standard webbrowser...")

        try:
            opened = webbrowser.open(file_uri)
            if opened:
                print(f"  [+] Standard webbrowser requested opening: {file_uri}")
            else:
                print(f"  [!] Standard webbrowser reported failure to open.")
                print(f"      Platform: {platform.system()}. Try opening manually: {filepath.resolve()}")
        except webbrowser.Error as e:
             print(f"  [!] Error using webbrowser module: {e}")
             print(f"      Try opening manually: {filepath.resolve()}")
        except Exception as e:
            print(f"  [!] Unexpected error using webbrowser: {e}")
            print(f"      Try opening manually: {filepath.resolve()}")

# =====================================
# Main Execution Logic
# =====================================

def main() -> int:
    """
    Main function to orchestrate the Lorenz system simulation and analysis.

    Returns:
        0 if successful, 1 otherwise.
    """
    print("=" * 60)
    print(" Lorenz System Simulation, Analysis, and Visualization")
    print("=" * 60)

    try:
        output_path_obj = pathlib.Path(OUTPUT_DIR_NAME)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Output directory: {output_path_obj.resolve()}")
    except OSError as e:
        print(f"[!] Fatal Error: Cannot create output directory '{OUTPUT_DIR_NAME}': {e}")
        return 1

    # Construct full paths for output files
    animation_path = output_path_obj / VIDEO_ANIMATION_FILENAME
    interactive_html_path = output_path_obj / INTERACTIVE_HTML_FILENAME # Added
    static_plot_path = output_path_obj / STATIC_PLOT_FILENAME
    lle_results_path = output_path_obj / LLE_RESULTS_FILENAME

    # --- System Parameters and Time Spans ---
    lorenz_params: LorenzParams = (LORENZ_PARAM_SIGMA, LORENZ_PARAM_RHO, LORENZ_PARAM_BETA)
    t_span_vis = np.linspace(0.0, VIS_T_END, VIS_NUM_STEPS)
    t_span_lle = np.linspace(0.0, LLE_T_END, LLE_NUM_STEPS)
    print(f"[*] System Parameters: σ={lorenz_params[0]:.1f}, ρ={lorenz_params[1]:.1f}, β={lorenz_params[2]:.2f}")
    print(f"[*] Visualization Time: 0.0 to {VIS_T_END} ({VIS_NUM_STEPS} steps)")
    print(f"[*] LLE Calculation Time: 0.0 to {LLE_T_END} ({LLE_NUM_STEPS} steps)")

    # --- Solve for Visualization ---
    try:
        solutions = solve_visual_trajectories(INITIAL_STATES, t_span_vis, lorenz_system, lorenz_params)
    except Exception as e:
        print(f"[!] Fatal Error during ODE solving for visualization: {e}")
        return 1

    if not solutions:
        print("[!] Aborting: No trajectories could be solved for visualization.")
        return 1
    first_solution = solutions[0]

    # --- Calculate LLE ---
    lle_estimate, lle_interpretation, lle_sim_time = calculate_lle(
        INITIAL_STATES[0], t_span_lle, lorenz_system, lorenz_params, LLE_EPSILON
    )

    # --- Save LLE Results ---
    save_lle_results(
        lle_results_path, lorenz_params,
        lle_sim_time, LLE_T_END, len(t_span_lle), LLE_EPSILON, lle_interpretation
    )

    # --- Generate and Save Static Plots ---
    generate_save_static_plots(
        static_plot_path, solutions, t_span_vis, INITIAL_STATES,
        lorenz_params, lle_estimate, STATIC_PLOT_DPI
    )

    # --- Generate and Save Animation Video ---
    # Note: TAIL_LENGTH constant is interpreted as *points* here
    if ANIMATION_WRITER: # Check if writer is available
        animation_saved = generate_save_animation(
            animation_path, first_solution, t_span_vis, FRAME_STEP, TAIL_LENGTH,
            ANIMATION_INTERVAL, ANIMATION_WRITER, ANIMATION_FPS, ANIMATION_DPI
        )
    else:
        print("\n[*] Skipping animation saving (no valid writer).")
        animation_saved = False # Ensure variable exists


    # --- Generate and Save Interactive HTML Plot --- (Added)
    html_saved = generate_save_interactive_html(
        interactive_html_path, first_solution, t_span_vis, lorenz_params
    )

    # --- Attempt to Open INTERACTIVE HTML (Cross-Platform) --- (Modified target)
    if html_saved and interactive_html_path.exists():
        open_file_cross_platform(interactive_html_path)
    elif not html_saved:
        print("\n[*] Skipping file opening: Interactive HTML generation failed.")
    else:
        print("\n[*] Skipping file opening: HTML file not found after reported success.")


    print("\n[+] Lorenz script finished successfully.")
    print("=" * 60)
    return 0


# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
