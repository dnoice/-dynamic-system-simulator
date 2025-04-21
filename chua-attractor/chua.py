# -*- coding: utf-8 -*-
"""
Chua's Circuit Simulator, Analyzer, and Visualizer (PEP 8 Compliant)

This script simulates Chua's Circuit, a simple electronic circuit famous for
exhibiting a wide range of nonlinear dynamics, including chaotic behavior often
visualized as a "double-scroll" attractor.

It performs the following actions:
1.  **Defines Chua's System:** Implements the system of three ordinary
    differential equations describing the circuit, including the piecewise-linear
    characteristic of Chua's diode.
2.  **Solves ODEs:** Integrates the Chua system equations over time using
    SciPy's odeint for multiple initial conditions.
3.  **Calculates LLE:** Estimates the Largest Lyapunov Exponent (LLE) to quantify
    the system's sensitivity to initial conditions (chaos).
4.  **Generates Static Plots:** Creates a multi-panel figure (Matplotlib) showing
    the 3D attractor and its 2D projections (XY, XZ, YZ).
5.  **Generates Animation:** Produces an MP4 video animation of the attractor's
    trajectory (requires FFmpeg).
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
    - ffmpeg (External command-line tool, optional for video generation)
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
from mpl_toolkits.mplot3d import Axes3D
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
OUTPUT_DIR_NAME: str = "chua_output"
VIDEO_ANIMATION_FILENAME: str = "chua_animation.mp4"
INTERACTIVE_HTML_FILENAME: str = "chua_interactive.html"
STATIC_PLOT_FILENAME: str = "chua_static_plots.png"
LLE_RESULTS_FILENAME: str = "chua_lle_results.txt"

# --- Chua's System Parameters (Classic Double-Scroll Regime) ---
# These parameters correspond to the voltages across capacitors (x, y)
# and the current through the inductor (z).
CHUA_PARAM_ALPHA: float = 9.0        # Scales time, related to C2/G
CHUA_PARAM_BETA: float = 100.0 / 7.0  # Related to C2*R^2/L (~14.286)
CHUA_PARAM_M0: float = -1.0 / 7.0     # Inner slope of Chua's diode * R (~-0.143)
CHUA_PARAM_M1: float = 2.0 / 7.0      # Outer slope of Chua's diode * R (~0.286)
CHUA_PARAM_E: float = 1.0           # Breakpoint voltage (often normalized to 1)

# --- Initial States (Near origin, typical for observing double-scroll) ---
INITIAL_STATES: List[List[float]] = [
    [0.7, 0.0, 0.0], # A common starting point to generate the double scroll
    [0.71, 0.0, 0.0],
    [0.7, 0.01, 0.0],
]

# --- Simulation Time Spans ---
VIS_T_END: float = 100.0       # End time for visualization
VIS_NUM_STEPS: int = 20000     # Number of steps for visualization
LLE_T_END: float = 250.0       # End time for LLE calculation (needs longer integration)
LLE_NUM_STEPS: int = 50000     # Number of steps for LLE calculation

# --- LLE Calculation Settings ---
LLE_EPSILON: float = 1e-9      # Initial separation for LLE

# --- Animation Settings ---
# Check if ffmpeg writer is available
DEFAULT_ANIMATION_WRITER: str = "ffmpeg"
if not writers.is_available(DEFAULT_ANIMATION_WRITER):
    print(f"[!] Warning: Matplotlib writer '{DEFAULT_ANIMATION_WRITER}' not found.")
    print("    Animation saving might fail. Consider installing FFmpeg.")
    ANIMATION_WRITER: Optional[str] = None # Disable animation saving
else:
    ANIMATION_WRITER: Optional[str] = DEFAULT_ANIMATION_WRITER

ANIMATION_FPS: int = 30        # Frames per second
ANIMATION_DPI: int = 100       # Video frame resolution
FRAME_STEP: int = 15           # Use every Nth point for animation (controls speed)
TAIL_LENGTH: int = 250         # Tail length in number of *frames*
ANIMATION_INTERVAL: int = 30   # Delay between frames (ms)

# --- Visualization Settings ---
STATIC_PLOT_DPI: int = 150     # Static plot resolution

# Type alias for state vector
State = np.ndarray
# Type alias for system parameters tuple (alpha, beta, m0, m1, [E - optional])
ChuaParams = Tuple[float, float, float, float]
ChuaParamsWithE = Tuple[float, float, float, float, float]

# =====================================
# Function Definitions
# =====================================

def chua_nonlinear_resistor(x: float, m0: float, m1: float, E: float = 1.0) -> float:
    """
    Calculates the output of the piecewise-linear Chua's diode function f(x).

    f(x) = m0*x                  if |x| <= E
         = m1*x + (m0-m1)*E      if x > E
         = m1*x - (m0-m1)*E      if x < -E

    Args:
        x: Input value (voltage V_C1).
        m0: Slope in the inner region.
        m1: Slope in the outer regions.
        E: Breakpoint voltage (default: 1.0).

    Returns:
        The value of f(x).
    """
    # Standard conditional implementation (more readable)
    if abs(x) <= E:
        return m0 * x
    elif x > E:
        return m1 * x + (m0 - m1) * E
    else: # x < -E
        return m1 * x - (m0 - m1) * E

    # Alternative compact implementation (mathematically equivalent)
    # return m1 * x + 0.5 * (m0 - m1) * (abs(x + E) - abs(x - E))


def chua_system(state: State, t: float, alpha: float, beta: float, m0: float, m1: float, E: float = CHUA_PARAM_E) -> State:
    """
    Defines the differential equations for Chua's Circuit.

    Args:
        state: Current state vector [x, y, z] = [V_C1, V_C2, I_L*R].
        t: Current time t (required by odeint).
        alpha, beta, m0, m1: Chua system parameters.
        E: Breakpoint voltage for the nonlinear resistor (default uses global const).

    Returns:
        np.ndarray: Array containing the derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    f_x = chua_nonlinear_resistor(x, m0, m1, E)

    # Chua's equations
    dxdt = alpha * (y - x - f_x)
    dydt = x - y + z
    dzdt = -beta * y
    return np.array([dxdt, dydt, dzdt])

# --- solve_visual_trajectories (Generic ODE solver) ---
def solve_visual_trajectories(
    initial_states: List[List[float]],
    t_span: np.ndarray,
    system_func: Callable, # More generic type hint
    sys_params: Tuple,      # Parameters tuple for system_func
) -> List[np.ndarray]:
    """
    Solves the ODE system for visualization for multiple initial states.

    Args:
        initial_states: A list of initial state vectors [[x0, y0, z0], ...].
        t_span: Array of time points for the solution.
        system_func: The function defining the ODE system (e.g., chua_system).
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
            # Note: odeint passes parameters as a tuple via 'args'
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
    system_func: Callable, # More generic type hint
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
                # Rescale perturbation along the difference vector
                state1 = state0 + (diff_vector / distance) * epsilon
            else:
                print(f"  [!] Warning: LLE trajectories separation near zero at step {i}. Re-perturbing.")
                state1 = state0 + perturb_vector # Re-apply initial perturbation

            current_t += dt_lle
            if (i + 1) % (max(1, total_steps_lle // 20)) == 0:
                progress = 100 * (i + 1) / total_steps_lle
                print(f"    LLE Progress: {progress:.1f}%", end='\r')

        print("\n  [+] LLE Calculation loop finished.")

    except Exception as e:
        print(f"\n  [!] An error occurred during LLE calculation: {e}")
        return None, f"LLE Calculation Failed (Runtime Error: {e}).", current_t

    lle_estimate: Optional[float] = None
    lle_interpretation: str = "LLE Calculation Failed (Unknown error after loop)."

    if current_t > 1e-9: # Ensure simulation ran for a meaningful duration
        lle_estimate = lle_sum / current_t
        print(f"  [+] LLE Calculation Complete. Total time simulated: {current_t:.2f} units.")
        result_text = f"Estimated LLE: {lle_estimate:.4f}\n"
        if lle_estimate > 0.01:
            result_text += "  Interpretation: Positive LLE suggests chaotic behavior."
        elif lle_estimate < -0.01:
            result_text += "  Interpretation: Negative LLE suggests convergence to stable point/cycle."
        else:
            result_text += "  Interpretation: LLE near zero suggests periodic or quasi-periodic behavior."
        lle_interpretation = result_text
        print(f"\n{lle_interpretation}")
    else:
        lle_interpretation = "LLE Calculation Failed (simulation time near zero)."
        print(f"  [!] {lle_interpretation}")

    return lle_estimate, lle_interpretation, current_t


# --- save_lle_results (Adapted for Chua parameters) ---
def save_lle_results(
    filepath: pathlib.Path,
    params: ChuaParams, # Takes the 4 main Chua parameters
    sim_time: float,
    target_time: float,
    num_steps: int,
    epsilon: float,
    interpretation: str,
) -> None:
    """
    Saves LLE calculation results and Chua parameters to a text file.

    Args:
        filepath: Path object for the output text file.
        params: Tuple of Chua parameters (alpha, beta, m0, m1).
        sim_time: Actual simulation time achieved.
        target_time: Target simulation time for LLE.
        num_steps: Number of integration steps used.
        epsilon: Initial separation distance used.
        interpretation: String summarizing the LLE result.
    """
    print(f"[*] Saving LLE results to: {filepath}")
    param_alpha, param_beta, param_m0, param_m1 = params
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("=" * 30 + "\n")
            f.write(" Chua's Circuit Analysis Results\n")
            f.write("=" * 30 + "\n\n")
            f.write("System Parameters:\n")
            f.write(f"  alpha = {param_alpha:.4f}\n")
            f.write(f"  beta  = {param_beta:.4f}\n")
            f.write(f"  m0    = {param_m0:.4f}\n")
            f.write(f"  m1    = {param_m1:.4f}\n")
            # E is implicitly CHUA_PARAM_E (assumed 1.0 if not specified)
            f.write(f"  E     = {CHUA_PARAM_E:.4f} (Implicit breakpoint)\n\n")
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

# --- generate_save_static_plots (Adapted for Chua) ---
def generate_save_static_plots(
    filepath: pathlib.Path,
    solutions: List[np.ndarray],
    t_span: np.ndarray,
    initial_states: List[List[float]],
    params: ChuaParams, # Takes the 4 main Chua parameters
    lle_est: Optional[float],
    dpi: int,
) -> None:
    """
    Generates and saves static plots (3D and 2D projections) of Chua's attractor.

    Args:
        filepath: Path object for the output PNG file.
        solutions: List of trajectory arrays.
        t_span: Time array corresponding to the solutions.
        initial_states: List of starting points used.
        params: Tuple of Chua parameters (alpha, beta, m0, m1).
        lle_est: Estimated LLE value (or None).
        dpi: Dots per inch for the saved image.
    """
    if not solutions:
        print("[!] Skipping static plot generation: No solution data available.")
        return

    print(f"\n[*] Generating and saving static plots to: {filepath.name}")
    param_alpha, param_beta, param_m0, param_m1 = params
    fig_static = plt.figure(figsize=(14, 10), dpi=dpi)

    # --- Title ---
    title_lle_part = f" | LLE ≈ {lle_est:.4f}" if lle_est is not None else ""
    title = (
        f"Chua's Circuit (α={param_alpha:.2f}, β={param_beta:.2f}, "
        f"m0={param_m0:.2f}, m1={param_m1:.2f}){title_lle_part}"
    )
    fig_static.suptitle(title, fontsize=16)

    # --- 3D Plot ---
    ax3d = fig_static.add_subplot(2, 2, 1, projection="3d")
    num_trajectories = len(solutions)
    all_x = np.concatenate([s[:, 0] for s in solutions])
    all_y = np.concatenate([s[:, 1] for s in solutions])
    all_z = np.concatenate([s[:, 2] for s in solutions])

    cmap_3d = 'plasma' # Colormap suitable for Chua
    linewidth_3d = 0.6
    alpha_3d = 0.7

    for i, solution in enumerate(solutions):
        x_sol, y_sol, z_sol = solution[:, 0], solution[:, 1], solution[:, 2]
        points = np.array([x_sol, y_sol, z_sol]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_span.min(), t_span.max())
        lc = Line3DCollection(
            segments, cmap=cmap_3d, norm=norm, linewidth=linewidth_3d, alpha=alpha_3d
        )
        lc.set_array(t_span[:-1])
        ax3d.add_collection(lc)

    ax3d.set_xlabel("X ($V_{C1}$)") # Use LaTeX for variable names
    ax3d.set_ylabel("Y ($V_{C2}$)")
    ax3d.set_zlabel("Z ($I_L R$)") # Assuming z is scaled current
    ax3d.set_title(f"3D View ({num_trajectories} Trajectories)")
    ax3d.grid(True, linestyle='--', alpha=0.5)

    starts = np.array(initial_states)
    ax3d.scatter(
        starts[:, 0], starts[:, 1], starts[:, 2],
        color='lime', s=70, marker='o', edgecolor='black',
        label='Start Points', depthshade=False, zorder=10
    )
    ax3d.legend(loc='upper left')

    pad_x = (all_x.max() - all_x.min()) * 0.05
    pad_y = (all_y.max() - all_y.min()) * 0.05
    pad_z = (all_z.max() - all_z.min()) * 0.05
    ax3d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax3d.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax3d.set_zlim(all_z.min() - pad_z, all_z.max() + pad_z)

    # View angle optimized for the double-scroll structure
    ax3d.view_init(elev=20, azim=-60)

    # --- 2D Projections (using the first trajectory) ---
    x_p, y_p, z_p = solutions[0][:, 0], solutions[0][:, 1], solutions[0][:, 2]
    scatter_args = {'c': t_span, 'cmap': 'magma', 's': 0.5, 'alpha': 0.6} # Different cmap

    ax_xy = fig_static.add_subplot(2, 2, 2)
    ax_xy.scatter(x_p, y_p, **scatter_args)
    ax_xy.set_xlabel("X ($V_{C1}$)")
    ax_xy.set_ylabel("Y ($V_{C2}$)")
    ax_xy.set_title("X-Y Projection")
    ax_xy.grid(True, linestyle=':', alpha=0.6)
    ax_xy.set_aspect('equal', adjustable='box')

    ax_xz = fig_static.add_subplot(2, 2, 3)
    ax_xz.scatter(x_p, z_p, **scatter_args)
    ax_xz.set_xlabel("X ($V_{C1}$)")
    ax_xz.set_ylabel("Z ($I_L R$)")
    ax_xz.set_title("X-Z Projection")
    ax_xz.grid(True, linestyle=':', alpha=0.6)
    ax_xz.set_aspect('equal', adjustable='box')

    ax_yz = fig_static.add_subplot(2, 2, 4)
    ax_yz.scatter(y_p, z_p, **scatter_args)
    ax_yz.set_xlabel("Y ($V_{C2}$)")
    ax_yz.set_ylabel("Z ($I_L R$)")
    ax_yz.set_title("Y-Z Projection")
    ax_yz.grid(True, linestyle=':', alpha=0.6)
    ax_yz.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        fig_static.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  [+] Static plots saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not save static plots: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred while saving static plots: {e}")
    finally:
        plt.close(fig_static) # Ensure figure closure


# --- generate_save_animation (Adapted for Chua) ---
def generate_save_animation(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    frame_step: int,
    tail_len_frames: int,
    interval_ms: int,
    writer_name: Optional[str],
    fps: int,
    dpi: int,
) -> bool:
    """
    Generates and saves an MP4 animation of Chua's attractor trajectory.

    Args:
        filepath: Path object for the output MP4 file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        frame_step: Step size through data for each animation frame.
        tail_len_frames: Length of trajectory tail in number of frames.
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

    print(f"\n[*] Generating animation video (using '{writer_name}' writer)...")
    x_anim, y_anim, z_anim = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]
    num_points = len(t_span)
    num_frames = num_points // frame_step

    if num_frames < 2:
        print("  [!] Error: Not enough data points for animation.")
        return False

    fig_anim = plt.figure(figsize=(10, 8), dpi=dpi)
    ax_anim = fig_anim.add_subplot(111, projection='3d')

    line_anim, = ax_anim.plot([], [], [], lw=1.5, color='cyan') # Different color
    point_anim, = ax_anim.plot([], [], [], 'o', color='magenta', markersize=6, zorder=10)

    ax_anim.set_xlim(x_anim.min(), x_anim.max())
    ax_anim.set_ylim(y_anim.min(), y_anim.max())
    ax_anim.set_zlim(z_anim.min(), z_anim.max())
    ax_anim.set_xlabel("X ($V_{C1}$)")
    ax_anim.set_ylabel("Y ($V_{C2}$)")
    ax_anim.set_zlabel("Z ($I_L R$)")
    ax_anim.set_title("Chua Attractor Animation")
    ax_anim.grid(True, linestyle=':', alpha=0.4)
    ax_anim.view_init(elev=20, azim=-60) # Consistent view angle

    def update(frame_idx_scaled: int) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
        """Updates the line and point data for each frame."""
        current_data_idx = frame_idx_scaled * frame_step
        tail_start_data_idx = max(0, current_data_idx - (tail_len_frames * frame_step))

        line_anim.set_data(x_anim[tail_start_data_idx : current_data_idx+1 : frame_step],
                           y_anim[tail_start_data_idx : current_data_idx+1 : frame_step])
        line_anim.set_3d_properties(z_anim[tail_start_data_idx : current_data_idx+1 : frame_step])

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
            if n > 0 and ((i + 1) % 20 == 0 or (i + 1) == n):
                print(f"    Saving frame {i+1}/{n} ({100*(i+1)/n:.1f}%)", end='\r')

        anim.save(filepath, writer=writer_name, fps=fps, dpi=dpi, progress_callback=progress_update)
        print("\n  [+] Animation video saved successfully.")
        animation_saved = True
    except FileNotFoundError:
        print(f"\n  [!] Error: Animation writer '{writer_name}' command not found or failed.")
        print(f"      Ensure '{writer_name}' (e.g., FFmpeg) is installed and in the system PATH.")
    except Exception as e:
        print(f"\n  [!] Error occurred while saving animation: {e}")
    finally:
        print(" " * 80, end='\r') # Clear progress line
        plt.close(fig_anim)

    return animation_saved

# --- generate_save_interactive_html (Adapted for Chua) ---
def generate_save_interactive_html(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    params: ChuaParams, # Takes the 4 main Chua parameters
) -> bool:
    """
    Generates and saves an interactive 3D plot of Chua's attractor using Plotly.

    Args:
        filepath: Path object for the output HTML file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        params: Tuple of Chua parameters (alpha, beta, m0, m1).

    Returns:
        True if HTML was saved successfully, False otherwise.
    """
    if solution_data.size == 0:
         print("\n[*] Skipping interactive HTML generation: No solution data available.")
         return False

    print(f"\n[*] Generating interactive HTML plot to: {filepath.name}")
    param_alpha, param_beta, param_m0, param_m1 = params
    x_trace, y_trace, z_trace = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]

    try:
        trace = go.Scatter3d(
            x=x_trace, y=y_trace, z=z_trace, mode='lines',
            line=dict(color=t_span, colorscale='plasma', width=3, # Use Chua cmap
                      colorbar=dict(title='Time')),
            name='Chua Trajectory'
        )

        layout = go.Layout(
            title=dict(
                text=(f"Interactive Chua Attractor<br>"
                      f"(α={param_alpha:.2f}, β={param_beta:.2f}, "
                      f"m0={param_m0:.2f}, m1={param_m1:.2f})"),
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (V_C1)', yaxis_title='Y (V_C2)', zaxis_title='Z (I_L R)',
                xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                # Adjust aspect ratio and camera for Chua's double-scroll shape
                aspectratio=dict(x=1, y=1, z=0.6),
                camera_eye=dict(x=1.8, y=-1.8, z=0.8)
            ),
            margin=dict(l=10, r=10, b=10, t=50),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(filepath, include_plotlyjs='cdn') # Use CDN for smaller file size

        print(f"  [+] Interactive HTML saved successfully to {filepath.name}")
        return True

    except ImportError:
        print("  [!] Error: Plotly library is not installed (pip install plotly).")
        return False
    except Exception as e:
        print(f"  [!] An unexpected error occurred generating interactive HTML: {e}")
        return False

# --- open_file_cross_platform (Copied from previous refinement) ---
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
    Main function to orchestrate the Chua's Circuit simulation and analysis.

    Returns:
        0 if successful, 1 otherwise.
    """
    print("=" * 60)
    print(" Chua's Circuit Simulation, Analysis, and Visualization")
    print("=" * 60)

    try:
        output_path_obj = pathlib.Path(OUTPUT_DIR_NAME)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Output directory: {output_path_obj.resolve()}")
    except OSError as e:
        print(f"[!] Fatal Error: Cannot create output directory '{OUTPUT_DIR_NAME}': {e}")
        return 1

    video_animation_path = output_path_obj / VIDEO_ANIMATION_FILENAME
    interactive_html_path = output_path_obj / INTERACTIVE_HTML_FILENAME
    static_plot_path = output_path_obj / STATIC_PLOT_FILENAME
    lle_results_path = output_path_obj / LLE_RESULTS_FILENAME

    chua_params: ChuaParams = (CHUA_PARAM_ALPHA, CHUA_PARAM_BETA, CHUA_PARAM_M0, CHUA_PARAM_M1)
    # Note: E is passed implicitly via default in chua_system or can be added
    # to chua_params if explicitly needed by a modified system_func.
    t_span_vis = np.linspace(0.0, VIS_T_END, VIS_NUM_STEPS)
    t_span_lle = np.linspace(0.0, LLE_T_END, LLE_NUM_STEPS)
    print(f"[*] System Parameters: α={chua_params[0]:.2f}, β={chua_params[1]:.2f}, m0={chua_params[2]:.2f}, m1={chua_params[3]:.2f}, E={CHUA_PARAM_E:.1f}")
    print(f"[*] Visualization Time: 0.0 to {VIS_T_END} ({VIS_NUM_STEPS} steps)")
    print(f"[*] LLE Calculation Time: 0.0 to {LLE_T_END} ({LLE_NUM_STEPS} steps)")

    # --- Solve for Visualization ---
    try:
        solutions = solve_visual_trajectories(INITIAL_STATES, t_span_vis, chua_system, chua_params)
    except Exception as e:
        print(f"[!] Fatal Error during ODE solving for visualization: {e}")
        return 1

    if not solutions:
        print("[!] Aborting: No trajectories could be solved for visualization.")
        return 1
    first_solution = solutions[0]

    # --- Calculate LLE ---
    lle_estimate, lle_interpretation, lle_sim_time = calculate_lle(
        INITIAL_STATES[0], t_span_lle, chua_system, chua_params, LLE_EPSILON
    )

    # --- Save LLE Results ---
    save_lle_results(
        lle_results_path, chua_params,
        lle_sim_time, LLE_T_END, len(t_span_lle), LLE_EPSILON, lle_interpretation
    )

    # --- Generate and Save Static Plots ---
    generate_save_static_plots(
        static_plot_path, solutions, t_span_vis, INITIAL_STATES,
        chua_params, lle_estimate, STATIC_PLOT_DPI
    )

    # --- Generate and Save Animation Video ---
    if ANIMATION_WRITER: # Check if writer is available
        generate_save_animation(
            video_animation_path, first_solution, t_span_vis, FRAME_STEP, TAIL_LENGTH,
            ANIMATION_INTERVAL, ANIMATION_WRITER, ANIMATION_FPS, ANIMATION_DPI
        )
    else:
        print("\n[*] Skipping animation saving (no valid writer).")

    # --- Generate and Save Interactive HTML Plot ---
    html_saved = generate_save_interactive_html(
        interactive_html_path, first_solution, t_span_vis, chua_params
    )

    # --- Attempt to Open INTERACTIVE HTML (Cross-Platform) ---
    if html_saved and interactive_html_path.exists():
        open_file_cross_platform(interactive_html_path)
    elif not html_saved:
        print("\n[*] Skipping file opening: Interactive HTML generation failed.")
    else:
        print("\n[*] Skipping file opening: HTML file not found after reported success.")

    print("\n[+] Chua script finished successfully.")
    print("=" * 60)
    return 0


# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
