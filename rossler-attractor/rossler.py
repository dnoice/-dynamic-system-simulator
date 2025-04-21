# -*- coding: utf-8 -*-
"""
Rössler System Simulator, Analyzer, and Visualizer (PEP 8 Compliant)

This script simulates the Rössler system, another classic example of a dynamical
system exhibiting chaotic behavior, characterized by a distinct folded-band attractor.

It performs the following actions:
1.  **Defines Rössler System:** Implements the system of three ODEs.
2.  **Solves ODEs:** Integrates the Rössler equations using SciPy's odeint.
3.  **Calculates LLE:** Estimates the Largest Lyapunov Exponent (LLE).
4.  **Generates Static Plots:** Creates Matplotlib plots (3D attractor, 2D projections).
5.  **Generates Animation:** Produces an MP4 video animation (requires FFmpeg).
6.  **Generates Interactive HTML:** Creates an interactive 3D Plotly plot.
7.  **Saves Results:** Stores LLE results and outputs in a dedicated directory.
8.  **Opens Output:** Attempts to automatically open the generated HTML file.

Dependencies:
    - Python 3.x
    - numpy
    - scipy
    - matplotlib
    - plotly
    - ffmpeg (External command-line tool, optional for video generation)

Usage:
  Run from the command line: python <script_name>.py
  Outputs saved in 'rossler_output' directory.
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
from mpl_toolkits.mplot3d import Axes3D # Explicitly needed for projection='3d'
from scipy.integrate import odeint

# --- Backend Configuration ---
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
OUTPUT_DIR_NAME: str = "rossler_output"
VIDEO_ANIMATION_FILENAME: str = "rossler_animation.mp4"
INTERACTIVE_HTML_FILENAME: str = "rossler_interactive.html"
STATIC_PLOT_FILENAME: str = "rossler_static_plots.png"
LLE_RESULTS_FILENAME: str = "rossler_lle_results.txt"

# --- Rössler System Parameters (Classic Chaotic Regime) ---
ROSSLER_PARAM_A: float = 0.2
ROSSLER_PARAM_B: float = 0.2
ROSSLER_PARAM_C: float = 5.7

# --- Initial States ---
INITIAL_STATES: List[List[float]] = [
    [-1.0, 0.0, 0.0],    # A common starting point
    [-1.01, 0.0, 0.0],
    [-1.0, 0.01, 0.0],
]

# --- Simulation Time Spans ---
VIS_T_END: float = 100.0       # End time for visualization
VIS_NUM_STEPS: int = 20000     # Number of steps for visualization
LLE_T_END: float = 250.0       # End time for LLE calculation
LLE_NUM_STEPS: int = 50000     # Number of steps for LLE calculation

# --- LLE Calculation Settings ---
LLE_EPSILON: float = 1e-9      # Initial separation for LLE

# --- Animation Settings ---
DEFAULT_ANIMATION_WRITER: str = "ffmpeg"
if not writers.is_available(DEFAULT_ANIMATION_WRITER):
    print(f"[!] Warning: Matplotlib writer '{DEFAULT_ANIMATION_WRITER}' not found.")
    print("    Animation saving might fail. Install FFmpeg.")
    ANIMATION_WRITER: Optional[str] = None
else:
    ANIMATION_WRITER: Optional[str] = DEFAULT_ANIMATION_WRITER

ANIMATION_FPS: int = 30
ANIMATION_DPI: int = 100
FRAME_STEP: int = 15           # Plot every Nth point
TAIL_LENGTH: int = 250         # Tail length in number of *points*
ANIMATION_INTERVAL: int = 30   # Delay between frames (ms)

# --- Visualization Settings ---
STATIC_PLOT_DPI: int = 150

# Type alias for state vector
State = np.ndarray
# Type alias for system parameters tuple (a, b, c)
RosslerParams = Tuple[float, float, float]

# =====================================
# Function Definitions
# =====================================

def rossler_system(state: State, t: float, a: float, b: float, c: float) -> State:
    """
    Defines the differential equations for the Rössler system.

    Args:
        state: Current state vector [x, y, z].
        t: Current time t (required by odeint).
        a, b, c: Rössler system parameters.

    Returns:
        np.ndarray: Array containing the derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return np.array([dxdt, dydt, dzdt])

# --- solve_visual_trajectories (Generic ODE solver) ---
def solve_visual_trajectories(
    initial_states: List[List[float]],
    t_span: np.ndarray,
    system_func: Callable,
    sys_params: Tuple,
) -> List[np.ndarray]:
    """
    Solves the ODE system for visualization for multiple initial states.

    Args:
        initial_states: List of initial state vectors [[x0, y0, z0], ...].
        t_span: Array of time points for the solution.
        system_func: The function defining the ODE system (e.g., rossler_system).
        sys_params: Tuple of system parameters required by system_func.

    Returns:
        List of numpy arrays, each holding a trajectory [x(t), y(t), z(t)].
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
            print("      Skipping this trajectory.")
    if solutions:
        print("  [+] Visualization integration complete.")
    else:
        print("  [!] No trajectories were successfully integrated.")
    return solutions

# --- calculate_lle (Generic LLE calculator) ---
def calculate_lle(
    initial_state: List[float],
    t_span_lle: np.ndarray,
    system_func: Callable,
    sys_params: Tuple,
    epsilon: float,
) -> Tuple[Optional[float], str, float]:
    """
    Estimates the Largest Lyapunov Exponent (LLE) for a given dynamical system.

    Args:
        initial_state: Starting state vector [x0, y0, z0].
        t_span_lle: Array of time points for LLE calculation.
        system_func: Function defining the ODE system.
        sys_params: Tuple of system parameters.
        epsilon: Initial small separation distance.

    Returns:
        Tuple containing: Estimated LLE (float or None), interpretation string, actual simulation time (float).
    """
    print("\n[*] Calculating Largest Lyapunov Exponent (LLE)...")
    if len(t_span_lle) < 2:
        print("  [!] Error: LLE time span needs >= 2 points.")
        return None, "LLE Calculation Failed (invalid time span).", 0.0

    dt_lle = t_span_lle[1] - t_span_lle[0]
    total_steps_lle = len(t_span_lle)
    lle_sum = 0.0
    state0 = np.array(initial_state, dtype=float)
    perturb_vector = np.zeros_like(state0); perturb_vector[0] = epsilon
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

            state0 = sol0[1]; state1 = sol1[1]
            diff_vector = state1 - state0; distance = np.linalg.norm(diff_vector)

            if distance > 1e-15:
                lle_sum += np.log(distance / epsilon)
                state1 = state0 + (diff_vector / distance) * epsilon # Rescale
            else:
                print(f"  [!] Warning: LLE separation near zero at step {i}. Re-perturbing.")
                state1 = state0 + perturb_vector

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

    if current_t > 1e-9:
        lle_estimate = lle_sum / current_t
        print(f"  [+] LLE Calculation Complete. Time: {current_t:.2f} units.")
        result_text = f"Estimated LLE: {lle_estimate:.4f}\n"
        if lle_estimate > 0.01:
            result_text += "  Interpretation: Positive LLE -> chaotic behavior."
        elif lle_estimate < -0.01:
            result_text += "  Interpretation: Negative LLE -> stable point/cycle."
        else:
            result_text += "  Interpretation: LLE near zero -> periodic/quasi-periodic."
        lle_interpretation = result_text
        print(f"\n{lle_interpretation}")
    else:
        lle_interpretation = "LLE Calculation Failed (simulation time near zero)."
        print(f"  [!] {lle_interpretation}")

    return lle_estimate, lle_interpretation, current_t

# --- save_lle_results (Adapted for Rössler parameters) ---
def save_lle_results(
    filepath: pathlib.Path,
    params: RosslerParams, # Takes the (a, b, c) tuple
    sim_time: float,
    target_time: float,
    num_steps: int,
    epsilon: float,
    interpretation: str,
) -> None:
    """
    Saves LLE results and Rössler parameters to a text file.

    Args:
        filepath: Path object for the output text file.
        params: Tuple of Rössler parameters (a, b, c).
        sim_time: Actual simulation time achieved.
        target_time: Target simulation time for LLE.
        num_steps: Number of integration steps used.
        epsilon: Initial separation distance used.
        interpretation: String summarizing the LLE result.
    """
    print(f"[*] Saving LLE results to: {filepath}")
    a, b, c = params
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("=" * 30 + "\n")
            f.write(" Rössler System Analysis Results\n")
            f.write("=" * 30 + "\n\n")
            f.write("System Parameters:\n")
            f.write(f"  a = {a:.4f}\n")
            f.write(f"  b = {b:.4f}\n")
            f.write(f"  c = {c:.4f}\n\n")
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
        print(f"  [!] An unexpected error occurred saving LLE results: {e}")

# --- generate_save_static_plots (Adapted for Rössler) ---
def generate_save_static_plots(
    filepath: pathlib.Path,
    solutions: List[np.ndarray],
    t_span: np.ndarray,
    initial_states: List[List[float]],
    params: RosslerParams, # Takes the (a, b, c) tuple
    lle_est: Optional[float],
    dpi: int,
) -> None:
    """
    Generates and saves static plots (3D and 2D projections) of the Rössler attractor.

    Args:
        filepath: Path object for the output PNG file.
        solutions: List of trajectory arrays.
        t_span: Time array corresponding to solutions.
        initial_states: List of starting points used.
        params: Tuple of Rössler parameters (a, b, c).
        lle_est: Estimated LLE value (or None).
        dpi: Dots per inch for the saved image.
    """
    if not solutions:
        print("[!] Skipping static plot generation: No solution data.")
        return

    print(f"\n[*] Generating and saving static plots to: {filepath.name}")
    a, b, c = params
    fig_static = plt.figure(figsize=(14, 10), dpi=dpi)

    title_lle_part = f" | LLE ≈ {lle_est:.4f}" if lle_est is not None else ""
    title = f"Rössler System (a={a:.2f}, b={b:.2f}, c={c:.2f}){title_lle_part}"
    fig_static.suptitle(title, fontsize=16)

    ax3d = fig_static.add_subplot(2, 2, 1, projection="3d")
    num_trajectories = len(solutions)
    all_x = np.concatenate([s[:, 0] for s in solutions])
    all_y = np.concatenate([s[:, 1] for s in solutions])
    all_z = np.concatenate([s[:, 2] for s in solutions])

    cmap_3d = 'viridis'
    linewidth_3d = 0.6; alpha_3d = 0.7

    for i, solution in enumerate(solutions):
        x_sol, y_sol, z_sol = solution[:, 0], solution[:, 1], solution[:, 2]
        points = np.array([x_sol, y_sol, z_sol]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_span.min(), t_span.max())
        lc = Line3DCollection(
            segments, cmap=cmap_3d, norm=norm, linewidth=linewidth_3d, alpha=alpha_3d
        )
        lc.set_array(t_span[:-1]); ax3d.add_collection(lc)

    ax3d.set_xlabel("X Axis"); ax3d.set_ylabel("Y Axis"); ax3d.set_zlabel("Z Axis")
    ax3d.set_title(f"3D View ({num_trajectories} Trajectories)")
    ax3d.grid(True, linestyle='--', alpha=0.5)

    starts = np.array(initial_states)
    ax3d.scatter(
        starts[:, 0], starts[:, 1], starts[:, 2], color='red', s=70,
        marker='o', edgecolor='black', label='Start Points', depthshade=False, zorder=10
    )
    ax3d.legend(loc='upper left')

    pad = 0.05
    ax3d.set_xlim(all_x.min()*(1-pad), all_x.max()*(1+pad))
    ax3d.set_ylim(all_y.min()*(1-pad), all_y.max()*(1+pad))
    ax3d.set_zlim(all_z.min()*(1-pad), all_z.max()*(1+pad))

    ax3d.view_init(elev=30, azim=-45) # Adjust view for Rössler's structure

    x_p, y_p, z_p = solutions[0][:, 0], solutions[0][:, 1], solutions[0][:, 2]
    scatter_args = {'c': t_span, 'cmap': 'plasma', 's': 0.5, 'alpha': 0.7}

    ax_xy = fig_static.add_subplot(2, 2, 2); ax_xy.scatter(x_p, y_p, **scatter_args); ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y"); ax_xy.set_title("X-Y Projection"); ax_xy.grid(True, linestyle=':', alpha=0.6); ax_xy.set_aspect('equal', adjustable='box')
    ax_xz = fig_static.add_subplot(2, 2, 3); ax_xz.scatter(x_p, z_p, **scatter_args); ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z"); ax_xz.set_title("X-Z Projection"); ax_xz.grid(True, linestyle=':', alpha=0.6); ax_xz.set_aspect('equal', adjustable='box')
    ax_yz = fig_static.add_subplot(2, 2, 4); ax_yz.scatter(y_p, z_p, **scatter_args); ax_yz.set_xlabel("Y"); ax_yz.set_ylabel("Z"); ax_yz.set_title("Y-Z Projection"); ax_yz.grid(True, linestyle=':', alpha=0.6); ax_yz.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        fig_static.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  [+] Static plots saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not save static plots: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred saving static plots: {e}")
    finally:
        plt.close(fig_static)

# --- generate_save_animation (Adapted for Rössler) ---
def generate_save_animation(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    frame_step: int,
    tail_len_points: int,
    interval_ms: int,
    writer_name: Optional[str],
    fps: int,
    dpi: int,
) -> bool:
    """
    Generates and saves an animation of the Rössler attractor trajectory.

    Args:
        filepath: Path object for the output animation file (MP4/GIF).
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        frame_step: Step size through data for each animation frame.
        tail_len_points: Length of trajectory tail in number of data points.
        interval_ms: Delay between frames in milliseconds.
        writer_name: Name of Matplotlib animation writer (e.g., 'ffmpeg'). None skips saving.
        fps: Frames per second for the output video.
        dpi: Dots per inch for video frames.

    Returns:
        True if animation was saved successfully, False otherwise.
    """
    if writer_name is None:
        print("\n[*] Skipping animation generation: No valid writer.")
        return False
    if solution_data.size == 0:
         print("\n[*] Skipping animation generation: No solution data.")
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

    line_anim, = ax_anim.plot([], [], [], lw=1.5, color='magenta') # Line color
    point_anim, = ax_anim.plot([], [], [], 'o', color='lime', markersize=6, zorder=10) # Marker color

    ax_anim.set_xlim(x_anim.min(), x_anim.max())
    ax_anim.set_ylim(y_anim.min(), y_anim.max())
    ax_anim.set_zlim(z_anim.min(), z_anim.max())
    ax_anim.set_xlabel("X Axis"); ax_anim.set_ylabel("Y Axis"); ax_anim.set_zlabel("Z Axis")
    ax_anim.set_title("Rössler Attractor Animation")
    ax_anim.grid(True, linestyle=':', alpha=0.4)
    ax_anim.view_init(elev=30, azim=-45) # Consistent view

    def update(frame_idx_scaled: int) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
        current_data_idx = frame_idx_scaled * frame_step
        tail_start_data_idx = max(0, current_data_idx - tail_len_points)

        line_anim.set_data(x_anim[tail_start_data_idx : current_data_idx+1],
                           y_anim[tail_start_data_idx : current_data_idx+1])
        line_anim.set_3d_properties(z_anim[tail_start_data_idx : current_data_idx+1])

        point_anim.set_data(x_anim[current_data_idx : current_data_idx+1],
                            y_anim[current_data_idx : current_data_idx+1])
        point_anim.set_3d_properties(z_anim[current_data_idx : current_data_idx+1])
        return line_anim, point_anim

    print(f"  [*] Creating animation with {num_frames} frames...")
    try:
        anim = FuncAnimation(fig_anim, update, frames=num_frames, interval=interval_ms, blit=True, repeat=False)
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
        print("\n  [+] Animation saved successfully.")
        animation_saved = True
    except FileNotFoundError:
        print(f"\n  [!] Error: Animation writer '{writer_name}' not found or failed.")
        print(f"      Ensure '{writer_name}' is installed and in PATH.")
    except Exception as e:
        print(f"\n  [!] Error saving animation: {e}")
    finally:
        print(" " * 80, end='\r') # Clear progress line
        plt.close(fig_anim)

    return animation_saved

# --- generate_save_interactive_html (Adapted for Rössler) ---
def generate_save_interactive_html(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    params: RosslerParams, # Takes the (a, b, c) tuple
) -> bool:
    """
    Generates and saves an interactive 3D plot of the Rössler attractor using Plotly.

    Args:
        filepath: Path object for the output HTML file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        params: Tuple of Rössler parameters (a, b, c).

    Returns:
        True if HTML was saved successfully, False otherwise.
    """
    if solution_data.size == 0:
         print("\n[*] Skipping interactive HTML generation: No solution data.")
         return False

    print(f"\n[*] Generating interactive HTML plot to: {filepath.name}")
    a, b, c = params
    x_trace, y_trace, z_trace = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]

    try:
        trace = go.Scatter3d(
            x=x_trace, y=y_trace, z=z_trace, mode='lines',
            line=dict(color=t_span, colorscale='Viridis', width=3, # Good default colorscale
                      colorbar=dict(title='Time')),
            name='Rössler Trajectory'
        )

        layout = go.Layout(
            title=dict(
                text=f"Interactive Rössler Attractor<br>(a={a:.2f}, b={b:.2f}, c={c:.2f})",
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis',
                xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                # Adjust aspect ratio and camera for Rössler (flatter than Lorenz)
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=1.2, y=-1.8, z=0.8) # Adjust for good initial view
            ),
            margin=dict(l=10, r=10, b=10, t=50),
            paper_bgcolor='rgba(255,255,255,0.9)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(filepath, include_plotlyjs='cdn')

        print(f"  [+] Interactive HTML saved successfully to {filepath.name}")
        return True

    except ImportError:
        print("  [!] Error: Plotly library not installed (pip install plotly).")
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
    file_uri = filepath.resolve().as_uri()

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
    Main function to orchestrate the Rössler system simulation and analysis.

    Returns:
        0 if successful, 1 otherwise.
    """
    print("=" * 60)
    print(" Rössler System Simulation, Analysis, and Visualization")
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

    rossler_params: RosslerParams = (ROSSLER_PARAM_A, ROSSLER_PARAM_B, ROSSLER_PARAM_C)
    t_span_vis = np.linspace(0.0, VIS_T_END, VIS_NUM_STEPS)
    t_span_lle = np.linspace(0.0, LLE_T_END, LLE_NUM_STEPS)
    print(f"[*] System Parameters: a={rossler_params[0]:.2f}, b={rossler_params[1]:.2f}, c={rossler_params[2]:.2f}")
    print(f"[*] Visualization Time: 0.0 to {VIS_T_END} ({VIS_NUM_STEPS} steps)")
    print(f"[*] LLE Calculation Time: 0.0 to {LLE_T_END} ({LLE_NUM_STEPS} steps)")

    # --- Solve for Visualization ---
    try:
        solutions = solve_visual_trajectories(INITIAL_STATES, t_span_vis, rossler_system, rossler_params)
    except Exception as e:
        print(f"[!] Fatal Error during ODE solving for visualization: {e}")
        return 1

    if not solutions:
        print("[!] Aborting: No trajectories could be solved.")
        return 1
    first_solution = solutions[0]

    # --- Calculate LLE ---
    lle_estimate, lle_interpretation, lle_sim_time = calculate_lle(
        INITIAL_STATES[0], t_span_lle, rossler_system, rossler_params, LLE_EPSILON
    )

    # --- Save LLE Results ---
    save_lle_results(
        lle_results_path, rossler_params,
        lle_sim_time, LLE_T_END, len(t_span_lle), LLE_EPSILON, lle_interpretation,
    )

    # --- Generate and Save Static Plots ---
    generate_save_static_plots(
        static_plot_path, solutions, t_span_vis, INITIAL_STATES,
        rossler_params, lle_estimate, STATIC_PLOT_DPI,
    )

    # --- Generate and Save Animation Video ---
    if ANIMATION_WRITER:
        animation_saved = generate_save_animation(
            video_animation_path, first_solution, t_span_vis, FRAME_STEP, TAIL_LENGTH,
            ANIMATION_INTERVAL, ANIMATION_WRITER, ANIMATION_FPS, ANIMATION_DPI,
        )
    else:
        print("\n[*] Skipping animation saving (no valid writer).")
        animation_saved = False # Define variable anyway

    # --- Generate and Save Interactive HTML Plot ---
    html_saved = generate_save_interactive_html(
        interactive_html_path, first_solution, t_span_vis, rossler_params
    )

    # --- Attempt to Open INTERACTIVE HTML (Cross-Platform) ---
    # Replaced attempt_termux_open with open_file_cross_platform
    if html_saved and interactive_html_path.exists():
        open_file_cross_platform(interactive_html_path)
    elif not html_saved:
        print("\n[*] Skipping file opening: Interactive HTML generation failed.")
    else:
        print("\n[*] Skipping file opening: HTML file not found after reported success.")


    print("\n[+] Rössler script finished successfully.")
    print("=" * 60)
    return 0


# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
