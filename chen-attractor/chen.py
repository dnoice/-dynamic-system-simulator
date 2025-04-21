# -*- coding: utf-8 -*-
"""
Chen Attractor Simulator, Analyzer, and Visualizer (PEP 8 Compliant)

This script simulates the Chen chaotic attractor, a dynamical system exhibiting
chaotic behavior. It performs the following actions:

1.  **Solves ODEs:** Integrates the Chen system differential equations over time
    using SciPy's odeint for multiple initial conditions.
2.  **Calculates LLE:** Estimates the Largest Lyapunov Exponent (LLE) to quantify
    the system's sensitivity to initial conditions, indicating chaos if positive.
3.  **Generates Static Plots:** Creates a multi-panel figure (using Matplotlib)
    showing the 3D attractor and its 2D projections (XY, XZ, YZ).
4.  **Generates Animation:** Produces an MP4 video animation of the attractor's
    trajectory over time (requires FFmpeg).
5.  **Generates Interactive HTML:** Creates an interactive 3D plot using Plotly,
    allowing users to rotate, zoom, and explore the attractor in a web browser.
6.  **Saves Results:** Stores LLE calculation details and plots/videos in a
    dedicated output directory.
7.  **Opens Output:** Attempts to automatically open the generated interactive
    HTML file using the system's default web browser or 'termux-open' if available.

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
# Use 'Agg' backend for non-interactive environments or when saving figures
# without displaying them. Suppress potential UserWarning related to backend.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    try:
        matplotlib.use("Agg")
    except ImportError:
        print("[!] Warning: Failed to set Matplotlib backend to 'Agg'.")
        # Continue without setting backend, might cause issues in headless environments.

# =====================================
# Configuration Constants
# =====================================

# --- Output Settings ---
OUTPUT_DIR_NAME: str = "chen_output"
VIDEO_ANIMATION_FILENAME: str = "chen_animation.mp4"
INTERACTIVE_HTML_FILENAME: str = "chen_interactive.html"
STATIC_PLOT_FILENAME: str = "chen_static_plots.png"
LLE_RESULTS_FILENAME: str = "chen_lle_results.txt"

# --- Chen System Parameters (Classic Chaotic Regime) ---
CHEN_PARAM_A: float = 35.0
CHEN_PARAM_B: float = 3.0
CHEN_PARAM_C: float = 28.0

# --- Initial States (Slightly perturbed starting points) ---
INITIAL_STATES: List[List[float]] = [
    [-5.0, -5.0, 1.0],  # A common starting point for Chen attractor visualizations
    [-5.1, -5.0, 1.0],
    [-5.0, -5.1, 1.0],
]

# --- Simulation Time Spans ---
VIS_T_END: float = 50.0        # End time for visualization simulation
VIS_NUM_STEPS: int = 15000     # Number of time steps for visualization
LLE_T_END: float = 150.0       # End time for LLE calculation (needs longer time)
LLE_NUM_STEPS: int = 30000     # Number of time steps for LLE calculation

# --- LLE Calculation Settings ---
LLE_EPSILON: float = 1e-9      # Initial separation for LLE calculation

# --- Animation Settings ---
# Check if ffmpeg writer is available
DEFAULT_ANIMATION_WRITER: str = "ffmpeg"
if not writers.is_available(DEFAULT_ANIMATION_WRITER):
    print(f"[!] Warning: Matplotlib writer '{DEFAULT_ANIMATION_WRITER}' not found.")
    print("    Animation saving might fail. Consider installing FFmpeg.")
    # You could fall back to another writer if needed, e.g., 'pillow' for gifs
    # Or set ANIMATION_WRITER = None to disable animation saving attempt
    ANIMATION_WRITER: Optional[str] = None
else:
    ANIMATION_WRITER: Optional[str] = DEFAULT_ANIMATION_WRITER

ANIMATION_FPS: int = 30        # Frames per second for the video
ANIMATION_DPI: int = 100       # Resolution (dots per inch) for the video frames
FRAME_STEP: int = 10           # Use every Nth point for animation frame (controls speed)
TAIL_LENGTH: int = 300         # Number of *frames* (not points) for the trajectory tail
ANIMATION_INTERVAL: int = 25   # Delay between frames in milliseconds (influences playback speed)

# --- Visualization Settings ---
STATIC_PLOT_DPI: int = 150     # Resolution for the static PNG plot

# Type alias for state vector
State = np.ndarray

# =====================================
# Function Definitions
# =====================================

def chen_system(state: State, t: float, a: float, b: float, c: float) -> State:
    """
    Defines the differential equations for the Chen attractor.

    Args:
        state: Current state vector [x, y, z].
        t: Current time t (required by odeint, though not used in the equations).
        a, b, c: Chen system parameters.

    Returns:
        np.ndarray: Array containing the derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])

def solve_visual_trajectories(
    initial_states: List[List[float]],
    t_span: np.ndarray,
    system_func: Callable[[State, float, float, float, float], State],
    sys_params: Tuple[float, float, float],
) -> List[np.ndarray]:
    """
    Solves the ODE system for visualization purposes for multiple initial states.

    Args:
        initial_states: A list of initial state vectors [[x0, y0, z0], ...].
        t_span: Array of time points for the solution.
        system_func: The function defining the ODE system (e.g., chen_system).
        sys_params: Tuple of system parameters (a, b, c) required by system_func.

    Returns:
        A list of numpy arrays, where each array holds the trajectory [x(t), y(t), z(t)]
        corresponding to an initial state.
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

def calculate_lle(
    initial_state: List[float],
    t_span_lle: np.ndarray,
    system_func: Callable[[State, float, float, float, float], State],
    sys_params: Tuple[float, float, float],
    epsilon: float,
) -> Tuple[Optional[float], str, float]:
    """
    Estimates the Largest Lyapunov Exponent (LLE) for the given system.

    Args:
        initial_state: The starting state vector [x0, y0, z0].
        t_span_lle: Array of time points for the LLE calculation.
        system_func: The function defining the ODE system.
        sys_params: Tuple of system parameters (a, b, c).
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
    # Perturb only the first component for simplicity
    perturb_vector = np.zeros_like(state0)
    perturb_vector[0] = epsilon
    state1 = state0 + perturb_vector
    current_t = 0.0

    print("  [*] Running reference and perturbed simulations...")
    try:
        for i in range(total_steps_lle - 1):
            # Integrate both trajectories over one small time step dt_lle
            t_step_span = [current_t, current_t + dt_lle]
            sol0 = odeint(system_func, state0, t_step_span, args=sys_params)
            sol1 = odeint(system_func, state1, t_step_span, args=sys_params)

            # Check if integration was successful (returned expected shape)
            if sol0.shape[0] < 2 or sol1.shape[0] < 2:
                 print(f"\n  [!] Error: ODE integration failed at step {i}.")
                 return None, "LLE Calculation Failed (integration error).", current_t

            state0 = sol0[1]
            state1 = sol1[1]

            diff_vector = state1 - state0
            distance = np.linalg.norm(diff_vector)

            # Avoid log(0) or division by zero/very small numbers
            if distance > 1e-15:
                lle_sum += np.log(distance / epsilon)
                # Rescale the perturbed trajectory back to separation epsilon along the difference vector
                state1 = state0 + (diff_vector / distance) * epsilon
            else:
                # Trajectories collapsed or are too close; re-perturb
                print(f"  [!] Warning: Trajectories separation near zero at step {i}. Re-perturbing.")
                state1 = state0 + perturb_vector # Re-apply initial perturbation

            current_t += dt_lle

            # Progress update
            if (i + 1) % (max(1, total_steps_lle // 20)) == 0: # Update progress roughly 20 times
                progress = 100 * (i + 1) / total_steps_lle
                print(f"    Progress: {progress:.1f}%", end='\r')

        print("\n  [+] LLE Calculation loop finished.")

    except Exception as e:
        print(f"\n  [!] An error occurred during LLE calculation: {e}")
        return None, f"LLE Calculation Failed (Runtime Error: {e}).", current_t

    lle_estimate: Optional[float] = None
    lle_interpretation: str = "LLE Calculation Failed (division by zero?)."

    if current_t > 1e-9: # Avoid division by zero if simulation time is negligible
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
        print("  [!] LLE calculation could not proceed (simulation time is near zero).")


    return lle_estimate, lle_interpretation, current_t

def save_lle_results(
    filepath: pathlib.Path,
    params: Tuple[float, float, float],
    sim_time: float,
    target_time: float,
    num_steps: int,
    epsilon: float,
    interpretation: str,
) -> None:
    """
    Saves LLE calculation results and parameters to a text file.

    Args:
        filepath: Path object for the output text file.
        params: Tuple of Chen parameters (a, b, c).
        sim_time: Actual simulation time achieved during LLE calculation.
        target_time: The target simulation time for LLE calculation.
        num_steps: The number of integration steps used for LLE.
        epsilon: The initial separation distance used.
        interpretation: The string summarizing the LLE result and its meaning.
    """
    print(f"[*] Saving LLE results to: {filepath}")
    param_a, param_b, param_c = params
    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("=" * 30 + "\n")
            f.write(" Chen Attractor Analysis Results\n")
            f.write("=" * 30 + "\n\n")
            f.write("System Parameters:\n")
            f.write(f"  a = {param_a:.4f}\n")
            f.write(f"  b = {param_b:.4f}\n")
            f.write(f"  c = {param_c:.4f}\n\n")
            f.write("LLE Calculation Settings:\n")
            f.write(f"  Target Integration Time: {target_time:.2f} units\n")
            f.write(f"  Actual Integration Time: {sim_time:.2f} units\n")
            f.write(f"  Number of Steps:         {num_steps - 1}\n") # N points = N-1 steps
            f.write(f"  Initial Separation (ε):  {epsilon:.2e}\n\n")
            f.write("Lyapunov Exponent Results:\n")
            f.write(f"{interpretation}\n")
        print(f"  [+] LLE results saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not write LLE results to file: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred while saving LLE results: {e}")

def generate_save_static_plots(
    filepath: pathlib.Path,
    solutions: List[np.ndarray],
    t_span: np.ndarray,
    initial_states: List[List[float]],
    params: Tuple[float, float, float],
    lle_est: Optional[float],
    dpi: int,
) -> None:
    """
    Generates and saves static plots (3D and 2D projections) of the attractor.

    Args:
        filepath: Path object for the output PNG file.
        solutions: List of trajectory arrays ([x(t), y(t), z(t)]).
        t_span: Time array corresponding to the solutions.
        initial_states: List of starting points used.
        params: Tuple of Chen parameters (a, b, c).
        lle_est: Estimated LLE value (or None).
        dpi: Dots per inch for the saved image.
    """
    if not solutions:
        print("[!] Skipping static plot generation: No solution data available.")
        return

    print(f"\n[*] Generating and saving static plots to: {filepath.name}")
    param_a, param_b, param_c = params
    fig_static = plt.figure(figsize=(14, 10), dpi=dpi)

    # --- Title ---
    title_lle_part = f" | LLE ≈ {lle_est:.4f}" if lle_est is not None else ""
    title = (
        f"Chen Attractor (a={param_a:.1f}, b={param_b:.1f}, c={param_c:.1f})"
        f"{title_lle_part}"
    )
    fig_static.suptitle(title, fontsize=16)

    # --- 3D Plot ---
    ax3d = fig_static.add_subplot(2, 2, 1, projection="3d")
    num_trajectories = len(solutions)
    # Concatenate all points to find overall bounds
    all_x = np.concatenate([s[:, 0] for s in solutions])
    all_y = np.concatenate([s[:, 1] for s in solutions])
    all_z = np.concatenate([s[:, 2] for s in solutions])

    cmap_3d = 'inferno' # Colormap for trajectories
    linewidth_3d = 0.6
    alpha_3d = 0.7

    for i, solution in enumerate(solutions):
        x_sol, y_sol, z_sol = solution[:, 0], solution[:, 1], solution[:, 2]
        points = np.array([x_sol, y_sol, z_sol]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a Line3DCollection for efficient plotting with color mapping
        norm = plt.Normalize(t_span.min(), t_span.max())
        lc = Line3DCollection(
            segments, cmap=cmap_3d, norm=norm, linewidth=linewidth_3d, alpha=alpha_3d
        )
        lc.set_array(t_span[:-1]) # Color segments based on time
        ax3d.add_collection(lc)

    ax3d.set_xlabel("X Axis")
    ax3d.set_ylabel("Y Axis")
    ax3d.set_zlabel("Z Axis")
    ax3d.set_title(f"3D View ({num_trajectories} Trajectories)")
    ax3d.grid(True, linestyle='--', alpha=0.5)

    # Plot starting points
    starts = np.array(initial_states)
    ax3d.scatter(
        starts[:, 0], starts[:, 1], starts[:, 2],
        color='lime', s=70, marker='o', edgecolor='black',
        label='Start Points', depthshade=False, zorder=10 # zorder makes them appear on top
    )
    ax3d.legend(loc='upper left')

    # Set limits slightly padded
    pad_x = (all_x.max() - all_x.min()) * 0.05
    pad_y = (all_y.max() - all_y.min()) * 0.05
    pad_z = (all_z.max() - all_z.min()) * 0.05
    ax3d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax3d.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax3d.set_zlim(all_z.min() - pad_z, all_z.max() + pad_z)

    # Adjust view angle for better visualization of Chen structure
    ax3d.view_init(elev=25, azim=-120)

    # --- 2D Projections (using the first trajectory for clarity) ---
    x_p, y_p, z_p = solutions[0][:, 0], solutions[0][:, 1], solutions[0][:, 2]
    scatter_args = {'c': t_span, 'cmap': 'viridis', 's': 0.5, 'alpha': 0.6} # Different cmap for projections

    ax_xy = fig_static.add_subplot(2, 2, 2)
    ax_xy.scatter(x_p, y_p, **scatter_args)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("X-Y Projection")
    ax_xy.grid(True, linestyle=':', alpha=0.6)
    ax_xy.set_aspect('equal', adjustable='box') # Often helpful for projections

    ax_xz = fig_static.add_subplot(2, 2, 3)
    ax_xz.scatter(x_p, z_p, **scatter_args)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("X-Z Projection")
    ax_xz.grid(True, linestyle=':', alpha=0.6)
    ax_xz.set_aspect('equal', adjustable='box')

    ax_yz = fig_static.add_subplot(2, 2, 4)
    ax_yz.scatter(y_p, z_p, **scatter_args)
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_title("Y-Z Projection")
    ax_yz.grid(True, linestyle=':', alpha=0.6)
    ax_yz.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    try:
        fig_static.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  [+] Static plots saved successfully to {filepath.name}")
    except IOError as e:
        print(f"  [!] Error: Could not save static plots: {e}")
    except Exception as e:
        print(f"  [!] An unexpected error occurred while saving static plots: {e}")
    finally:
        plt.close(fig_static) # Ensure figure is closed to free memory


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
    Generates and saves an MP4 animation of the attractor trajectory.

    Args:
        filepath: Path object for the output MP4 file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data.
        frame_step: Step size through the data for each animation frame.
        tail_len_frames: Length of the trajectory tail in number of frames.
        interval_ms: Delay between frames in milliseconds.
        writer_name: Name of the Matplotlib animation writer (e.g., 'ffmpeg').
                     If None, saving will be skipped.
        fps: Frames per second for the output video.
        dpi: Dots per inch for the video frames.

    Returns:
        True if animation was saved successfully, False otherwise.
    """
    if writer_name is None:
        print("\n[*] Skipping animation generation: No valid writer specified (e.g., ffmpeg not found).")
        return False
    if solution_data.size == 0:
         print("\n[*] Skipping animation generation: No solution data available.")
         return False

    print(f"\n[*] Generating animation video (using '{writer_name}' writer)...")
    x_anim, y_anim, z_anim = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]
    num_points = len(t_span)
    num_frames = num_points // frame_step

    if num_frames < 2:
        print("  [!] Error: Not enough data points for animation (check VIS_NUM_STEPS and FRAME_STEP).")
        return False

    fig_anim = plt.figure(figsize=(10, 8), dpi=dpi)
    ax_anim = fig_anim.add_subplot(111, projection='3d') # Use 111 for single subplot

    # Initialize plot elements: line for tail, point for current position
    line_anim, = ax_anim.plot([], [], [], lw=1.5, color='deepskyblue') # Slightly thicker line
    point_anim, = ax_anim.plot([], [], [], 'o', color='red', markersize=6, zorder=10)

    # Set axis limits based on the full trajectory
    ax_anim.set_xlim(x_anim.min(), x_anim.max())
    ax_anim.set_ylim(y_anim.min(), y_anim.max())
    ax_anim.set_zlim(z_anim.min(), z_anim.max())

    ax_anim.set_xlabel("X Axis")
    ax_anim.set_ylabel("Y Axis")
    ax_anim.set_zlabel("Z Axis")
    ax_anim.set_title("Chen Attractor Animation")
    ax_anim.grid(True, linestyle=':', alpha=0.4)
    ax_anim.view_init(elev=25, azim=-120) # Consistent view angle

    # Update function for animation frames
    def update(frame_idx_scaled: int) -> Tuple[matplotlib.lines.Line2D, matplotlib.lines.Line2D]:
        """Updates the line and point data for each frame."""
        # Calculate the actual index in the full dataset
        current_data_idx = frame_idx_scaled * frame_step
        # Calculate the start index for the tail, ensuring it doesn't go below 0
        # Tail length is specified in *frames*, convert to data points
        tail_start_data_idx = max(0, current_data_idx - (tail_len_frames * frame_step))

        # Update tail line data (using slicing with step)
        line_anim.set_data(x_anim[tail_start_data_idx : current_data_idx+1 : frame_step],
                           y_anim[tail_start_data_idx : current_data_idx+1 : frame_step])
        line_anim.set_3d_properties(z_anim[tail_start_data_idx : current_data_idx+1 : frame_step])

        # Update current position marker data (index must be exact)
        point_anim.set_data(x_anim[current_data_idx : current_data_idx+1],
                            y_anim[current_data_idx : current_data_idx+1])
        point_anim.set_3d_properties(z_anim[current_data_idx : current_data_idx+1])

        # Update title with time (optional)
        # current_time = t_span[current_data_idx]
        # ax_anim.set_title(f"Chen Attractor Animation (t = {current_time:.2f})")

        return line_anim, point_anim

    print(f"  [*] Creating animation with {num_frames} frames...")
    # Create animation object
    # Note: blit=True can improve performance but might cause issues on some backends/OS.
    # If animation fails or looks wrong, try blit=False.
    try:
        anim = FuncAnimation(
            fig_anim, update, frames=num_frames,
            interval=interval_ms, blit=True, repeat=False
            )
    except Exception as e:
         print(f"  [!] Error initializing FuncAnimation (try blit=False?): {e}")
         plt.close(fig_anim)
         return False

    animation_saved = False
    print(f"  [*] Saving animation to: {filepath.name} (this may take a while)...")
    try:
        # Progress callback for user feedback during saving
        def progress_update(i: int, n: int):
            if n > 0 and ((i + 1) % 20 == 0 or (i + 1) == n): # Update every 20 frames or on the last frame
                print(f"    Saving frame {i+1}/{n} ({100*(i+1)/n:.1f}%)", end='\r')

        anim.save(
            filepath,
            writer=writer_name,
            fps=fps,
            dpi=dpi, # Use figure DPI for saving quality
            progress_callback=progress_update
            )
        print("\n  [+] Animation video saved successfully.")
        animation_saved = True
    except FileNotFoundError:
        # This error is more specific than the initial check, as it means
        # Matplotlib found the writer entry but couldn't execute the command.
        print(f"\n  [!] Error: Animation writer '{writer_name}' command failed.")
        print(f"      Ensure '{writer_name}' is installed and accessible in your system's PATH.")
        print(f"      (e.g., for FFmpeg: 'sudo apt install ffmpeg' or 'brew install ffmpeg')")
    except Exception as e:
        print(f"\n  [!] Error occurred while saving animation: {e}")
        print(f"      Writer: {writer_name}, FPS: {fps}, DPI: {dpi}")
    finally:
        # Clear the progress line
        print(" " * 80, end='\r')
        plt.close(fig_anim) # Ensure figure is closed

    return animation_saved

def generate_save_interactive_html(
    filepath: pathlib.Path,
    solution_data: np.ndarray,
    t_span: np.ndarray,
    params: Tuple[float, float, float],
) -> bool:
    """
    Generates and saves an interactive 3D plot using Plotly.

    Args:
        filepath: Path object for the output HTML file.
        solution_data: Single trajectory array [x(t), y(t), z(t)].
        t_span: Time array corresponding to the solution data (used for color).
        params: Tuple of Chen parameters (a, b, c).

    Returns:
        True if HTML was saved successfully, False otherwise.
    """
    if solution_data.size == 0:
         print("\n[*] Skipping interactive HTML generation: No solution data available.")
         return False

    print(f"\n[*] Generating interactive HTML plot to: {filepath.name}")
    param_a, param_b, param_c = params
    x_trace, y_trace, z_trace = solution_data[:, 0], solution_data[:, 1], solution_data[:, 2]

    try:
        # Create the 3D line trace
        trace = go.Scatter3d(
            x=x_trace,
            y=y_trace,
            z=z_trace,
            mode='lines',
            line=dict(
                color=t_span,        # Color line segments by time
                colorscale='inferno',# Choose a visually appealing colorscale
                width=3,             # Line width
                colorbar=dict(title='Time') # Add a color bar legend
            ),
            name='Chen Trajectory'
        )

        # Define the layout
        layout = go.Layout(
            title=dict(
                text=f"Interactive Chen Attractor<br>(a={param_a:.1f}, b={param_b:.1f}, c={param_c:.1f})",
                x=0.5, # Center title
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", zerolinecolor="white"),
                aspectratio=dict(x=1, y=1, z=0.7), # Adjust aspect ratio if needed
                camera_eye=dict(x=1.5, y=1.5, z=1.5) # Initial camera position
            ),
            margin=dict(l=10, r=10, b=10, t=50), # Adjust margins
            paper_bgcolor='rgba(255,255,255,0.9)', # Slightly off-white background
            plot_bgcolor='rgba(0,0,0,0)'        # Transparent plot area background
        )

        # Create the figure and save to HTML
        fig = go.Figure(data=[trace], layout=layout)
        # Use 'cdn' to load Plotly.js from network, reducing file size.
        # Use include_plotlyjs=True for a fully self-contained file.
        fig.write_html(filepath, include_plotlyjs='cdn')

        print(f"  [+] Interactive HTML saved successfully to {filepath.name}")
        return True

    except ImportError:
        print("  [!] Error: Plotly library is not installed.")
        print("      Please install it: pip install plotly")
        return False
    except Exception as e:
        print(f"  [!] An unexpected error occurred while generating interactive HTML: {e}")
        return False


def open_file_cross_platform(filepath: pathlib.Path) -> None:
    """
    Attempts to open the specified file using the default system application.
    Prioritizes 'termux-open' if available (for Android/Termux),
    otherwise uses the standard 'webbrowser' module.

    Args:
        filepath: Path object of the file to open.
    """
    print(f"\n[*] Attempting to open file: {filepath.resolve()}")
    file_uri = filepath.resolve().as_uri() # Use file:// URI scheme

    # --- Check for Termux Environment ---
    # Method 1: Check for termux-open command existence
    termux_open_cmd = shutil.which('termux-open')
    # Method 2: Check environment variable (might not always be set)
    is_termux_env = 'TERMUX_VERSION' in os.environ
    # Method 3: Check specific Termux paths (less reliable)
    # is_termux_path = os.path.exists('/data/data/com.termux')

    termux_used = False
    if termux_open_cmd:
        print(f"  [*] 'termux-open' command found. Attempting to use it...")
        command = [termux_open_cmd, str(filepath.resolve())]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=15
                )
            print(f"  [+] 'termux-open' executed successfully.")
            if result.stdout:
                 print(f"    Output: {result.stdout.strip()}")
            termux_used = True # Flag that Termux method was tried and succeeded
        except FileNotFoundError:
            # Should not happen if shutil.which found it, but handle defensively
            print(f"  [!] Error: '{termux_open_cmd}' command not found during execution (unexpected).")
        except subprocess.CalledProcessError as e:
            print(f"  [!] Error: 'termux-open' command failed (exit code {e.returncode}).")
            if e.stderr:
                print(f"    Stderr: {e.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("  [!] Error: 'termux-open' command timed out.")
        except Exception as e:
            print(f"  [!] An unexpected error occurred executing 'termux-open': {e}")

    # --- Fallback to standard webbrowser module ---
    if not termux_used:
        if termux_open_cmd:
             print("  [*] 'termux-open' failed, falling back to standard webbrowser...")
        else:
             print("  [*] 'termux-open' not found. Using standard webbrowser module...")

        try:
            opened = webbrowser.open(file_uri)
            if opened:
                print(f"  [+] Standard webbrowser module successfully requested opening: {file_uri}")
            else:
                print(f"  [!] Standard webbrowser module reported it could not open the file.")
                print(f"      Platform: {platform.system()}")
                print(f"      Try opening the file manually: {filepath.resolve()}")
        except webbrowser.Error as e:
             print(f"  [!] An error occurred using the webbrowser module: {e}")
             print(f"      Try opening the file manually: {filepath.resolve()}")
        except Exception as e:
            print(f"  [!] An unexpected error occurred using the webbrowser module: {e}")
            print(f"      Try opening the file manually: {filepath.resolve()}")


# =====================================
# Main Execution Logic
# =====================================

def main() -> int:
    """
    Main function to orchestrate the Chen Attractor simulation and analysis.

    Returns:
        0 if successful, 1 otherwise.
    """
    print("=" * 60)
    print(" Chen Attractor Simulation, Analysis, and Visualization")
    print("=" * 60)

    # --- Setup Output Directory ---
    try:
        output_path_obj = pathlib.Path(OUTPUT_DIR_NAME)
        output_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Output directory: {output_path_obj.resolve()}")
    except OSError as e:
        print(f"[!] Fatal Error: Could not create output directory '{OUTPUT_DIR_NAME}': {e}")
        return 1 # Indicate failure

    # Construct full paths for output files
    video_animation_path = output_path_obj / VIDEO_ANIMATION_FILENAME
    interactive_html_path = output_path_obj / INTERACTIVE_HTML_FILENAME
    static_plot_path = output_path_obj / STATIC_PLOT_FILENAME
    lle_results_path = output_path_obj / LLE_RESULTS_FILENAME

    # --- System Parameters and Time Spans ---
    chen_params = (CHEN_PARAM_A, CHEN_PARAM_B, CHEN_PARAM_C)
    t_span_vis = np.linspace(0.0, VIS_T_END, VIS_NUM_STEPS)
    t_span_lle = np.linspace(0.0, LLE_T_END, LLE_NUM_STEPS)
    print(f"[*] System Parameters: a={chen_params[0]}, b={chen_params[1]}, c={chen_params[2]}")
    print(f"[*] Visualization Time: 0.0 to {VIS_T_END} ({VIS_NUM_STEPS} steps)")
    print(f"[*] LLE Calculation Time: 0.0 to {LLE_T_END} ({LLE_NUM_STEPS} steps)")

    # --- Solve for Visualization ---
    # Catch potential errors during ODE solving
    try:
        solutions = solve_visual_trajectories(INITIAL_STATES, t_span_vis, chen_system, chen_params)
    except Exception as e:
        print(f"[!] Fatal Error during ODE solving for visualization: {e}")
        return 1 # Indicate failure

    if not solutions:
        print("[!] Aborting: No trajectories could be solved for visualization.")
        return 1 # Indicate failure

    # Use the first trajectory for single-trajectory outputs (animation, HTML)
    first_solution = solutions[0]

    # --- Calculate LLE ---
    # Initial state for LLE can be the first one used for visualization
    lle_estimate, lle_interpretation, lle_sim_time = calculate_lle(
        INITIAL_STATES[0], t_span_lle, chen_system, chen_params, LLE_EPSILON
    )

    # --- Save LLE Results ---
    save_lle_results(
        lle_results_path, chen_params,
        lle_sim_time, LLE_T_END, len(t_span_lle), LLE_EPSILON, lle_interpretation
    )

    # --- Generate and Save Static Plots ---
    generate_save_static_plots(
        static_plot_path, solutions, t_span_vis, INITIAL_STATES,
        chen_params, lle_estimate, STATIC_PLOT_DPI
    )

    # --- Generate and Save Animation Video ---
    # Check ANIMATION_WRITER again in case it was set to None earlier
    if ANIMATION_WRITER:
        generate_save_animation(
            video_animation_path, first_solution, t_span_vis, FRAME_STEP, TAIL_LENGTH,
            ANIMATION_INTERVAL, ANIMATION_WRITER, ANIMATION_FPS, ANIMATION_DPI
        )
    else:
        print("\n[*] Skipping animation saving (no valid writer).")


    # --- Generate and Save Interactive HTML Plot ---
    html_saved = generate_save_interactive_html(
        interactive_html_path, first_solution, t_span_vis, chen_params
        )

    # --- Attempt to Open INTERACTIVE HTML (Cross-Platform) ---
    if html_saved and interactive_html_path.exists():
        open_file_cross_platform(interactive_html_path)
    elif not html_saved:
        print("\n[*] Skipping file opening: Interactive HTML generation failed.")
    else:
        # This case should be rare if html_saved is True
        print("\n[*] Skipping file opening: Interactive HTML file not found despite reporting success.")


    print("\n[+] Script finished successfully.")
    print("=" * 60)
    return 0 # Indicate success


# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) # Exit with 0 for success, 1 for failure
