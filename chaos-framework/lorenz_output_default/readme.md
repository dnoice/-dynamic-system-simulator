# Unified Chaos Simulator Framework (v2.0)

**Explore the fascinating worlds of classic chaotic systems like Lorenz, Rössler, Chua, Thomas, and Chen using this unified simulation and visualization tool.**  
This framework provides a flexible platform, accessible via both command-line arguments for quick runs and an interactive prompt mode for guided exploration. Generate insightful analyses (Largest Lyapunov Exponent), static plots, dynamic animations, and interactive 3D visualizations.

## 1. Introduction to Chaos

Deterministic chaos describes systems governed by precise rules yet exhibiting unpredictable long-term behavior. This framework allows exploration of key chaotic principles:

- **Sensitivity to Initial Conditions (Butterfly Effect):** Tiny differences in starting values lead to wildly different outcomes.
- **Strange Attractors:** Trajectories remain confined to specific bounded regions with complex, fractal-like structures.
- **Largest Lyapunov Exponent (LLE):** Measures the exponential rate of divergence between nearby trajectories, with a positive value indicating chaos.

## 2. Included Chaotic Systems

The simulator supports five classic chaotic systems:

| System  | Description |
|---------|------------|
| **Lorenz**  | Atmospheric modeling; famous "butterfly" attractor. |
| **Rössler** | Simple chaotic design; features "folded-band" attractor. |
| **Chua**    | Electronic circuit model; produces "double-scroll" attractor. |
| **Thomas**  | Cyclic symmetry with sine functions; intricate looping structures. |
| **Chen**    | Related to Lorenz but with distinct chaotic dynamics. |

### Example: Lorenz System Equations

$$
\begin{aligned}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= x (\rho - z) - y \\
\frac{dz}{dt} &= x y - \beta z
\end{aligned}
$$

## 3. Framework Features

- **Multi-System Support:** Simulate Lorenz, Rössler, Chua, Thomas, or Chen attractors.
- **Dual Operation Modes:**  
  - **Command-Line Interface (CLI):** Run preconfigured simulations quickly.  
  - **Interactive Prompt Mode:** Guided setup with parameter selection.
- **Advanced Numerical Solving:** Uses `scipy.integrate.odeint` for accurate integration.
- **Analysis Tools:** Computes LLE for chaos detection.
- **Visualization Options:**  
  - **Static Plots:** Multi-panel PNG images of 3D attractors + 2D projections.
  - **Animations:** MP4 video showing trajectory evolution.
  - **Interactive 3D HTML Visualization:** Rotate/zoom attractors in-browser.
- **Standardized Outputs:** Organized in a dedicated directory for easy access.
- **Automated File Opening:** Launch interactive plots automatically.

## 4. Example Outputs

### Static Plot (3D Attractor + 2D Projections)
[Click here to view example plot](../assets/example_static_plot.png)

### Animation Video (MP4)
[Click here to view example animation](../assets/example_animation.mp4)

## 5. Dependencies

Ensure the following are installed:

```bash
pip install numpy scipy matplotlib plotly
```

Additionally, **FFmpeg** is required for MP4 animations:
```bash
# Install FFmpeg (Linux)
sudo apt install ffmpeg

# Install FFmpeg (Mac)
brew install ffmpeg

# Install FFmpeg (Termux)
pkg install ffmpeg
```

## 6. Running the Script

### Interactive Mode (Recommended):
```bash
python chaos_framework_final.py
```
Guided setup allows parameter selection and visualization choices.

### Command-Line Mode:
```bash
python chaos_framework_final.py --system lorenz
python chaos_framework_final.py --system chua --p1 10.0 --no_lle
python chaos_framework_final.py --system thomas --output_dir my_run --no_lle --no_static --no_animation
```

## 7. Command-Line Arguments Reference

| Argument | Description |
|----------|------------|
| `-s, --system`  | Select chaotic system (`lorenz`, `rossler`, etc.). Required in CLI mode. |
| `--x0, --y0, --z0` | Set initial conditions for simulation. |
| `--p1, --p2, --p3, --p4` | Override system-specific parameters. |
| `--vis_t_end, --vis_steps` | Adjust visualization duration and resolution. |
| `--lle_t_end, --lle_steps` | Configure LLE calculation range. |
| `--frame_step, --tail_length` | Animation frame rate and trajectory length. |
| `--output_dir` | Custom output folder name. |
| `--no_lle, --no_static, --no_animation, --no_html` | Skip specific output generation. |

## 8. Output Files

Generated results are stored under `<system_name>_output_<mode>/`.  

| File | Description |
|------|------------|
| `<system_name>_lle_results.txt` | Chaos analysis results. |
| `<system_name>_static_plots.png` | Snapshot of attractor structure. |
| `<system_name>_animation.mp4` | Dynamic visualization of trajectory evolution. |
| `<system_name>_interactive.html` | Interactive 3D plot viewable in a browser. |

## 9. Further Exploration

- **Compare Systems:** Run different attractors and analyze LLE results.
- **Parameter Sweeps:** Modify parameters to observe bifurcations and system changes.
- **Interactive Testing:** Use prompt mode to easily experiment with values.
- **Framework Extension:** Define new chaotic systems within `SYSTEM_DATA`.

---
