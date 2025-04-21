# Python Dynamical Systems Simulator Suite

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Welcome! This repository contains Python scripts for simulating, analyzing, and visualizing various fascinating dynamical systems, with a focus on classic chaotic attractors.

Whether you want simple standalone scripts for individual systems or a powerful, unified framework with advanced features, you'll find it here. Explore the beautiful complexity of chaos!

![Showcase Attractor Collage](showcase_collage.png)
*(Suggestion: Create a collage image showcasing outputs from Lorenz, Rössler, Chua, Thomas, and Chen and replace placeholder)*

## Repository Contents

This repository is organized into the following main parts:

1.  **Individual System Scripts:**
    * Dedicated directories (`lorenz/`, `rossler/`, `chua/`, `thomas/`, `chen/`) each containing a simpler, standalone Python script focused *only* on simulating and potentially plotting that specific system. These are great starting points or for straightforward demonstrations. *(Assumption: You will create/organize these)*
    * *(Optional: Add brief description or link to each directory)*

2.  **Unified Chaos Framework (`framework/`):**
    * Contains the advanced, multi-system simulator (`chaos_framework.py`).
    * **This is the most feature-rich tool here.** It supports all included systems, LLE calculation, static plots, animation videos, interactive HTML plots (via Plotly), and can be run via command-line flags or a user-friendly interactive mode.
    * **For detailed usage, features, and examples of the framework, please see its dedicated README:** 👉 **[Framework README](./framework/README.md)** 👈

## Getting Started

* **Python:** Requires Python 3.x.
* **Dependencies:** Core dependencies include `numpy`, `scipy`, `matplotlib`, and `plotly`. Install via pip:
    ```bash
    pip install numpy scipy matplotlib plotly
    ```
* **Animation:** Generating `.mp4` animations requires `ffmpeg` to be installed externally (use your system's package manager like `apt`, `brew`, `pkg`, etc.).
* Refer to the specific README within the `framework/` directory for detailed framework dependencies and setup. Individual system scripts might have simpler requirements.

## Basic Usage

* **Individual Scripts:** Navigate into a system's directory and run its script (e.g., `cd lorenz && python lorenz_script.py`). Check the script's comments or add specific READMEs in those folders for details.
* **Unified Framework:** Navigate into the `framework/` directory.
    * For interactive mode: `python chaos_framework.py`
    * For command-line help: `python chaos_framework.py --help`
    * For specific system run via CLI: `python chaos_framework.py --system <name> [options...]`
    * **See the [Framework README](./framework/README.md) for all CLI options.**

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for details.

---

Feel free to explore, experiment, and summon some chaos!
