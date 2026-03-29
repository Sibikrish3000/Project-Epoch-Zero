# Contributing to Project Epoch Zero

Thank you for your interest in contributing to **Project Epoch Zero**! As a research-grade Space Situational Awareness and Orbital Defense system, community contributions are essential to its continuous improvement and real-world applicability. This document outlines the guidelines and governance expectations for participating in the project.

## Code of Conduct & Governance

By participating in this project, you agree to foster a welcoming, respectful, and harassment-free environment for all participants. 

### Governance Expectations
Project Epoch Zero is primarily an open-source research initiative designed to advance orbital mechanics and Physics-Informed Neural Networks (PINNs). 
- **Lead Maintainers**: Final decisions on architecture, core physical simulations, and model weights are made by the core maintainers. 
- **Decentralized Support**: The project serves as a public common architecture. While we provide best-effort support via GitHub Issues, we encourage the community to help answer questions and review PRs.

## How to Contribute

We welcome contributions in several forms: bug reports, feature requests, documentation improvements, and code changes.

### 1. Reporting Bugs and Security Issues
If you find a bug or issue, please use the GitHub Issue Tracker. 
Provide a clear description of the issue, including:
- Steps to reproduce
- Operating system and environment details
- Logs or screenshots (especially for the WebGL frontend)

### 2. Suggesting Enhancements
Enhancements to the SGP4 propagation pipeline, neural network architecture, or WebGL engine are highly welcome. Please open an Issue outlining the proposed feature before writing code to ensure it aligns with the project's roadmap and scope.

### 3. Submitting Pull Requests (PRs)
To contribute code to the repository:
1. **Fork the Repository**: Create your own fork and clone it locally.
2. **Branch**: Create a new branch off the `main` branch (e.g., `feature/improved-drag-model`).
3. **Commit**: Write clear, descriptive commit messages.
4. **Test**: Ensure your changes are fully covered by tests in the `tests/` directory and pass existing tests.
5. **Document**: If you added a new feature, update the `README.md` and inline docstrings.
6. **Submit PR**: Open a Pull Request against the `main` branch. Provide a comprehensive summary of the changes and link to any related issues.

## Development Setup

We recommend using `uv` or a `conda` environment to manage dependencies:
```bash
git clone https://github.com/Sibikrish3000/Project-Epoch-Zero.git
cd Project-Epoch-Zero
uv sync # or pip install -r requirements.txt
```

To run the local test suite:
```bash
pytest tests/
```

Thank you helping make orbit propagation safer and more accessible!
