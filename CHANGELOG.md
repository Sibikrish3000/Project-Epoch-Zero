# Changelog

All notable changes to **Project Epoch Zero** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-03-29
### Added
- **Fleet Surveillance Mode**: New ability to process massive constellation groups or debris clouds simultaneously rather than single pairings.
- **WebGL Mission Control**: Replaced legacy Dash/Plotly static UI with a high-performance `Three.js` dynamic 3D visualizer running at 60 FPS.
- **Dynamic Tails**: Render trajectory trails dynamically across time instead of static orbit loops.
- **Flask REST API**: A streamlined Python backend tailored for high-frequency coordinate JSON streaming.

### Changed
- **Gated-PINN (v3.3) Architecture**: Updated the PyTorch model to incorporate a hard bounding Physics Gate ($\Gamma(t)=\tanh(\lambda t)$) to prevent "Phantom Drift" at epoch $t=0$.
- **SGP4 Propagation Pipeline**: Enhanced with J2 constraints and F10.7 space weather proxy inputs.

## [4.1.0] - 2025-11-15
### Added
- Integrated CelesTrak live API fetching via `tle_fetcher.py`.
- Automated debris classification tagging.

### Fixed
- Addressed memory leaks in the Monte Carlo risk modeling during extended propagation horizons.
- Resolved Mahalanobis distance NaN errors during perfectly co-orbital alignments.

## [4.0.0] - 2025-06-01
### Added
- First stable release of the Physics-Informed Neural Network (PINN) architecture.
- Added support for Solar Cycle 25 thermospheric drag anomalies.
- Plotly 3D Conjunction dashboards.

### Changed
- Transitioned purely deterministic analytical codebase into an ML-hybrid stack.
