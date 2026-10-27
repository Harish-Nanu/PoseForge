# PoseForge

A real-time **augmented reality marker tracking pipeline** in modern C++ (C++20) with OpenCV.
PoseForge finds ArUco markers in camera frames, estimates each marker's 6-DoF pose with
`solvePnP`, and draws a 3D cube anchored on it. Work is split into a multithreaded
capture → detect → render pipeline with per-stage latency measurement.

> **Status: WIP.** See the roadmap below.

## Roadmap
1. Project scaffold, CMake, gtest, CI
2. Synthetic scene generator (marker rendered at a known pose, ground truth for tests)
3. ArUco marker detection
4. Pose estimation (`solvePnP`, IPPE square) + reprojection error
5. AR overlay: 3D cube and axes renderer, `poseforge run` on camera / video / synthetic input
6. Bounded thread-safe queue + multithreaded pipeline
7. Parallel detection workers with in-order reassembly
8. Latency statistics (p50/p95/p99) + `poseforge bench` (sequential vs pipelined)
9. AddressSanitizer / ThreadSanitizer jobs in CI

## Build
```bash
sudo apt install cmake ninja-build pkg-config libopencv-dev libgtest-dev   # OpenCV >= 4.7
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```
