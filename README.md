# bluerov2-docking :anchor:

ROS2 simulation and perception stack for autonomous BlueROV2 underwater docking using ArUco markers and LED detection. UNSW Thesis research project.

https://github.com/user-attachments/assets/768a735d-8360-44a3-a05e-0bcd23649349

## Overview

The system drives the BlueROV2 onto a dock through a perception-to-control pipeline, sequenced by a finite-state machine:

1. **Perception** fuses multiple ArUco markers into a single 6-DOF dock pose and smooths it with a Kalman filter, publishing the filtered pose plus a health signal. (Far-range acquisition is out of scope — the pipeline assumes the dock is already detected.)
2. **Coarse approach** (PBVS) servos on the filtered dock pose to drive the vehicle to a standoff point on the dock's entry axis, ready for handoff. Runs in `ALT_HOLD` (the autopilot holds depth during the transit).
3. **Fine alignment** (align-then-advance PBVS) takes over at the standoff and drives the vehicle onto the dock entry, holding lateral/vertical/yaw alignment before each step forward. Runs in `STABILIZE` so the controller owns the precise terminal descent.
4. **Orchestrator** — a [YASMIN](https://github.com/uleroboticsgroup/yasmin) state machine sequences `IDLE → COARSE → FINE → DOCKED`, gates each phase controller via `/docking/state`, and owns the flight-mode and arming transitions. The operator engages autonomy by holding a joystick deadman button (momentary); releasing hands control back to manual `POSHOLD`.

### Packages

| Package | Description |
|---------|-------------|
| `description` | Docking station Gazebo model + sinusoidal dock-motion plugin, world file, RViz + Foxglove configs |
| `sim` | Simulation launch file |
| `perception` | ArUco detection, multi-marker fusion + Kalman filter, LED mock publisher |
| `control` | Coarse approach + fine alignment PBVS controllers |
| `orchestrator` | YASMIN docking FSM + joystick deadman/engage relay |
| `interfaces` | Custom messages (filter health, dock-pose measurement, coarse/fine status, docking state) |

## Prerequisites

- **x86_64 (amd64) host required** — the base image and ArduSub SITL have no ARM builds. Mac (Apple Silicon) and ARM VMs are not supported.
- [Docker](https://docs.docker.com/get-docker/)
- NVIDIA GPU: `nvidia-container-toolkit` on the host
- Nouveau/Intel/AMD: no extra GPU setup required

## Setup

Clone the repo:

```bash
git clone https://github.com/alanchoi00/bluerov2-docking.git
cd bluerov2-docking
```

### Dev container (recommended)

Requires [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

1. Open the repo in VS Code and select **Reopen in Container** when prompted.
2. Choose your GPU variant:
   - `bluerov2-docking (Nouveau)`: AMD/Intel or no GPU
   - `bluerov2-docking (NVIDIA)`: NVIDIA GPU

VS Code will pull the image, build the container, and run `rosdep install` + `colcon build` automatically. The workspace is mounted so edits in VS Code are reflected inside the container immediately.

To update: pull the new image with `docker pull`, then run **Dev Containers: Rebuild Container** from the command palette.

### Docker Compose

1. Start the container:

   ```bash
   # Nouveau/Intel/AMD
   docker compose -f .docker/compose/nouveau-desktop.yaml up -d

   # NVIDIA
   docker compose -f .docker/compose/nvidia-desktop.yaml up -d
   ```

2. Open a shell and build:

   ```bash
   docker exec -it <container_name> bash
   cd /home/ubuntu/ws_docking
   rosdep install -y --from-paths src --ignore-src --rosdistro jazzy \
     --skip-keys='gz-transport13 gz-sim8 gz-math7 gz-msgs10 gz-plugin2'
   colcon build --symlink-install
   source install/setup.bash
   ```

3. Stop when done:

   ```bash
   docker compose -f .docker/compose/nouveau-desktop.yaml down
   ```

To update the base image:

```bash
docker pull ghcr.io/alanchoi00/blue-sim:jazzy-desktop        # Nouveau
docker pull ghcr.io/alanchoi00/blue-sim:jazzy-desktop-nvidia  # NVIDIA
```

## Simulation

The Gazebo environment is built on [alanchoi00/blue-sim](https://github.com/alanchoi00/blue-sim) (ROS2 Jazzy). Pre-built Docker images are published to GHCR.

### Launch

```bash
ros2 launch sim sim.launch.py
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `use_ardusub` | `true` | Start ArduSub SITL + MAVROS bridge |
| `flight_mode` | `POSHOLD` | ArduSub startup flight mode (the docking FSM overrides it per phase when `use_control:=true`) |
| `use_joy` | `false` | Joystick teleop |
| `use_key` | `false` | Keyboard teleop |
| `use_mock_led` | `true` | Publish simulated LED point cloud |
| `use_aruco` | `true` | Run ArUco marker detection |
| `use_docking_rviz` | `false` | Open RViz with docking config |
| `use_foxglove` | `false` | Start `foxglove_bridge` (WebSocket on port 8765) |
| `use_control` | `false` | Start the full docking stack: coarse + fine PBVS controllers and the docking FSM |
| `use_deadman` | `false` | Joystick deadman: relay `/cmd_vel_auto` → `/cmd_vel` only while the deadman button is held, and publish `/docking/engaged` to engage/disengage the FSM |
| `use_fsm_viewer` | `false` | Web FSM visualizer (YASMIN viewer) at `http://localhost:<fsm_viewer_port>` |
| `fsm_viewer_port` | `5000` | Port for the FSM visualizer |

E.g., launch without ArduSub for faster startup:

```bash
ros2 launch sim sim.launch.py use_ardusub:=false use_docking_rviz:=true
```

Run the full docking stack with joystick engage and the FSM viewer:

```bash
ros2 launch sim sim.launch.py use_control:=true use_joy:=true use_deadman:=true \
  use_fsm_viewer:=true use_docking_rviz:=true
```

> With `use_control:=true` the docking FSM owns the flight mode per phase (`POSHOLD` at idle, `ALT_HOLD` for coarse, `STABILIZE` for fine), so `flight_mode` only sets the *startup* mode — leave it at the `POSHOLD` default. Hold the joystick deadman button (default: RB / button 5) to engage autonomy; release to hand control back to manual `POSHOLD`.

### Dock motion

The docking station is not fixed: a Gazebo system plugin (`DockSway`) drives it on a prescribed sinusoidal heave/sway trajectory to emulate swell, so perception and control are exercised against a moving target. Amplitude, period, and phase are set on the `<plugin filename="DockSway">` block in `src/description/models/docking_station/model.sdf` (defaults: heave and sway of +/- 0.1 m, 4 s period). Set `<enabled>false</enabled>` there to freeze the dock for a static baseline.

### Foxglove

[Foxglove](https://foxglove.dev/) (free tier) or [Lichtblick](https://github.com/lichtblick-suite/lichtblick) gives a single window for live 3D, camera, plots, logs, and MCAP replay — sharing the same WebSocket protocol, so either viewer works.

```bash
ros2 launch sim sim.launch.py use_foxglove:=true
```

Connect the viewer to `ws://localhost:8765` (open connection -> **Foxglove WebSocket**; the dev container runs with `--network=host`), then open the layout `src/description/foxglove/docking.json` from the app.

## Related

- [alanchoi00/blue-sim](https://github.com/alanchoi00/blue-sim): Gazebo simulation base
- [Robotic-Decision-Making-Lab/blue](https://github.com/Robotic-Decision-Making-Lab/blue): upstream blue package
