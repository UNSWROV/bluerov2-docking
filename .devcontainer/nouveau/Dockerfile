FROM ghcr.io/unswrov/blue-sim:jazzy-desktop

ENV USER_WORKSPACE=/home/ubuntu/ws_docking
WORKDIR $USER_WORKSPACE

# Install debugging/linting tools
RUN /home/ubuntu/.venv/blue/bin/pip install \
    pre-commit \
    mypy

USER root
RUN apt-get update && apt-get install -y \
    ros-jazzy-aruco-ros \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-rqt-common-plugins \
    ros-jazzy-yasmin \
    ros-jazzy-yasmin-ros \
    ros-jazzy-yasmin-viewer \
    && rm -rf /var/lib/apt/lists/*
USER ubuntu

ENV PYTHONWARNINGS="ignore"
