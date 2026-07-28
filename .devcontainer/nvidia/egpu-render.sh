# Route containerized GL apps (gazebo, rviz2) onto the NVIDIA eGPU via PRIME offload,
# but ONLY when the topology actually calls for it.
#
# SOURCE this in the terminal you launch the sim from (do not execute it):
#   source .devcontainer/nvidia/egpu-render.sh        # offload ON if (and only if) needed
#   source .devcontainer/nvidia/egpu-render.sh off    # force host/default GL path
#
# The vars only affect processes started from the shell that sourced this, which is
# why executing the file (a subshell) would do nothing.
#
# Topology note: PRIME offload is only meaningful when X's PRIMARY is the Intel iGPU
# and the NVIDIA GPU shows up as a render-offload provider named NVIDIA-G<n>. When X
# runs ON the eGPU instead (primary provider NVIDIA-0, via AllowExternalGpus), there
# is no NVIDIA-G<n> provider and plain GL already renders on the eGPU. Setting the
# offload vars then points __NV_PRIME_RENDER_OFFLOAD_PROVIDER at a nonexistent
# provider and every GL app dies with "NV-GLX BadWindow". So on ON we read the actual
# provider from `xrandr --listproviders` and refuse to set the vars when there is
# nothing to offload to, instead of hard-coding NVIDIA-G1.

if [ "${1:-on}" = "off" ]; then
  unset __NV_PRIME_RENDER_OFFLOAD __NV_PRIME_RENDER_OFFLOAD_PROVIDER \
        __GLX_VENDOR_LIBRARY_NAME __VK_LAYER_NV_optimus
  echo "eGPU render offload OFF (host default GL path)"
elif ! command -v xrandr >/dev/null 2>&1; then
  echo "egpu-render: xrandr not found; cannot verify topology."
  echo "  Leaving GL vars unset (default GL path). Install x11-xserver-utils to enable auto-detect."
else
  _egpu_provider=$(xrandr --listproviders 2>/dev/null | grep -oE 'NVIDIA-G[0-9]+' | head -n1)
  if [ -n "$_egpu_provider" ]; then
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __NV_PRIME_RENDER_OFFLOAD_PROVIDER="$_egpu_provider"
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export __VK_LAYER_NV_optimus=NVIDIA_only
    echo "eGPU render offload ON (provider $_egpu_provider, auto-detected)"
  else
    unset __NV_PRIME_RENDER_OFFLOAD __NV_PRIME_RENDER_OFFLOAD_PROVIDER \
          __GLX_VENDOR_LIBRARY_NAME __VK_LAYER_NV_optimus
    echo "egpu-render: no NVIDIA-G<n> offload provider present."
    echo "  X is running ON the eGPU already (primary NVIDIA-0); plain GL renders there."
    echo "  Leaving GL vars unset -- no offload needed."
  fi
  unset _egpu_provider
fi
