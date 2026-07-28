#include "dock_sway/DockSway.hh"

#include <chrono>
#include <cstdlib>
#include <string>
#include <gz/plugin/Register.hh>
#include <gz/sim/Util.hh>
#include <gz/common/Console.hh>

namespace {
// Env-var overrides let a batch sweep set the dock motion per launch without a
// rebuild or SDF templating: the runner exports DOCK_SWAY_* and gazebo (a child of
// ros2 launch) inherits them. Empty/unset/unparseable -> the SDF value is kept.
double envOr(const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}
bool envOrBool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    const std::string s(v);
    if (s == "1" || s == "true" || s == "True" || s == "TRUE") return true;
    if (s == "0" || s == "false" || s == "False" || s == "FALSE") return false;
    return fallback;
}
}  // namespace

namespace dock_sway {
void DockSway::Configure(
    const gz::sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    gz::sim::EntityComponentManager &ecm,
    gz::sim::EventManager &eventMgr
) {
    this->_model = gz::sim::Model(entity);
    if (!this->_model.Valid(ecm)) {
        gzerr << "DockSway: not attached to a model, disabling.\n";
        return;
    }

    this->_params.enabled =
        envOrBool("DOCK_SWAY_ENABLED", sdf->Get<bool>("enabled", true).first);
    this->_params.period =
        envOr("DOCK_SWAY_PERIOD", sdf->Get<double>("period", 4.0).first);
    this->_params.heave_amplitude = envOr(
        "DOCK_SWAY_HEAVE_AMPLITUDE", sdf->Get<double>("heave_amplitude", 0.1).first);
    this->_params.sway_amplitude = envOr(
        "DOCK_SWAY_SWAY_AMPLITUDE", sdf->Get<double>("sway_amplitude", 0.1).first);
    this->_params.heave_phase =
        envOr("DOCK_SWAY_HEAVE_PHASE", sdf->Get<double>("heave_phase", 0.0).first);
    this->_params.sway_phase =
        envOr("DOCK_SWAY_SWAY_PHASE", sdf->Get<double>("sway_phase", 1.5708).first);

    this->_home_pose = gz::sim::worldPose(entity, ecm);
    this->_home_yaw = this->_home_pose.Rot().Yaw();

    this->_valid = true;
    gzmsg << "DockSway: configured, enabled=" << this->_params.enabled
          << " period=" << this->_params.period << "s"
          << " sway_amp=" << this->_params.sway_amplitude
          << " heave_amp=" << this->_params.heave_amplitude
          << " sway_phase=" << this->_params.sway_phase << "\n";
}

void DockSway::PreUpdate(
    const gz::sim::UpdateInfo &info,
    gz::sim::EntityComponentManager &ecm
) {
    if (!this->_valid || info.paused) {
        return;
    }

    const double t = std::chrono::duration<double>(info.simTime).count();

    auto offset = WorldOffset(t, this->_params, this->_home_yaw);

    auto target = gz::math::Pose3d(this->_home_pose.Pos() + offset, this->_home_pose.Rot());

    this->_model.SetWorldPoseCmd(ecm, target);
}
} // namespace dock_sway

GZ_ADD_PLUGIN(
    dock_sway::DockSway,
    gz::sim::System,
    dock_sway::DockSway::ISystemConfigure,
    dock_sway::DockSway::ISystemPreUpdate
)

GZ_ADD_PLUGIN_ALIAS(dock_sway::DockSway, "dock::DockSway")
