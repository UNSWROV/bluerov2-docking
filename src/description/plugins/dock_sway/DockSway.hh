#pragma once

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/math/Pose3.hh>
#include "dock_sway/DockSwayMotion.hh"

namespace dock_sway {

class DockSway
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate {
public:
    void Configure(const gz::sim::Entity &entity, const std::shared_ptr<const sdf::Element> &sdf, gz::sim::EntityComponentManager &ecm, gz::sim::EventManager &eventMgr) override;
    void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override;
private:
    gz::sim::Model _model{gz::sim::kNullEntity};
    SwayParams _params{};
    gz::math::Pose3d _home_pose{};
    double _home_yaw{};
    bool _valid{false};
};
} // namespace dock_sway
