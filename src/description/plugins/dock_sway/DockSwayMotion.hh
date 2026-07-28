#include <cmath>
#include <gz/math/Vector3.hh>
#include <gz/math/Quaternion.hh>

namespace dock_sway {

struct SwayParams {
    bool enabled;
    double period;
    double heave_amplitude;
    double heave_phase;
    double sway_amplitude;
    double sway_phase;
};

/**
 * Returns the dock local translation offset at time `t` seconds.
 */
gz::math::Vector3d LocalOffset(double t, const SwayParams &p) {
    if (!p.enabled || (p.heave_amplitude == 0.0  && p.sway_amplitude == 0.0) || p.period <= 0.0) {
        return gz::math::Vector3d::Zero;
    }

    auto sinWave = [t, p](double A, double phi) -> double {
        return A * std::sin(2*M_PI*t/p.period + phi);
    };

    return gz::math::Vector3d(sinWave(p.sway_amplitude, p.sway_phase), 0.0, sinWave(p.heave_amplitude, p.heave_phase));
}

/**
 * Rotates LocalOffset about world Z by yaw, returns the world translation to the home position
 */
gz::math::Vector3d WorldOffset(double t, const SwayParams &p, double yaw) {
    auto localOffset = LocalOffset(t, p);
    gz::math::Quaterniond q(gz::math::Vector3d(0.0, 0.0, yaw));
    return q.RotateVector(localOffset);
}

} // namespace dock_sway


