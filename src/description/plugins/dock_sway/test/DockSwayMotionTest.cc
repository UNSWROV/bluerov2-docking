#include <cmath>
#include <gtest/gtest.h>
#include <gz/math/Vector3.hh>
#include "dock_sway/DockSwayMotion.hh"

using dock_sway::SwayParams;
using dock_sway:: LocalOffset;
using dock_sway::WorldOffset;

namespace {
    constexpr double kA = 0.1;
    constexpr double kT = 4.0;
    SwayParams DefaultParams() {
        return SwayParams{true, kT, kA, 0.0, kA, M_PI_2};
    }
}

TEST(DockSwayMotion, ZeroHeaveAtRestStart) {
    // heave uses phase 0, so sin(0) = 0 at t = 0.
    auto o = LocalOffset(0.0, DefaultParams());
    EXPECT_NEAR(o.Z(), 0.0, 1e-9);
}

TEST(DockSwayMotion, SwayAtAmplitudeAtStart) {
  // sway uses phase pi/2, so sin(pi/2) = 1 at t = 0 -> full amplitude on local X.
  auto o = LocalOffset(0.0, DefaultParams());
  EXPECT_NEAR(o.X(), kA, 1e-9);
  EXPECT_NEAR(o.Y(), 0.0, 1e-9);
}

TEST(DockSwayMotion, AmplitudeBoundedOverPeriod) {
  auto p = DefaultParams();
  for (int i = 0; i <= 400; ++i) {
    double t = kT * i / 400.0;
    auto o = LocalOffset(t, p);
    EXPECT_LE(std::abs(o.X()), kA + 1e-9);
    EXPECT_LE(std::abs(o.Z()), kA + 1e-9);
  }
}

TEST(DockSwayMotion, Periodic) {
  auto p = DefaultParams();
  auto a = LocalOffset(0.7, p);
  auto b = LocalOffset(0.7 + kT, p);
  EXPECT_NEAR(a.X(), b.X(), 1e-6);
  EXPECT_NEAR(a.Z(), b.Z(), 1e-6);
}

TEST(DockSwayMotion, PeakSpeedMatchesFormula) {
  // Numerical derivative peak of a single axis should approach A * 2*pi / T.
  auto p = DefaultParams();
  p.sway_phase = 0.0;  // isolate a clean sine on X for the derivative
  const double dt = 1e-4;
  double vmax = 0.0;
  for (int i = 0; i < static_cast<int>(kT / dt); ++i) {
    double t = i * dt;
    double dx = (LocalOffset(t + dt, p).X() - LocalOffset(t, p).X()) / dt;
    vmax = std::max(vmax, std::abs(dx));
  }
  EXPECT_NEAR(vmax, kA * 2.0 * M_PI / kT, 1e-3);
}

TEST(DockSwayMotion, DisabledIsFrozen) {
  auto p = DefaultParams();
  p.enabled = false;
  auto o = LocalOffset(1.3, p);
  EXPECT_EQ(o, gz::math::Vector3d::Zero);
}

TEST(DockSwayMotion, ZeroAmplitudeIsFrozen) {
  SwayParams p{true, kT, 0.0, 0.0, 0.0, 0.0};
  EXPECT_EQ(LocalOffset(2.1, p), gz::math::Vector3d::Zero);
}

TEST(DockSwayMotion, WorldFrameRotatesByYaw) {
  // At t = 0 the local offset is (kA, 0, 0). Under yaw 270 deg (pointing -Y world),
  // a local +X maps to world -Y (rotation about world Z by 3*pi/2).
  auto p = DefaultParams();
  const double yaw = 3.0 * M_PI_2;  // 270 deg, matches ocean.world dock yaw ~4.712
  auto w = WorldOffset(0.0, p, yaw);
  EXPECT_NEAR(w.X(), 0.0, 1e-6);
  EXPECT_NEAR(w.Y(), -kA, 1e-6);
  EXPECT_NEAR(w.Z(), 0.0, 1e-9);
}