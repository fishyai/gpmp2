/**
*  @file testTrajUtils.cpp
*  @author Jing Dong
*  @date May 30, 2016
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <gpmp2/planner/TrajUtils.h>
#include <gpmp2/geometry/Pose2Vector.h>

#include <iostream>


using namespace std;
using namespace gpmp2;



/* ************************************************************************** */
TEST(TrajUtils, interpolateLinear) {

  gtsam::Matrix Qc = 0.01 * gtsam::Matrix::Identity(2,2);
  gtsam::noiseModel::Gaussian::shared_ptr Qc_model = gtsam::noiseModel::Gaussian::Covariance(Qc);

  // values to be interpolated
  gtsam::Values init_values;
  // const velocity vb = 10, dt = 0.1
  init_values.insert(gtsam::Symbol('x', 0), (gtsam::Vector(2) << 0, 0).finished());
  init_values.insert(gtsam::Symbol('x', 1), (gtsam::Vector(2) << 1, 0).finished());
  init_values.insert(gtsam::Symbol('v', 0), (gtsam::Vector(2) << 10, 0).finished());
  init_values.insert(gtsam::Symbol('v', 1), (gtsam::Vector(2) << 10, 0).finished());

  // interpolate 5 interval in the middle
  gtsam::Values inter_values = interpolateArmTraj(init_values, Qc_model, 0.1, 4);

  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 0, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 0)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 0.2, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 1)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 0.4, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 2)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 0.6, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 3)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 0.8, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 4)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 1, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('x', 5)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 0)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 1)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 2)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 3)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 4)), 1e-6));
  EXPECT(gtsam::assert_equal((gtsam::Vector(2) << 10, 0).finished(), inter_values.at<gtsam::Vector>(gtsam::Symbol('v', 5)), 1e-6));
}

/* ************************************************************************** */
TEST(TrajUtils, initPose2VectorTrajStraightLine_1) {
  gtsam::Values init_values = initPose2VectorTrajStraightLine(
      gtsam::Pose2(1,3,M_PI-0.5), (gtsam::Vector(2) << 2,4).finished(),
      gtsam::Pose2(3,7,-M_PI+0.5), (gtsam::Vector(2) << 4,8).finished(), 5);

  EXPECT(gtsam::assert_equal(Pose2Vector(gtsam::Pose2(1,3,M_PI-0.5), (gtsam::Vector(2) << 2,4).finished()),
      init_values.at<Pose2Vector>(gtsam::Symbol('x', 0)), 1e-6));
  //EXPECT(gtsam::assert_equal(Pose2Vector(gtsam::Pose2(1.4,3.8,M_PI-0.3), (gtsam::Vector(2) << 2.4,4.8).finished()),
  //    init_values.at<Pose2Vector>(gtsam::Symbol('x', 1)), 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
