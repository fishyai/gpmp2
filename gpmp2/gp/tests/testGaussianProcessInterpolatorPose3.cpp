/**
*  @file testGaussianProcessInterpolatorPose3.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/gp/GaussianProcessInterpolatorPose3.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(GaussianProcessInterpolatorPose3, interpolatePose) {
  gtsam::Pose3 p1, p2, expect, actual;
  gtsam::Vector6 v1, v2;
  gtsam::Matrix actualH1, actualH2, actualH3, actualH4;
  gtsam::Matrix expectH1, expectH2, expectH3, expectH4;
  gtsam::Matrix6 Qc = 0.01 * gtsam::Matrix::Identity(6,6);
  gtsam::noiseModel::Gaussian::shared_ptr Qc_model = gtsam::noiseModel::Gaussian::Covariance(Qc);
  double dt = 0.1, tau = 0.03;
  GaussianProcessInterpolatorPose3 base(Qc_model, dt, tau);

  // test at origin
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  v1 = (gtsam::Vector6() << 0, 0, 0, 0, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 0, 0, 0, 0).finished();
  actual = base.interpolatePose(p1, v1, p2, v2, actualH1, actualH2,
      actualH3, actualH4);
  expect = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-8));


  // test forward
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0.1, 0, 0));
  v1 = (gtsam::Vector6() << 0, 0, 0, 1, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 0, 1, 0, 0).finished();
  actual = base.interpolatePose(p1, v1, p2, v2, actualH1, actualH2,
      actualH3, actualH4);
  expect = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0.03, 0, 0));
  expectH1 = numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-4);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-4);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-4);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-4);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-6));


  // test rotate
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0, 0, 0), gtsam::Point3(0, 0, 0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0.1, 0, 0), gtsam::Point3(0, 0, 0));
  v1 = (gtsam::Vector6() << 0, 0, 1, 0, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 1, 0, 0, 0).finished();
  actual = base.interpolatePose(p1, v1, p2, v2, actualH1, actualH2,
      actualH3, actualH4);
  expect = gtsam::Pose3(gtsam::Rot3::Ypr(0.03, 0, 0), gtsam::Point3(0.0, 0, 0));
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-8));


  // some random stuff, just test jacobians
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0.4, -0.8, 0.2), gtsam::Point3(3, -8, 2));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0.1, 0.3, -0.5), gtsam::Point3(-9, 3, 4));
  v1 = (gtsam::Vector6() << 0.1, -0.2, -1.4, 0.5, 0.9, 0.7).finished();
  v2 = (gtsam::Vector6() << 0.6, 0.3, -0.9, 0.4, -0.2, 0.8).finished();
  actual = base.interpolatePose(p1, v1, p2, v2, actualH1, actualH2,
      actualH3, actualH4);
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessInterpolatorPose3::interpolatePose, base,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-8));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-8));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-8));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-8));

}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
