/**
*  @file testGaussianProcessPriorPose3.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <gpmp2/gp/GaussianProcessPriorPose3.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(GaussianProcessPriorPose3Test, Factor) {

  const double delta_t = 0.1;
  gtsam::Matrix Qc = 0.01 * gtsam::Matrix::Identity(6,6);
  gtsam::noiseModel::Gaussian::shared_ptr Qc_model = gtsam::noiseModel::Gaussian::Covariance(Qc);
  gtsam::Key key_pose1 = gtsam::Symbol('x', 1), key_pose2 = gtsam::Symbol('x', 2);
  gtsam::Key key_vel1 = gtsam::Symbol('v', 1), key_vel2 = gtsam::Symbol('v', 2);
  GaussianProcessPriorPose3 factor(key_pose1, key_vel1, key_pose2, key_vel2, delta_t, Qc_model);
  gtsam::Pose3 p1, p2;
  gtsam::Vector6 v1, v2;
  gtsam::Matrix actualH1, actualH2, actualH3, actualH4;
  gtsam::Matrix expectH1, expectH2, expectH3, expectH4;
  gtsam::Vector actual, expect;


  // test at origin
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  v1 = (gtsam::Vector6() << 0, 0, 0, 0, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 0, 0, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, p2, v2, actualH1, actualH2, actualH3, actualH4);
  expect = (gtsam::Vector(12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-6));


  // test at const forward velocity v1 = v2 = 1.0;
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), gtsam::Point3(0.1, 0.0, 0.0));
  v1 = (gtsam::Vector6() << 0, 0, 0, 1, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 0, 1, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, p2, v2, actualH1, actualH2, actualH3, actualH4);
  expect = (gtsam::Vector(12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-6));


  // test at const rotation w1 = w2 = 1.0;
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(0.1, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
  v1 = (gtsam::Vector6() << 0, 0, 1, 0, 0, 0).finished();
  v2 = (gtsam::Vector6() << 0, 0, 1, 0, 0, 0).finished();
  actual = factor.evaluateError(p1, v1, p2, v2, actualH1, actualH2, actualH3, actualH4);
  expect = (gtsam::Vector(12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).finished();
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-6));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-6));


  // some random stuff just for testing jacobian (error is not zero)
  p1 = gtsam::Pose3(gtsam::Rot3::Ypr(-0.1, 1.2, 0.3), gtsam::Point3(-4.0, 2.0, 14.0));
  p2 = gtsam::Pose3(gtsam::Rot3::Ypr(2.4, -2.5, 3.7), gtsam::Point3(9.0, -8.0, -7.0));
  v1 = (gtsam::Vector6() << 2, 3, 1, 5, 4, 9).finished();
  v2 = (gtsam::Vector6() << 1, 3, 8, 0, 6, 4).finished();
  actual = factor.evaluateError(p1, v1, p2, v2, actualH1, actualH2, actualH3, actualH4);
  expectH1 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          _1, v1, p2, v2, boost::none, boost::none, boost::none, boost::none)), p1, 1e-6);
  expectH2 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, _1, p2, v2, boost::none, boost::none, boost::none, boost::none)), v1, 1e-6);
  expectH3 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Pose3&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, _1, v2, boost::none, boost::none, boost::none, boost::none)), p2, 1e-6);
  expectH4 = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector6&)>(
      boost::bind(&GaussianProcessPriorPose3::evaluateError, factor,
          p1, v1, p2, _1, boost::none, boost::none, boost::none, boost::none)), v2, 1e-6);
  EXPECT(gtsam::assert_equal(expectH1, actualH1, 1e-5));
  EXPECT(gtsam::assert_equal(expectH2, actualH2, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-6));
  EXPECT(gtsam::assert_equal(expectH3, actualH3, 1e-5));
  EXPECT(gtsam::assert_equal(expectH4, actualH4, 1e-6));
}

/* ************************************************************************** */
TEST(GaussianProcessPriorPose3Test, Optimization) {
  /**
   * A simple graph:
   *
   * p1   p2
   * |    |
   * x1   x2
   *  \  /
   *   gp
   *  /  \
   * v1  v2
   *
   * p1 and p2 are pose prior factor to fix the poses, gp is the GP factor
   * that get correct velocity of v2
   */

  gtsam::noiseModel::Isotropic::shared_ptr model_prior =
      gtsam::noiseModel::Isotropic::Sigma(6, 0.001);
  double delta_t = 1;
  gtsam::Matrix Qc = 0.01 * gtsam::Matrix::Identity(6,6);
  gtsam::noiseModel::Gaussian::shared_ptr Qc_model = gtsam::noiseModel::Gaussian::Covariance(Qc);

  gtsam::Pose3 pose1(gtsam::Rot3(), gtsam::Point3(0,0,0)), pose2(gtsam::Rot3(), gtsam::Point3(1,0,0));
  gtsam::Vector v1 = (gtsam::Vector(6) << 0, 0, 0, 1, 0, 0).finished();
  gtsam::Vector v2 = (gtsam::Vector(6) << 0.1, 0.2, -0.3, 2.0, -0.5, 0.6).finished();   // rnd value

  gtsam::NonlinearFactorGraph graph;
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 1), pose1, model_prior));
  graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 2), pose2, model_prior));
  //graph.add(PriorFactor<Vector6>(Symbol('v', 1), v1, model_prior));
  graph.add(GaussianProcessPriorPose3(gtsam::Symbol('x', 1), gtsam::Symbol('v', 1),
      gtsam::Symbol('x', 2), gtsam::Symbol('v', 2), delta_t, Qc_model));

  gtsam::Values init_values;
  init_values.insert(gtsam::Symbol('x', 1), pose1);
  init_values.insert(gtsam::Symbol('v', 1), v1);
  init_values.insert(gtsam::Symbol('x', 2), pose2);
  init_values.insert(gtsam::Symbol('v', 2), v2);

  gtsam::GaussNewtonParams parameters;
  gtsam::GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values values = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(values), 1e-6);
  EXPECT(gtsam::assert_equal(pose1, values.at<gtsam::Pose3>(gtsam::Symbol('x', 1)), 1e-6));
  EXPECT(gtsam::assert_equal(pose2, values.at<gtsam::Pose3>(gtsam::Symbol('x', 2)), 1e-6));
  EXPECT(gtsam::assert_equal(v1, values.at<gtsam::Vector>(gtsam::Symbol('v', 1)), 1e-6));
  EXPECT(gtsam::assert_equal(v1, values.at<gtsam::Vector>(gtsam::Symbol('v', 2)), 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
