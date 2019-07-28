/**
*  @file    testGaussianPriorWorkspacePosition.cpp
*  @author  Mustafa Mukadam
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>

#include <gpmp2/kinematics/GaussianPriorWorkspacePositionArm.h>
#include <gpmp2/kinematics/ArmModel.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(GaussianPriorWorkspacePositionArm, error) {

  gtsam::Vector2 a(1, 1), alpha(0, 0), d(0, 0);
  ArmModel arm = ArmModel(Arm(2, a, alpha, d), BodySphereVector());
  gtsam::Vector2 q;
  gtsam::Point3 des_position;
  gtsam::Vector actual, expect;
  gtsam::Matrix H_exp, H_act;
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  // zero
  {
    q = gtsam::Vector2(0, 0);
    des_position = gtsam::Point3(2, 0, 0);
    GaussianPriorWorkspacePositionArm factor(0, arm, 1, des_position, cost_model);
    actual = factor.evaluateError(q, H_act);
    expect = gtsam::Vector3(0, 0, 0);
    H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
        boost::bind(&GaussianPriorWorkspacePositionArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
    EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
    EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
  }

  // 45 deg
  {
    q = gtsam::Vector2(M_PI/4.0, 0);
    des_position = gtsam::Point3(1.414213562373095, 1.414213562373095, 0);
    GaussianPriorWorkspacePositionArm factor(0, arm, 1, des_position, cost_model);
    actual = factor.evaluateError(q, H_act);
    expect = gtsam::Vector3(0, 0, 0);
    H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
        boost::bind(&GaussianPriorWorkspacePositionArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
    EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
    EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
  }

  // non zero error
  {
    q = gtsam::Vector2(M_PI/4.0, 0);
    des_position = gtsam::Point3(2, 0, 0);
    GaussianPriorWorkspacePositionArm factor(0, arm, 1, des_position, cost_model);
    actual = factor.evaluateError(q, H_act);
    expect = gtsam::Vector3(-0.585786437626905, 1.414213562373095, 0);
    H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
        boost::bind(&GaussianPriorWorkspacePositionArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
    EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
    EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
  }
}


/* ************************************************************************** */
TEST(GaussianPriorWorkspacePositionArm, optimization) {

  // use optimization to solve inverse kinematics
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);

  gtsam::Vector a = (gtsam::Vector(2) << 1, 1).finished();
  gtsam::Vector alpha = (gtsam::Vector(2) << 0, 0).finished();
  gtsam::Vector d = (gtsam::Vector(2) << 0, 0).finished();
  ArmModel arm = ArmModel(Arm(2, a, alpha, d), BodySphereVector());
  gtsam::Point3 des_position(1.414213562373095, 1.414213562373095, 0);

  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector q = (gtsam::Vector(2) << M_PI/4.0, 0).finished();
  gtsam::Vector qinit = (gtsam::Vector(2) << 0, 0).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(GaussianPriorWorkspacePositionArm(qkey, arm, 1, des_position, cost_model));
  gtsam::Values init_values;
  init_values.insert(qkey, qinit);

  gtsam::LevenbergMarquardtParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setAbsoluteErrorTol(1e-12);
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values results = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(results), 1e-3);
  EXPECT(gtsam::assert_equal(q, results.at<gtsam::Vector>(qkey), 1e-3));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
