/**
*  @file testGoalFactorArm.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <gpmp2/kinematics/GoalFactorArm.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(GoalFactorArm, error) {

  // settings
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(3, 1.0);

  // 2 link simple example
  gtsam::Vector2 a(1, 1), alpha(0, 0), d(0, 0);
  Arm arm(2, a, alpha, d);
  gtsam::Vector2 q;
  gtsam::Point3 goal;
  GoalFactorArm factor;
  gtsam::Vector actual, expect;
  gtsam::Matrix H_exp, H_act;

  // zero
  q = gtsam::Vector2(0, 0);
  goal = gtsam::Point3(2, 0, 0);
  factor = GoalFactorArm(0, cost_model, arm, goal);
  actual = factor.evaluateError(q, H_act);
  expect = gtsam::Vector3(0, 0, 0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&GoalFactorArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));

  // 45 deg
  q = gtsam::Vector2(M_PI/4.0, 0);
  goal = gtsam::Point3(1.414213562373095, 1.414213562373095, 0);
  factor = GoalFactorArm(0, cost_model, arm, goal);
  actual = factor.evaluateError(q, H_act);
  expect = gtsam::Vector3(0, 0, 0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&GoalFactorArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));

  // non zero error
  q = gtsam::Vector2(M_PI/4.0, 0);
  goal = gtsam::Point3(2, 0, 0);
  factor = GoalFactorArm(0, cost_model, arm, goal);
  actual = factor.evaluateError(q, H_act);
  expect = gtsam::Vector3(-0.585786437626905, 1.414213562373095, 0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&GoalFactorArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
}


/* ************************************************************************** */
TEST(GoalFactorArm, optimization_1) {

  // use optimization to solve inverse kinematics
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);

  // 2 link simple example
  gtsam::Vector a = (gtsam::Vector(2) << 1, 1).finished();
  gtsam::Vector alpha = (gtsam::Vector(2) << 0, 0).finished();
  gtsam::Vector d = (gtsam::Vector(2) << 0, 0).finished();
  Arm arm(2, a, alpha, d);
  gtsam::Point3 goal(1.414213562373095, 1.414213562373095, 0);

  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector q = (gtsam::Vector(2) << M_PI/4.0, 0).finished();
  gtsam::Vector qinit = (gtsam::Vector(2) << 0, 0).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(GoalFactorArm(qkey, cost_model, arm, goal));
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


