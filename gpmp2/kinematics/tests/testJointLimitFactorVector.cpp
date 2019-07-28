/**
*  @file testJointLimitFactorVector.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <gpmp2/kinematics/JointLimitFactorVector.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(JointLimitFactorVector, error) {

  // settings
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

  // 2 link simple example
  gtsam::Vector2 dlimit(-5.0, -10.0), ulimit(5, 10.0);
  gtsam::Vector2 thresh(2.0, 2.0);
  JointLimitFactorVector factor(0, cost_model, dlimit, ulimit, thresh);
  gtsam::Vector2 conf;
  gtsam::Vector actual, expect;
  gtsam::Matrix H_exp, H_act;

  // zero
  conf = gtsam::Vector2(0.0, 0.0);
  actual = factor.evaluateError(conf, H_act);
  expect = gtsam::Vector2(0.0, 0.0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector2(const gtsam::Vector2&)>(
      boost::bind(&JointLimitFactorVector::evaluateError, factor, _1, boost::none)), conf, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));

  // over down limit
  conf = gtsam::Vector2(-10.0, -10.0);
  actual = factor.evaluateError(conf, H_act);
  expect = gtsam::Vector2(7.0, 2.0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector2(const gtsam::Vector2&)>(
    boost::bind(&JointLimitFactorVector::evaluateError, factor, _1, boost::none)), conf, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));

  // over up limit
  conf = gtsam::Vector2(10.0, 10.0);
  actual = factor.evaluateError(conf, H_act);
  expect = gtsam::Vector2(7.0, 2.0);
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector2(const gtsam::Vector2&)>(
    boost::bind(&JointLimitFactorVector::evaluateError, factor, _1, boost::none)), conf, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
}

/* ************************************************************************** */
TEST(JointLimitFactorVector, optimization_1) {
  // zero point

  // settings
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(2, 0.001);
  gtsam::noiseModel::Gaussian::shared_ptr prior_model = gtsam::noiseModel::Isotropic::Sigma(2, 1000);
  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector2 dlimit(-5.0, -10.0), ulimit(5, 10.0);
  gtsam::Vector2 thresh(2.0, 2.0);

  gtsam::Vector conf;
  conf = (gtsam::Vector(2) << 0.0, 0.0).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(JointLimitFactorVector(qkey, cost_model, dlimit, ulimit, thresh));
  graph.add(gtsam::PriorFactor<gtsam::Vector>(qkey, conf, prior_model));
  gtsam::Values init_values;
  init_values.insert(qkey, conf);

  gtsam::GaussNewtonParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setAbsoluteErrorTol(1e-12);
  gtsam::GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values results = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(results), 1e-6);
  EXPECT(gtsam::assert_equal(conf, results.at<gtsam::Vector>(qkey), 1e-6));
}

/* ************************************************************************** */
TEST(JointLimitFactorVector, optimization_2) {
  // over down limit

  // settings
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(2, 0.001);
  gtsam::noiseModel::Gaussian::shared_ptr prior_model = gtsam::noiseModel::Isotropic::Sigma(2, 1000);
  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector2 dlimit(-5.0, -10.0), ulimit(5, 10.0);
  gtsam::Vector2 thresh(2.0, 2.0);

  gtsam::Vector conf;
  conf = (gtsam::Vector(2) << -10.0, -10.0).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(JointLimitFactorVector(qkey, cost_model, dlimit, ulimit, thresh));
  graph.add(gtsam::PriorFactor<gtsam::Vector>(qkey, conf, prior_model));
  gtsam::Values init_values;
  init_values.insert(qkey, conf);

  gtsam::GaussNewtonParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setAbsoluteErrorTol(1e-12);
  gtsam::GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values results = optimizer.values();

  gtsam::Vector conf_limit = (gtsam::Vector(2) << -3.0, -8.0).finished();
  EXPECT(gtsam::assert_equal(conf_limit, results.at<gtsam::Vector>(qkey), 1e-6));
}

/* ************************************************************************** */
TEST(JointLimitFactorVector, optimization_3) {
  // over up limit

  // settings
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(2, 0.001);
  gtsam::noiseModel::Gaussian::shared_ptr prior_model = gtsam::noiseModel::Isotropic::Sigma(2, 1000);
  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector2 dlimit(-5.0, -10.0), ulimit(5, 10.0);
  gtsam::Vector2 thresh(2.0, 2.0);

  gtsam::Vector conf;
  conf = (gtsam::Vector(2) << 10.0, 10.0).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(JointLimitFactorVector(qkey, cost_model, dlimit, ulimit, thresh));
  graph.add(gtsam::PriorFactor<gtsam::Vector>(qkey, conf, prior_model));
  gtsam::Values init_values;
  init_values.insert(qkey, conf);

  gtsam::GaussNewtonParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setAbsoluteErrorTol(1e-12);
  gtsam::GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values results = optimizer.values();

  gtsam::Vector conf_limit = (gtsam::Vector(2) << 3.0, 8.0).finished();
  EXPECT(gtsam::assert_equal(conf_limit, results.at<gtsam::Vector>(qkey), 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}


