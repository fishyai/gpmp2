/**
*  @file    testGaussianPriorWorkspacePose.cpp
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

#include <gpmp2/kinematics/GaussianPriorWorkspacePoseArm.h>
#include <gpmp2/kinematics/ArmModel.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(GaussianPriorWorkspacePoseArm, error) {

  gtsam::Vector2 a(1, 1), alpha(M_PI/2, 0), d(0, 0);
  ArmModel arm = ArmModel(Arm(2, a, alpha, d), BodySphereVector());
  gtsam::Vector2 q;
  gtsam::Pose3 des;
  gtsam::Vector actual, expect;
  gtsam::Matrix H_exp, H_act;
  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(6, 1.0);

  q = gtsam::Vector2(M_PI/4.0, -M_PI/2);
  des = gtsam::Pose3();
  GaussianPriorWorkspacePoseArm factor(0, arm, 1, des, cost_model);
  actual = factor.evaluateError(q, H_act);
  expect = (gtsam::Vector(6) << 0.613943126, 1.48218982, -0.613943126, 1.1609828, 0.706727485, -0.547039678).finished();
  H_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector2&)>(
      boost::bind(&GaussianPriorWorkspacePoseArm::evaluateError, factor, _1, boost::none)), q, 1e-6);
  EXPECT(gtsam::assert_equal(expect, actual, 1e-6));
  EXPECT(gtsam::assert_equal(H_exp, H_act, 1e-6));
}


/* ************************************************************************** */
TEST(GaussianPriorWorkspacePoseArm, optimization) {

  gtsam::noiseModel::Gaussian::shared_ptr cost_model = gtsam::noiseModel::Isotropic::Sigma(6, 0.1);

  gtsam::Vector a = (gtsam::Vector(2) << 1, 1).finished();
  gtsam::Vector alpha = (gtsam::Vector(2) << 0, 0).finished();
  gtsam::Vector d = (gtsam::Vector(2) << 0, 0).finished();
  ArmModel arm = ArmModel(Arm(2, a, alpha, d), BodySphereVector());
  gtsam::Pose3 des = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(2, 0, 0));

  gtsam::Key qkey = gtsam::Symbol('x', 0);
  gtsam::Vector q = (gtsam::Vector(2) << 0, 0).finished();
  gtsam::Vector qinit = (gtsam::Vector(2) << M_PI/2, M_PI/2).finished();

  gtsam::NonlinearFactorGraph graph;
  graph.add(GaussianPriorWorkspacePoseArm(qkey, arm, 1, des, cost_model));
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
