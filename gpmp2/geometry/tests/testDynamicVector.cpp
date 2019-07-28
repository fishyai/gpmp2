/**
*  @file testDynamicVector.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gpmp2/geometry/DynamicVector.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Testable.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


GTSAM_CONCEPT_TESTABLE_INST(DynamicVector)
GTSAM_CONCEPT_LIE_INST(DynamicVector)

/* ************************************************************************** */
TEST(DynamicVector, Concept) {
  BOOST_CONCEPT_ASSERT((gtsam::IsGroup<DynamicVector>));
  BOOST_CONCEPT_ASSERT((gtsam::IsManifold<DynamicVector>));
  BOOST_CONCEPT_ASSERT((gtsam::IsLieGroup<DynamicVector>));
  BOOST_CONCEPT_ASSERT((gtsam::IsVectorSpace<DynamicVector>));
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, contructor) {
  gtsam::Vector a(3);
  a << 1,2,3;
  DynamicVector v(a);

  EXPECT_LONGS_EQUAL(3, v.dim());
  EXPECT(gtsam::assert_equal(a, v.vector(), 1e-9));
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, access_element) {
  gtsam::Vector a(3);
  a << 1,2,3;
  DynamicVector v(a);

  for (size_t i = 1; i < 3; i++)
    EXPECT_DOUBLES_EQUAL(a(i), v(i), 1e-9);
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, equals) {
  gtsam::Vector a(3), b(3);
  a << 1,2,3;
  b << 4,5,6;
  DynamicVector v1(a), v2(a), v3(b);

  EXPECT(gtsam::assert_equal(v1, v2, 1e-9));
  EXPECT(gtsam::assert_equal(v2, v1, 1e-9));
  EXPECT(!gtsam::assert_equal(v1, v3, 1e-9));
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, addition) {
  gtsam::Vector a(3), b(3), c(3);
  a << 1,2,3;
  b << 4,5,6;
  c << 5,7,9;
  DynamicVector v1(a), v2(b), v3(c);

  EXPECT(gtsam::assert_equal(v3, v1 + v2, 1e-9));
  EXPECT(gtsam::assert_equal(v3, v2 + v1, 1e-9));
  EXPECT(gtsam::assert_equal(v3, v1 + b, 1e-9));
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, subtraction) {
  gtsam::Vector a(3), b(3), c(3);
  a << 1,2,3;
  b << 4,5,6;
  c << 5,7,9;
  DynamicVector v1(a), v2(b), v3(c);

  EXPECT(gtsam::assert_equal(v1, v3 - v2, 1e-9));
  EXPECT(gtsam::assert_equal(v2, v3 - v1, 1e-9));
}

/* ************************************************************************** */
TEST_UNSAFE(DynamicVector, optimization) {

  gtsam::Vector a(3), b(3), c(3);
  a << 1,2,3;
  b << 4,5,6;
  DynamicVector v1(a), v2(b);

  // prior factor graph
  gtsam::noiseModel::Isotropic::shared_ptr model_prior =
      gtsam::noiseModel::Isotropic::Sigma(3, 0.001);
  gtsam::NonlinearFactorGraph graph;
  graph.add(gtsam::PriorFactor<DynamicVector>(gtsam::Symbol('x', 1), v1, model_prior));

  // init values
  gtsam::Values init_values;
  init_values.insert(gtsam::Symbol('x', 1), v2);

  // optimize!
  gtsam::GaussNewtonParams parameters;
  gtsam::GaussNewtonOptimizer optimizer(graph, init_values, parameters);
  optimizer.optimize();
  gtsam::Values values = optimizer.values();

  EXPECT_DOUBLES_EQUAL(0, graph.error(values), 1e-6);
  EXPECT(gtsam::assert_equal(v1, values.at<DynamicVector>(gtsam::Symbol('x', 1)), 1e-9));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
