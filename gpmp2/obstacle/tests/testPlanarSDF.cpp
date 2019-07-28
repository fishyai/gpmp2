/**
*  @file testPlanarSDFutils.cpp
*  @author Jing Dong
*  @date May 8 2016
**/

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/base/Matrix.h>

#include <gpmp2/obstacle/PlanarSDF.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


double sdf_wrapper(const PlanarSDF& field, const gtsam::Point2& p) {
  return field.getSignedDistance(p);
}

/* ************************************************************************** */
TEST(PlanarSDFutils, test1) {
  // data
  gtsam::Matrix data;
  data = (gtsam::Matrix(5, 5) <<
      1.7321, 1.4142, 1.4142, 1.4142, 1.7321,
      1.4142, 1, 1, 1, 1.4142,
      1.4142, 1, 1, 1, 1.4142,
      1.4142, 1, 1, 1, 1.4142,
      1.7321, 1.4142, 1.4142, 1.4142, 1.7321).finished();
  gtsam::Point2 origin(-0.2, -0.2);
  double cell_size = 0.1;

  // constructor
  PlanarSDF field(origin, cell_size, data);
  EXPECT_LONGS_EQUAL(5, field.x_count());
  EXPECT_LONGS_EQUAL(5, field.y_count());
  EXPECT_DOUBLES_EQUAL(0.1, field.cell_size(), 1e-9);
  EXPECT(assert_equal(origin, field.origin()));

  // access
  PlanarSDF::float_index idx;
  idx = field.convertPoint2toCell(gtsam::Point2(0, 0));
  EXPECT_DOUBLES_EQUAL(2, idx.get<0>(), 1e-9);
  EXPECT_DOUBLES_EQUAL(2, idx.get<1>(), 1e-9);
  EXPECT_DOUBLES_EQUAL(1, field.signed_distance(idx), 1e-9)

  idx = field.convertPoint2toCell(gtsam::Point2(0.18, -0.17));   // tri-linear interpolation
  EXPECT_DOUBLES_EQUAL(0.3, idx.get<0>(), 1e-9);
  EXPECT_DOUBLES_EQUAL(3.8, idx.get<1>(), 1e-9);
  EXPECT_DOUBLES_EQUAL(1.567372, field.signed_distance(idx), 1e-9)

  idx = boost::make_tuple(1.0, 2.0);
  EXPECT(assert_equal(gtsam::Point2(0.0, -0.1), field.convertCelltoPoint2(idx)));

  // gradient
  gtsam::Vector2 grad_act, grad_exp;
  gtsam::Point2 p;
  p = gtsam::Point2(-0.13, -0.14);
  field.getSignedDistance(p, grad_act);
  grad_exp = numericalDerivative11(boost::function<double(const gtsam::Point2&)>(
      boost::bind(sdf_wrapper, field, _1)), p, 1e-6);
  EXPECT(gtsam::assert_equal(grad_exp, grad_act, 1e-6));

  p = gtsam::Point2(0.18, 0.12);
  field.getSignedDistance(p, grad_act);
  grad_exp = numericalDerivative11(boost::function<double(const gtsam::Point2&)>(
      boost::bind(sdf_wrapper, field, _1)), p, 1e-6);
  EXPECT(gtsam::assert_equal(grad_exp, grad_act, 1e-6));

}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}


