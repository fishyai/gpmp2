/**
*  @file testObstacleSDFFactorArm.cpp
*  @author Jing Dong
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/kinematics/ArmModel.h>
#include <gpmp2/obstacle/ObstacleSDFFactor.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


typedef ObstacleSDFFactor<ArmModel> ObstacleSDFFactorArm;


inline gtsam::Vector errorWrapper(const ObstacleSDFFactorArm& factor,
    const gtsam::Vector& conf) {
  return factor.evaluateError(conf);
}

// convert sdf vector to hinge loss err vector
inline gtsam::Vector convertSDFtoErr(const gtsam::Vector& sdf, double eps) {
  gtsam::Vector err_ori = 0.0 - sdf.array() + eps;
  return (err_ori.array() > 0.0).select(err_ori, gtsam::Vector::Zero(err_ori.rows()));  // (R < s ? P : Q)
}

// data
SignedDistanceField sdf;

/* ************************************************************************** */
TEST(ObstacleSDFFactorArm, data) {

  double cell_size = 0.1;
  // zero orgin
  gtsam::Point3 origin(0,0,0);
  vector<gtsam::Matrix> field(3);

  field[0] = (gtsam::Matrix(7,7) <<
      0.2828, 0.2236, 0.2000, 0.2000, 0.2000, 0.2236, 0.2828,
      0.2236, 0.1414, 0.1000, 0.1000, 0.1000, 0.1414, 0.2236,
      0.2000, 0.1000, -0.1000, -0.1000, -0.1000, 0.1000, 0.2000,
      0.2000, 0.1000, -0.1000, -0.1000, -0.1000, 0.1000, 0.2000,
      0.2000, 0.1000, -0.1000, -0.1000, -0.1000, 0.1000, 0.2000,
      0.2236, 0.1414, 0.1000, 0.1000, 0.1000, 0.1414, 0.2236,
      0.2828, 0.2236, 0.2000, 0.2000, 0.2000, 0.2236, 0.2828).finished();
  field[1] = (gtsam::Matrix(7,7) <<
      0.3000, 0.2449, 0.2236, 0.2236, 0.2236, 0.2449, 0.3000,
      0.2449, 0.1732, 0.1414, 0.1414, 0.1414, 0.1732, 0.2449,
      0.2236, 0.1414, 0.1000, 0.1000, 0.1000, 0.1414, 0.2236,
      0.2236, 0.1414, 0.1000, 0.1000, 0.1000, 0.1414, 0.2236,
      0.2236, 0.1414, 0.1000, 0.1000, 0.1000, 0.1414, 0.2236,
      0.2449, 0.1732, 0.1414, 0.1414, 0.1414, 0.1732, 0.2449,
      0.3000, 0.2449, 0.2236, 0.2236, 0.2236, 0.2449, 0.3000).finished();
  field[2] = (gtsam::Matrix(7,7) <<
      0.3464, 0.3000, 0.2828, 0.2828, 0.2828, 0.3000, 0.3464,
      0.3000, 0.2449, 0.2236, 0.2236, 0.2236, 0.2449, 0.3000,
      0.2828, 0.2236, 0.2000, 0.2000, 0.2000, 0.2236, 0.2828,
      0.2828, 0.2236, 0.2000, 0.2000, 0.2000, 0.2236, 0.2828,
      0.2828, 0.2236, 0.2000, 0.2000, 0.2000, 0.2236, 0.2828,
      0.3000, 0.2449, 0.2236, 0.2236, 0.2236, 0.2449, 0.3000,
      0.3464, 0.3000, 0.2828, 0.2828, 0.2828, 0.3000, 0.3464).finished();

  sdf = SignedDistanceField(origin, cell_size, field);
}

/* ************************************************************************** */
TEST(ObstacleSDFFactorArm, error) {

  // 2 link simple example
  gtsam::Pose3 arm_base(gtsam::Rot3(), gtsam::Point3(0.05, 0.15, 0.05));
  gtsam::Vector2 a(0.1, 0.2), alpha(0, 0), d(0, 0);
  Arm abs_arm(2, a, alpha, d, arm_base);

  // body info, three spheres
  BodySphereVector body_spheres;
  const double r = 0.05;
  body_spheres.push_back(BodySphere(0, r, gtsam::Point3(-0.1, 0, 0)));
  body_spheres.push_back(BodySphere(0, r, gtsam::Point3(0, 0, 0)));
  body_spheres.push_back(BodySphere(1, r, gtsam::Point3(-0.1, 0, 0)));
  body_spheres.push_back(BodySphere(1, r, gtsam::Point3(0, 0, 0)));
  ArmModel arm(abs_arm, body_spheres);

  double obs_eps = 0.2;
  ObstacleSDFFactorArm factor(0, arm, sdf, 1.0, obs_eps);

  // just check cost of two link joint
  gtsam::Vector2 q;
  gtsam::Vector err_act, err_exp, sdf_exp;
  gtsam::Matrix H1_exp, H1_act;


  // origin zero case
  q = gtsam::Vector2(0, 0);
  err_act = factor.evaluateError(q, H1_act);
  sdf_exp = (gtsam::Vector(4) << 0.1810125, 0.099675, 0.06035, 0.06035).finished();
  err_exp = convertSDFtoErr(sdf_exp, obs_eps+r);
  H1_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector2&)>(
      boost::bind(&errorWrapper, factor, _1)), q, 1e-6);
  EXPECT(gtsam::assert_equal(err_exp, err_act, 1e-6));
  EXPECT(gtsam::assert_equal(H1_exp, H1_act, 1e-6));

  // 45 deg case
  q = gtsam::Vector2(M_PI/4.0, 0);
  err_act = factor.evaluateError(q, H1_act);
  sdf_exp = (gtsam::Vector(4) << 0.1810125, 0.095702211510784, 0.01035442302156, 0).finished();
  err_exp = convertSDFtoErr(sdf_exp, obs_eps+r);
  H1_exp = gtsam::numericalDerivative11(boost::function<gtsam::Vector(const gtsam::Vector2&)>(
      boost::bind(&errorWrapper, factor, _1)), q, 1e-6);
  EXPECT(gtsam::assert_equal(err_exp, err_act, 1e-6));
  EXPECT(gtsam::assert_equal(H1_exp, H1_act, 1e-6));
}


/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}


