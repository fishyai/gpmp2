/**
 *  @file   testPose2MobileBase.cpp
 *  @author Mustafa Mukadam
 *  @date   Jan 22, 2018
 **/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/kinematics/Pose2MobileBase.h>
#include <gpmp2/kinematics/mobileBaseUtils.h>
#include <gpmp2/geometry/numericalDerivativeDynamic.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
// fk wrapper
gtsam::Pose3 fkpose(const Pose2MobileBase& r, const gtsam::Pose2& p, const gtsam::Vector& v) {
  vector<gtsam::Pose3> pos;
  r.forwardKinematics(p, boost::none, pos, boost::none);
  return pos[0];
}

gtsam::Vector3 fkvelocity(const Pose2MobileBase& r, const gtsam::Pose2& p, const gtsam::Vector& v) {
  vector<gtsam::Pose3> pos;
  vector<gtsam::Vector3> vel;
  r.forwardKinematics(p, v, pos, vel);
  return vel[0];
}

TEST(Pose2MobileBase, Example) {

  Pose2MobileBase robot;

  gtsam::Pose2 q;
  gtsam::Vector3 qdot;
  gtsam::Vector qdymc;
  vector<gtsam::Pose3> pvec_exp, pvec_act;
  vector<gtsam::Vector3> vvec_exp, vvec_act;
  vector<gtsam::Matrix> vJp_exp, vJp_act, vJv_exp, vJv_act;
  vector<gtsam::Matrix> pJp_exp, pJp_act;

  // origin with zero vel
  q = gtsam::Pose2();
  qdot = (gtsam::Vector3() << 0, 0, 0).finished();
  qdymc = qdot;
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3());
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&fkpose, robot, _1, qdot)), q, 1e-6));
  // vJp_exp.clear();
  // vJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Pose2&)>(
  //     boost::bind(&fkvelocity, robot, _1, qdot)), q, 1e-6));
  // vJv_exp.clear();
  // vJv_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
  //     boost::bind(&fkvelocity, robot, q, _1)), qdot, 1e-6));
  //robot.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  robot.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  // EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));

  // origin with none zero vel
  q = gtsam::Pose2();
  qdot = (gtsam::Vector3() << 1, 0, 1).finished();
  qdymc = qdot;
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3());
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(1, 0, 1));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&fkpose, robot, _1, qdot)), q, 1e-6));
  // vJp_exp.clear();
  // vJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Pose2&)>(
  //     boost::bind(&fkvelocity, robot, _1, qdot)), q, 1e-6));
  // vJv_exp.clear();
  // vJv_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
  //     boost::bind(&fkvelocity, robot, q, _1)), qdot, 1e-6));
  //robot.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  robot.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  // EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));

  // non-zero pose
  q = gtsam::Pose2(2.0, -1.0, M_PI/2.0);
  qdot = (gtsam::Vector3() << 0, 0, 0).finished();
  qdymc = qdot;
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/2.0, 0, 0), gtsam::Point3(2.0, -1.0, 0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&fkpose, robot, _1, qdot)), q, 1e-6));
  // vJp_exp.clear();
  // vJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Pose2&)>(
  //     boost::bind(&fkvelocity, robot, _1, qdot)), q, 1e-6));
  // vJv_exp.clear();
  // vJv_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
  //     boost::bind(&fkvelocity, robot, q, _1)), qdot, 1e-6));
  //robot.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  robot.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  // EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));

  // non-zero pose and vel
  q = gtsam::Pose2(2.0, -1.0, M_PI/2.0);
  qdot = (gtsam::Vector3() << 1, 0, 1).finished();
  qdymc = qdot;
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/2.0, 0, 0), gtsam::Point3(2.0, -1.0, 0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(1, 0, 1));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&fkpose, robot, _1, qdot)), q, 1e-6));
  // vJp_exp.clear();
  // vJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Pose2&)>(
  //     boost::bind(&fkvelocity, robot, _1, qdot)), q, 1e-6));
  // vJv_exp.clear();
  // vJv_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
  //     boost::bind(&fkvelocity, robot, q, _1)), qdot, 1e-6));
  //robot.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  robot.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  // EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));

  // other values to check Jacobians
  q = gtsam::Pose2(2.3, -1.2, 0.1);
  qdot = (gtsam::Vector3() << 3.2, 4.9, -3.3).finished();
  qdymc = qdot;
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&fkpose, robot, _1, qdot)), q, 1e-6));
  // vJp_exp.clear();
  // vJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Pose2&)>(
  //     boost::bind(&fkvelocity, robot, _1, qdot)), q, 1e-6));
  // vJv_exp.clear();
  // vJv_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
  //     boost::bind(&fkvelocity, robot, q, _1)), qdot, 1e-6));
  //robot.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  robot.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  // EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
