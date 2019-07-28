/**
*  @file testPose2MobileVetLinArm.cpp
*  @author Jing Dong
*  @date Aug 20, 2016
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/kinematics/Pose2MobileVetLinArm.h>
#include <gpmp2/kinematics/mobileBaseUtils.h>
#include <gpmp2/geometry/numericalDerivativeDynamic.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
// fk wrapper
gtsam::Pose3 fkpose(const Pose2MobileVetLinArm& r, const Pose2Vector& p, size_t i) {
  vector<gtsam::Pose3> pos;
  vector<gtsam::Vector3> vel;
  r.forwardKinematics(p, boost::none, pos, boost::none);
  return pos[i];
}

// TODO: tests for velocity not implemented
TEST(Pose2MobileVetLinArm, 2linkPlanarExamples) {

  // 2 link simple example, with none zero base poses
  gtsam::Vector2 a(1, 1), alpha(0, 0), d(0, 0);
  Arm arm(2, a, alpha, d);
  gtsam::Pose3 base_pose(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0.0, 2.0));
  Pose2MobileVetLinArm marm(arm, gtsam::Pose3(), base_pose, false);

  Pose2Vector q;
  //Vector6 qdot;
  //Vector qdymc;
  vector<gtsam::Pose3> pvec_exp, pvec_act;
  //vector<Vector3> vvec_exp, vvec_act;
  vector<gtsam::Matrix> pJp_exp, pJp_act;
  //vector<Matrix> vJp_exp, vJp_act, vJv_exp, vJv_act;


  // origin with zero pose
  q = Pose2Vector(gtsam::Pose2(), gtsam::Vector3(0.0, 0.0, 0.0));
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3());
  pvec_exp.push_back(gtsam::Pose3());
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.707106781186548, 0.707106781186548, 2.0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(2.414213562373095, 1.414213562373095, 2.0)));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(0))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(1))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(2))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(3))), q, 1e-6));
  marm.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act[2], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act[2], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act[3], 1e-6));
  

  // origin with none zero pose
  q = Pose2Vector(gtsam::Pose2(1.0, 0.0, M_PI/4.0), gtsam::Vector3(1.5, 0.0, M_PI/4.0));
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0, 0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0, 1.5)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/2.0, 0, 0), gtsam::Point3(1.707106781186548, 1.707106781186548, 3.5)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(3.0*M_PI/4.0, 0, 0), gtsam::Point3(1, 2.414213562373095, 3.5)));
  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(0))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(1))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(2))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(3))), q, 1e-6));
  marm.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act[2], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act[2], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act[3], 1e-6));

  // random stuff for jacobians
  
  // random base pose
  gtsam::Pose3 base_torso = gtsam::Pose3(gtsam::Rot3::Ypr(-1.4, -3.1, 8.2), gtsam::Point3(1.3, -2.1, -3.8));
  base_pose = gtsam::Pose3(gtsam::Rot3::Ypr(-3.3, -4.1, 6.5), gtsam::Point3(-1.3, 2.2, -0.8));
  marm = Pose2MobileVetLinArm(arm, base_torso, base_pose, false);

  q = Pose2Vector(gtsam::Pose2(1.2, -9.4, 1.5), gtsam::Vector3(2.5, -0.2, 4.0));  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(0))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(1))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(2))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(3))), q, 1e-6));
  marm.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act[3], 1e-6));

  // reserve lin act
  marm = Pose2MobileVetLinArm(arm, base_torso, base_pose, true);
  
  q = Pose2Vector(gtsam::Pose2(1.2, -9.4, 1.5), gtsam::Vector3(2.5, -0.2, 4.0));  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(0))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(1))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(2))), q, 1e-6));
  pJp_exp.push_back(numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&fkpose, marm, _1, size_t(3))), q, 1e-6));
  marm.forwardKinematics(q, boost::none, pvec_act, boost::none, pJp_act);
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act[3], 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
