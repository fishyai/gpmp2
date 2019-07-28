/**
*  @file testMobileBaseUtils.cpp
*  @author Jing Dong
*  @date Aug 20, 2016
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/kinematics/mobileBaseUtils.h>
#include <gpmp2/geometry/Pose2Vector.h>
#include <gpmp2/geometry/numericalDerivativeDynamic.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


/* ************************************************************************** */
TEST(mobileBaseUtils, computeBaseTransPose3) {

  gtsam::Matrix Hexp, Hact;
  gtsam::Pose3 pexp, pact, base_T_arm;
  gtsam::Pose2 p2;

  // zero base_T_arm cases
  base_T_arm = gtsam::Pose3();

  p2 = gtsam::Pose2();
  pexp = gtsam::Pose3();
  pact = computeBaseTransPose3(p2, base_T_arm, Hact);
  Hexp = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&computeBaseTransPose3, _1, base_T_arm, boost::none)), p2, 1e-6);
  EXPECT(gtsam::assert_equal(pexp, pact, 1e-9));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  p2 = gtsam::Pose2(1.3, 4.5, -0.3);
  pexp = gtsam::Pose3(gtsam::Rot3::Yaw(-0.3), gtsam::Point3(1.3, 4.5, 0));
  pact = computeBaseTransPose3(p2, base_T_arm, Hact);
  Hexp = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&computeBaseTransPose3, _1, base_T_arm, boost::none)), p2, 1e-6);
  EXPECT(gtsam::assert_equal(pexp, pact, 1e-9));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  // non-zero base_T_arm cases
  base_T_arm = gtsam::Pose3(gtsam::Rot3::Yaw(-0.3), gtsam::Point3(1,1,2));

  p2 = gtsam::Pose2();
  pexp = gtsam::Pose3(gtsam::Rot3::Yaw(-0.3), gtsam::Point3(1,1,2));
  pact = computeBaseTransPose3(p2, base_T_arm, Hact);
  Hexp = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&computeBaseTransPose3, _1, base_T_arm, boost::none)), p2, 1e-6);
  EXPECT(gtsam::assert_equal(pexp, pact, 1e-9));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  p2 = gtsam::Pose2(2, -2, M_PI_2);
  pexp = gtsam::Pose3(gtsam::Rot3::Yaw(M_PI_2-0.3), gtsam::Point3(1,-1,2));
  pact = computeBaseTransPose3(p2, base_T_arm, Hact);
  Hexp = gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Pose2&)>(
      boost::bind(&computeBaseTransPose3, _1, base_T_arm, boost::none)), p2, 1e-6);
  EXPECT(gtsam::assert_equal(pexp, pact, 1e-9));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));
}


/* ************************************************************************** */
// z dir lift utils
gtsam::Pose3 lift_arm_base_pose(const Pose2Vector& posevec, const gtsam::Pose3& base_T_arm,
    bool reverse_linact) {
  return liftBasePose3(posevec.pose(), posevec.configuration()(0), base_T_arm, 
      reverse_linact);
}

TEST(mobileBaseUtils, liftBasePose) {
  
  gtsam::Pose2 base_pose2;
  gtsam::Pose3 base_T_arm, arm_pose_exp, arm_pose_act;
  double lift;
  bool reverse_linact = false;
  Pose2Vector posevec;
  gtsam::Matrix64 Hexp, Hact;

  base_T_arm = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0.0, 2.0));

  // zero pose2, zero lift
  base_pose2 = gtsam::Pose2(0, 0, 0);
  lift = 0;
  posevec = Pose2Vector(base_pose2, (gtsam::Vector(1) << lift).finished());
  arm_pose_exp = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0.0, 2.0));
  arm_pose_act = liftBasePose3(base_pose2, lift, base_T_arm, reverse_linact, Hact);
  Hexp = numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&lift_arm_base_pose, _1, base_T_arm, reverse_linact)), posevec, 1e-6);
  EXPECT(gtsam::assert_equal(arm_pose_exp, arm_pose_act, 1e-6));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  // zero pose2, non-zero lift
  base_pose2 = gtsam::Pose2(0, 0, 0);
  lift = 1.5;
  posevec = Pose2Vector(base_pose2, (gtsam::Vector(1) << lift).finished());
  arm_pose_exp = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(1.0, 0.0, 3.5));
  arm_pose_act = liftBasePose3(base_pose2, lift, base_T_arm, reverse_linact, Hact);
  Hexp = numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&lift_arm_base_pose, _1, base_T_arm, reverse_linact)), posevec, 1e-6);
  EXPECT(gtsam::assert_equal(arm_pose_exp, arm_pose_act, 1e-6));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  // non-zero pose2, non-zero lift
  base_pose2 = gtsam::Pose2(1.0, 0, M_PI/4.0);
  lift = 2.5;
  posevec = Pose2Vector(base_pose2, (gtsam::Vector(1) << lift).finished());
  arm_pose_exp = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/2.0, 0, 0), gtsam::Point3(1.707106781186548, 0.707106781186548, 4.5));
  arm_pose_act = liftBasePose3(base_pose2, lift, base_T_arm, reverse_linact, Hact);
  Hexp = numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&lift_arm_base_pose, _1, base_T_arm, reverse_linact)), posevec, 1e-6);
  EXPECT(gtsam::assert_equal(arm_pose_exp, arm_pose_act, 1e-6));
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  // random for jacobian test
  base_T_arm = gtsam::Pose3(gtsam::Rot3::Ypr(3.3, -4.1, -6.5), gtsam::Point3(1.3, -2.2, 0.8));
  base_pose2 = gtsam::Pose2(2.1, -0.9, 4.0);
  lift = 3.7;
  posevec = Pose2Vector(base_pose2, (gtsam::Vector(1) << lift).finished());
  arm_pose_act = liftBasePose3(base_pose2, lift, base_T_arm, reverse_linact, Hact);
  Hexp = numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&lift_arm_base_pose, _1, base_T_arm, reverse_linact)), posevec, 1e-6);
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));

  // random for jacobian test, reserve lin act
  reverse_linact = true;
  arm_pose_act = liftBasePose3(base_pose2, lift, base_T_arm, reverse_linact, Hact);
  Hexp = numericalDerivativeDynamic(boost::function<gtsam::Pose3(const Pose2Vector&)>(
      boost::bind(&lift_arm_base_pose, _1, base_T_arm, reverse_linact)), posevec, 1e-6);
  EXPECT(gtsam::assert_equal(Hexp, Hact, 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
