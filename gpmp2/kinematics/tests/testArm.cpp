/**
*  @file testArm.cpp
*  @author Mustafa Mukadam, Jing Dong
*  @date Nov 11, 2015
**/

#include <CppUnitLite/TestHarness.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>

#include <gpmp2/kinematics/Arm.h>

#include <iostream>

using namespace std;
using namespace gpmp2;


// fk wrapper
gtsam::Pose3 fkpose(const Arm& arm, const gtsam::Vector& jp, const gtsam::Vector& jv, size_t i) {
  vector<gtsam::Pose3> pos;
  vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>> vel;
  arm.forwardKinematics(jp, jv, pos, vel);
  return pos[i];
}

gtsam::Vector3 fkvelocity(const Arm& arm, const gtsam::Vector& jp, const gtsam::Vector& jv, size_t i) {
  vector<gtsam::Pose3> pos;
  vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>> vel;
  arm.forwardKinematics(jp, jv, pos, vel);
  return vel[i];
}

/* ************************************************************************** */
TEST(Arm, 2linkPlanarExamples) {

  // 2 link simple example, with none zero base poses
  gtsam::Vector2 a(1, 1), alpha(0, 0), d(0, 0);
  gtsam::Pose3 base_pose(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(2.0, 1.0, -1.0));
  Arm arm(2, a, alpha, d, base_pose);
  gtsam::Vector2 q, qdot;
  gtsam::Vector qdymc;		// dynamic size version qdot
  vector<gtsam::Pose3> pvec_exp, pvec_act;
  vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>> vvec_exp, vvec_act;
  vector<gtsam::Matrix> vJp_exp, vJp_act, vJv_exp, vJv_act;
  vector<gtsam::Matrix> pJp_exp, pJp_act;


  // origin with zero vel
  q = gtsam::Vector2(0.0, 0.0);
  qdot = gtsam::Vector2(0.0, 0.0);
  qdymc = gtsam::Vector(qdot);
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(2.707106781186548, 1.707106781186548, -1.0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(3.414213562373095, 2.414213562373095, -1.0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));

  pJp_exp.clear();
  pJp_exp.push_back(numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));

  arm.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));


  // origin with none zero vel of j0
  q = gtsam::Vector2(0.0, 0.0);
  qdot = gtsam::Vector2(1.0, 0.0);
  qdymc = gtsam::Vector(qdot);
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(2.707106781186548, 1.707106781186548, -1.0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(3.414213562373095, 2.414213562373095, -1.0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(-0.707106781186548, 0.707106781186548, 0));
  vvec_exp.push_back(gtsam::Vector3(-1.414213562373095, 1.414213562373095, 0));

  pJp_exp.clear();
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));

  arm.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));


  // origin with none zero vel of j1
  q = gtsam::Vector2(0.0, 0.0);
  qdot = gtsam::Vector2(0.0, 1.0);
  qdymc = gtsam::Vector(qdot);
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(2.707106781186548, 1.707106781186548, -1.0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/4.0, 0, 0), gtsam::Point3(3.414213562373095, 2.414213562373095, -1.0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  vvec_exp.push_back(gtsam::Vector3(-0.707106781186548, 0.707106781186548, 0));

  pJp_exp.clear();
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));

  arm.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));


  // origin with none zero pos/vel
  q = gtsam::Vector2(M_PI/4.0, M_PI/4.0);
  qdot = gtsam::Vector2(0.1, 0.1);
  qdymc = gtsam::Vector(qdot);
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI/2.0, 0, 0), gtsam::Point3(2.0, 2.0, -1.0)));
  pvec_exp.push_back(gtsam::Pose3(gtsam::Rot3::Ypr(M_PI*0.75, 0, 0), gtsam::Point3(1.2928932188134528, 2.707106781186548, -1.0)));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(-0.1, 0, 0));
  vvec_exp.push_back(gtsam::Vector3(-0.241421356237309, -0.141421356237309, 0));

  pJp_exp.clear();
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector2&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector2&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));

  arm.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-9));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-9));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));
}

/* ************************************************************************** */
TEST(Arm, 3link3Dexample) {

  // random 3 link arm
  // no link rotation groud truth

  gtsam::Vector3 a(3.6, 9.8, -10.2), alpha(-0.4, 0.7, 0.9), d(-12.3, -3.4, 5.0);
  Arm arm(3, a, alpha, d);
  gtsam::Vector3 q, qdot;
  gtsam::Vector qdymc;		// dynamic size version qdot
  vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>> pointvec_exp;
  vector<gtsam::Pose3> pvec_act;
  vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>> vvec_exp, vvec_act;
  vector<gtsam::Matrix> vJp_exp, vJp_act, vJv_exp, vJv_act;
  vector<gtsam::Matrix> pJp_exp, pJp_act;


  // random example
  q = gtsam::Vector3(-1.1, 6.3, 2.4);
  qdot = gtsam::Vector3(10.78, 3, -6.3);
  qdymc = gtsam::Vector(qdot);

  pointvec_exp.clear();
  pointvec_exp.push_back(gtsam::Point3(1.6329, -3.2083, -12.3000));
  pointvec_exp.push_back(gtsam::Point3(5.0328, -12.4727, -15.4958));
  pointvec_exp.push_back(gtsam::Point3(1.4308, -22.9046, -12.8049));
  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(34.5860, 17.6032, 0));
  vvec_exp.push_back(gtsam::Vector3(158.3610, 66.9758, -11.4473));
  vvec_exp.push_back(gtsam::Vector3(240.7202, 32.6894, -34.1207));

  pJp_exp.clear();
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector3&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector3&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector3&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(2))), q, 1e-6));
  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(2))), q, 1e-6));
  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector3&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(2))), qdot, 1e-6));

  arm.forwardKinematics(q, qdymc, pvec_act, vvec_act, pJp_act, vJp_act, vJv_act);
  EXPECT(gtsam::assert_equal(pointvec_exp[0], pvec_act[0].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pointvec_exp[1], pvec_act[1].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pointvec_exp[2], pvec_act[2].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[2], vvec_act[2], 1e-3));
  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[2], vJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[2], vJv_act[2], 1e-6));
}

/* ************************************************************************** */
TEST(Arm, WAMexample) {

  // WAM arm exmaple, only joint position, no rotation
  gtsam::Vector7 a = (gtsam::Vector7() << 0.0, 0.0, 0.045, -0.045, 0.0, 0.0, 0.0).finished();
  gtsam::Vector7 alpha = (gtsam::Vector7() << -M_PI/2.0, M_PI/2.0, -M_PI/2.0, M_PI/2.0, -M_PI/2.0, M_PI/2.0, 0.0).finished();
  gtsam::Vector7 d = (gtsam::Vector7() << 0.0, 0.0, 0.55, 0.0, 0.3, 0.0, 0.06).finished();
  Arm arm(7, a, alpha, d);
  gtsam::Vector7 q, qdot;
  gtsam::Vector qdymc;		// dynamic size version qdot
  vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>> pvec_exp;
  vector<gtsam::Pose3> pvec_act1, pvec_act2;
  vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>> vvec_exp, vvec_act;
  vector<gtsam::Matrix> pJp_exp, pJp_act1, pJp_act2, vJp_exp, vJp_act, vJv_exp, vJv_act;

  // example
  q = (gtsam::Vector7() << 1.58,1.1,0,1.7,0,-1.24,1.57).finished();
  qdot = (gtsam::Vector7() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished();
  qdymc = gtsam::Vector(qdot);
  
  pvec_exp.clear();
  pvec_exp.push_back(gtsam::Point3(0, 0, 0));
  pvec_exp.push_back(gtsam::Point3(0, 0, 0));
  pvec_exp.push_back(gtsam::Point3(-0.0047, 0.5106, 0.2094));
  pvec_exp.push_back(gtsam::Point3(-0.0051, 0.5530, 0.2244));
  pvec_exp.push_back(gtsam::Point3(-0.0060, 0.6534, -0.0582));
  pvec_exp.push_back(gtsam::Point3(-0.0060, 0.6534, -0.0582));
  pvec_exp.push_back(gtsam::Point3(-0.0066, 0.7134, -0.0576));

  vvec_exp.clear();
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  vvec_exp.push_back(gtsam::Vector3(0, 0, 0));
  vvec_exp.push_back(gtsam::Vector3(-0.0557, 0.0204, -0.0511));
  vvec_exp.push_back(gtsam::Vector3(-0.0606, 0.0234, -0.0595));
  vvec_exp.push_back(gtsam::Vector3(-0.0999, -0.0335, -0.0796));
  vvec_exp.push_back(gtsam::Vector3(-0.0999, -0.0335, -0.0796));
  vvec_exp.push_back(gtsam::Vector3(-0.1029, -0.0333, -0.0976));

  pJp_exp.clear();
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(0))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(1))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(2))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(3))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(4))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(5))), q, 1e-6));
  pJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Pose3(const gtsam::Vector7&)>(
      boost::bind(&fkpose, arm, _1, qdot, size_t(6))), q, 1e-6));

  vJp_exp.clear();
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(0))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(1))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(2))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(3))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(4))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(5))), q, 1e-6));
  vJp_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, _1, qdot, size_t(6))), q, 1e-6));

  vJv_exp.clear();
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(0))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(1))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(2))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(3))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(4))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(5))), qdot, 1e-6));
  vJv_exp.push_back(gtsam::numericalDerivative11(boost::function<gtsam::Vector3(const gtsam::Vector7&)>(
      boost::bind(&fkvelocity, arm, q, _1, size_t(6))), qdot, 1e-6));

  // full fk with velocity
  arm.forwardKinematics(q, qdymc, pvec_act1, vvec_act, pJp_act1, vJp_act, vJv_act);
  // fk no velocity
  arm.forwardKinematics(q, boost::none, pvec_act2, boost::none, pJp_act2);

  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act1[0].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act1[1].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act1[2].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[3], pvec_act1[3].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[4], pvec_act1[4].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[5], pvec_act1[5].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[6], pvec_act1[6].translation(), 1e-3));

  EXPECT(gtsam::assert_equal(pvec_exp[0], pvec_act2[0].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[1], pvec_act2[1].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[2], pvec_act2[2].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[3], pvec_act2[3].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[4], pvec_act2[4].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[5], pvec_act2[5].translation(), 1e-3));
  EXPECT(gtsam::assert_equal(pvec_exp[6], pvec_act2[6].translation(), 1e-3));

  EXPECT(gtsam::assert_equal(vvec_exp[0], vvec_act[0], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[1], vvec_act[1], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[2], vvec_act[2], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[3], vvec_act[3], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[4], vvec_act[4], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[5], vvec_act[5], 1e-3));
  EXPECT(gtsam::assert_equal(vvec_exp[6], vvec_act[6], 1e-3));

  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act1[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act1[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act1[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act1[3], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[4], pJp_act1[4], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[5], pJp_act1[5], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[6], pJp_act1[6], 1e-6));

  EXPECT(gtsam::assert_equal(pJp_exp[0], pJp_act2[0], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[1], pJp_act2[1], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[2], pJp_act2[2], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[3], pJp_act2[3], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[4], pJp_act2[4], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[5], pJp_act2[5], 1e-6));
  EXPECT(gtsam::assert_equal(pJp_exp[6], pJp_act2[6], 1e-6));

  EXPECT(gtsam::assert_equal(vJp_exp[0], vJp_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[1], vJp_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[2], vJp_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[3], vJp_act[3], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[4], vJp_act[4], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[5], vJp_act[5], 1e-6));
  EXPECT(gtsam::assert_equal(vJp_exp[6], vJp_act[6], 1e-6));

  EXPECT(gtsam::assert_equal(vJv_exp[0], vJv_act[0], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[1], vJv_act[1], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[2], vJv_act[2], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[3], vJv_act[3], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[4], vJv_act[4], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[5], vJv_act[5], 1e-6));
  EXPECT(gtsam::assert_equal(vJv_exp[6], vJv_act[6], 1e-6));
}

/* ************************************************************************** */
/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
