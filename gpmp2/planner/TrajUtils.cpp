/**
 *  @file  TrajUtils-inl.h
 *  @brief utils for trajectory optimization, include initialization and interpolation
 *  @author Jing Dong, Mustafa Mukadam
 *  @date  May 11, 2015
 **/

#include <gpmp2/planner/TrajUtils.h>
#include <gpmp2/gp/GaussianProcessInterpolatorLinear.h>
#include <gpmp2/gp/GaussianProcessInterpolatorPose2.h>
#include <gpmp2/gp/GaussianProcessInterpolatorPose2Vector.h>

#include <gtsam/inference/Symbol.h>

#include <cmath>
#include <algorithm>

using namespace std;


namespace gpmp2 {

/* ************************************************************************** */
gtsam::Values initArmTrajStraightLine(const gtsam::Vector& init_conf,
    const gtsam::Vector& end_conf, size_t total_step) {

  gtsam::Values init_values;

  // init pose
  for (size_t i = 0; i <= total_step; i++) {
    gtsam::Vector conf;
    if (i == 0)
      conf = init_conf;
    else if (i == total_step)
      conf = end_conf;
    else
      conf = static_cast<double>(i) / static_cast<double>(total_step) * end_conf +
          (1.0 - static_cast<double>(i) / static_cast<double>(total_step)) * init_conf;

    init_values.insert(gtsam::Symbol('x', i), conf);
  }

  // init vel as avg vel
  gtsam::Vector avg_vel = (end_conf - init_conf) / static_cast<double>(total_step);
  for (size_t i = 0; i <= total_step; i++)
    init_values.insert(gtsam::Symbol('v', i), avg_vel);

  return init_values;
}

/* ************************************************************************** */
gtsam::Values initPose2VectorTrajStraightLine(const gtsam::Pose2& init_pose, const gtsam::Vector& init_conf,
    const gtsam::Pose2& end_pose, const gtsam::Vector& end_conf, size_t total_step) {

  gtsam::Values init_values;

  gtsam::Vector avg_vel = (gtsam::Vector(3+init_conf.size()) << end_pose.x()-init_pose.x(),
      end_pose.y()-init_pose.y(), end_pose.theta()-init_pose.theta(), 
      end_conf - init_conf).finished() / static_cast<double>(total_step);

  for (size_t i=0; i<=total_step; i++) {
    gtsam::Vector conf;
    gtsam::Pose2 pose;
    double ratio = static_cast<double>(i) / static_cast<double>(total_step);
    pose = gtsam::interpolate<gtsam::Pose2>(init_pose, end_pose, ratio);
    conf = (1.0 - ratio)*init_conf + ratio*end_conf;
    init_values.insert(gtsam::Symbol('x', i), Pose2Vector(pose, conf));
    init_values.insert(gtsam::Symbol('v', i), avg_vel);
  }

  return init_values;
}

/* ************************************************************************** */
gtsam::Values initPose2TrajStraightLine(const gtsam::Pose2& init_pose, const gtsam::Pose2& end_pose,
    size_t total_step) {

  gtsam::Values init_values;

  gtsam::Vector avg_vel = (gtsam::Vector(3) << end_pose.x()-init_pose.x(), end_pose.y()-init_pose.y(),
    end_pose.theta()-init_pose.theta()).finished() / static_cast<double>(total_step);

  for (size_t i=0; i<=total_step; i++) {
    gtsam::Pose2 pose;
    double ratio = static_cast<double>(i) / static_cast<double>(total_step);
    pose = gtsam::interpolate<gtsam::Pose2>(init_pose, end_pose, ratio);
    init_values.insert(gtsam::Symbol('x', i), pose);
    init_values.insert(gtsam::Symbol('v', i), avg_vel);
  }

  return init_values;
}

/* ************************************************************************** */
gtsam::Values interpolateArmTraj(const gtsam::Values& opt_values,
    const gtsam::SharedNoiseModel Qc_model, double delta_t, size_t inter_step) {

  // inter setting
  double inter_dt = delta_t / static_cast<double>(inter_step + 1);

  size_t last_pos_idx;
  size_t inter_pos_count = 0;

  // results
  gtsam::Values results;

  // TODO: gtsam keyvector has issue: free invalid pointer
  gtsam::KeyVector key_vec = opt_values.keys();

  // sort key list
  std::sort(key_vec.begin(), key_vec.end());

  for (size_t i = 0; i < key_vec.size(); i++) {
    gtsam::Key key = key_vec[i];

    if (gtsam::Symbol(key).chr() == 'x') {
      size_t pos_idx = gtsam::Symbol(key).index();

      if (pos_idx != 0) {
        // skip first pos to interpolate

        for (size_t inter_idx = 1; inter_idx <= inter_step+1; inter_idx++) {

          if (inter_idx == inter_step+1) {
            // last pose
            results.insert(gtsam::Symbol('x', inter_pos_count), opt_values.at<gtsam::Vector>(gtsam::Symbol('x', pos_idx)));
            results.insert(gtsam::Symbol('v', inter_pos_count), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', pos_idx)));

          } else {
            // inter pose
            double tau = static_cast<double>(inter_idx) * inter_dt;
            GaussianProcessInterpolatorLinear gp_inter(Qc_model, delta_t, tau);
            gtsam::Vector conf1 = opt_values.at<gtsam::Vector>(gtsam::Symbol('x', last_pos_idx));
            gtsam::Vector vel1  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', last_pos_idx));
            gtsam::Vector conf2 = opt_values.at<gtsam::Vector>(gtsam::Symbol('x', pos_idx));
            gtsam::Vector vel2  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', pos_idx));
            gtsam::Vector conf  = gp_inter.interpolatePose(conf1, vel1, conf2, vel2);
            gtsam::Vector vel  = gp_inter.interpolateVelocity(conf1, vel1, conf2, vel2);
            results.insert(gtsam::Symbol('x', inter_pos_count), conf);
            results.insert(gtsam::Symbol('v', inter_pos_count), vel);
          }
          inter_pos_count++;
        }

      } else {
        // cache first pose
        results.insert(gtsam::Symbol('x', 0), opt_values.at<gtsam::Vector>(gtsam::Symbol('x', 0)));
        results.insert(gtsam::Symbol('v', 0), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', 0)));
        inter_pos_count++;
      }

      last_pos_idx = pos_idx;
    }
  }

  return results;
}

/* ************************************************************************** */
gtsam::Values interpolateArmTraj(const gtsam::Values& opt_values,
    const gtsam::SharedNoiseModel Qc_model, double delta_t, size_t inter_step, 
    size_t start_index, size_t end_index) {

  gtsam::Values results;

  double inter_dt = delta_t / static_cast<double>(inter_step + 1);
  size_t result_index = 0;

  for (size_t i = start_index; i < end_index; i++) {

    results.insert(gtsam::Symbol('x', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('x', i)));
    results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i)));

    for (size_t inter_idx = 1; inter_idx <= inter_step; inter_idx++) {

      result_index++;
      double tau = static_cast<double>(inter_idx) * inter_dt;
      GaussianProcessInterpolatorLinear gp_inter(Qc_model, delta_t, tau);
      gtsam::Vector conf1 = opt_values.at<gtsam::Vector>(gtsam::Symbol('x', i));
      gtsam::Vector vel1  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i));
      gtsam::Vector conf2 = opt_values.at<gtsam::Vector>(gtsam::Symbol('x', i+1));
      gtsam::Vector vel2  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i+1));
      gtsam::Vector conf  = gp_inter.interpolatePose(conf1, vel1, conf2, vel2);
      gtsam::Vector vel  = gp_inter.interpolateVelocity(conf1, vel1, conf2, vel2);
      results.insert(gtsam::Symbol('x', result_index), conf);
      results.insert(gtsam::Symbol('v', result_index), vel);
    }

    result_index++;
  }

  results.insert(gtsam::Symbol('x', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('x', end_index)));
  results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', end_index)));

  return results;
}

/* ************************************************************************** */
gtsam::Values interpolatePose2MobileArmTraj(const gtsam::Values& opt_values,
    const gtsam::SharedNoiseModel Qc_model, double delta_t, size_t inter_step, 
    size_t start_index, size_t end_index) {

  gtsam::Values results;

  double inter_dt = delta_t / static_cast<double>(inter_step + 1);
  size_t result_index = 0;

  for (size_t i = start_index; i < end_index; i++) {

    results.insert(gtsam::Symbol('x', result_index), opt_values.at<Pose2Vector>(gtsam::Symbol('x', i)));
    results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i)));

    for (size_t inter_idx = 1; inter_idx <= inter_step; inter_idx++) {

      result_index++;
      double tau = static_cast<double>(inter_idx) * inter_dt;
      GaussianProcessInterpolatorPose2Vector gp_inter(Qc_model, delta_t, tau);
      Pose2Vector conf1 = opt_values.at<Pose2Vector>(gtsam::Symbol('x', i));
      gtsam::Vector vel1  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i));
      Pose2Vector conf2 = opt_values.at<Pose2Vector>(gtsam::Symbol('x', i+1));
      gtsam::Vector vel2  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i+1));
      Pose2Vector conf  = gp_inter.interpolatePose(conf1, vel1, conf2, vel2);
      gtsam::Vector vel  = gp_inter.interpolateVelocity(conf1, vel1, conf2, vel2);
      results.insert(gtsam::Symbol('x', result_index), conf);
      results.insert(gtsam::Symbol('v', result_index), vel);
    }

    result_index++;
  }

  results.insert(gtsam::Symbol('x', result_index), opt_values.at<Pose2Vector>(gtsam::Symbol('x', end_index)));
  results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', end_index)));

  return results;
}

/* ************************************************************************** */
gtsam::Values interpolatePose2Traj(const gtsam::Values& opt_values,
    const gtsam::SharedNoiseModel Qc_model, double delta_t, size_t inter_step, 
    size_t start_index, size_t end_index) {

  gtsam::Values results;

  double inter_dt = delta_t / static_cast<double>(inter_step + 1);
  size_t result_index = 0;

  for (size_t i = start_index; i < end_index; i++) {

    results.insert(gtsam::Symbol('x', result_index), opt_values.at<gtsam::Pose2>(gtsam::Symbol('x', i)));
    results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i)));

    for (size_t inter_idx = 1; inter_idx <= inter_step; inter_idx++) {

      result_index++;
      double tau = static_cast<double>(inter_idx) * inter_dt;
      GaussianProcessInterpolatorPose2 gp_inter(Qc_model, delta_t, tau);
      gtsam::Pose2 conf1 = opt_values.at<gtsam::Pose2>(gtsam::Symbol('x', i));
      gtsam::Vector vel1  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i));
      gtsam::Pose2 conf2 = opt_values.at<gtsam::Pose2>(gtsam::Symbol('x', i+1));
      gtsam::Vector vel2  = opt_values.at<gtsam::Vector>(gtsam::Symbol('v', i+1));
      gtsam::Pose2 conf  = gp_inter.interpolatePose(conf1, vel1, conf2, vel2);
      gtsam::Vector vel  = gp_inter.interpolateVelocity(conf1, vel1, conf2, vel2);
      results.insert(gtsam::Symbol('x', result_index), conf);
      results.insert(gtsam::Symbol('v', result_index), vel);
    }

    result_index++;
  }

  results.insert(gtsam::Symbol('x', result_index), opt_values.at<gtsam::Pose2>(gtsam::Symbol('x', end_index)));
  results.insert(gtsam::Symbol('v', result_index), opt_values.at<gtsam::Vector>(gtsam::Symbol('v', end_index)));

  return results;
}

}
