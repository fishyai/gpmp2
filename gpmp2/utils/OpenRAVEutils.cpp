/**
 *  @file  OpenRAVEutils.cpp
 *  @brief utility functions wrapped in OpenRAVE
 *  @author Jing Dong
 *  @date  Oct 31, 2015
 **/

#include <gpmp2/utils/OpenRAVEutils.h>

#include <gtsam/inference/Symbol.h>


namespace gpmp2 {

/* ************************************************************************** */
void convertValuesOpenRavePointer(size_t dof, const gtsam::Values& results, double *pointer, size_t total_step,
    const gtsam::Vector& joint_lower_limit, const gtsam::Vector& joint_upper_limit) {
  
  // check limit dof
  if (joint_lower_limit.size() != static_cast<int>(dof) ||
      joint_upper_limit.size() != static_cast<int>(dof))
    throw std::runtime_error("[convertValuesOpenRavePointer] ERROR: Joint limit size is different from DOF.");

  for (size_t i = 0; i <= total_step; i++) {
    const gtsam::Key pose_key = gtsam::Symbol('x', i);
    const gtsam::Key vel_key = gtsam::Symbol('v', i);
    const gtsam::Vector& conf = results.at<gtsam::Vector>(pose_key);
    const gtsam::Vector& vel = results.at<gtsam::Vector>(vel_key);

    // copy memory
    for (size_t j = 0; j < dof; j++) {
      // check for joint limits
      if (conf(j) < joint_lower_limit(j))       pointer[dof*i + j] = joint_lower_limit(j);
      else if (conf(j) > joint_upper_limit(j))  pointer[dof*i + j] = joint_upper_limit(j);
      else                                      pointer[dof*i + j] = conf(j);
      // vel
      pointer[dof*(i+total_step+1) + j] = vel(j);
    }
  }
}

/* ************************************************************************** */
void convertOpenRavePointerValues(size_t dof, gtsam::Values& results, double *pointer,
    size_t total_step) {

  results.clear();

  for (size_t i = 0; i <= total_step; i++) {

    // key
    const gtsam::Key pose_key = gtsam::Symbol('x', i);
    const gtsam::Key vel_key = gtsam::Symbol('v', i);

    // get conf and vel values
    gtsam::Vector conf(dof), vel(dof);
    for (size_t j = 0; j < dof; j++) {
      conf(j) = pointer[dof*i + j];
      vel(j) = pointer[dof*(i+total_step+1) + j];
    }

    // insert values
    results.insert(pose_key, conf);
    results.insert(vel_key, vel);
  }
}

}

