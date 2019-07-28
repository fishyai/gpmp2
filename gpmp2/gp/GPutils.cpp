/**
 *  @file  GPutils.cpp
 *  @brief GP utils, calculation of Qc, Q, Lamda matrices etc.
 *  @author Xinyan Yan, Jing Dong
 *  @date Qct 26, 2015
 **/

#include <gpmp2/gp/GPutils.h>


namespace gpmp2 {

/* ************************************************************************** */
gtsam::Matrix getQc(const gtsam::SharedNoiseModel Qc_model) {
  gtsam::noiseModel::Gaussian *Gassian_model =
      dynamic_cast<gtsam::noiseModel::Gaussian*>(Qc_model.get());
  return (Gassian_model->R().transpose() * Gassian_model->R()).inverse();
}

} // namespace gpmp2


