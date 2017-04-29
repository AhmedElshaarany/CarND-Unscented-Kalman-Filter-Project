#include <iostream>
#include "tools.h"

#define RMSE_SIZE 4

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // define rmse vector
  VectorXd rmse(RMSE_SIZE);

  // initialize rmse vector with zeros
  rmse.fill(0.0);

  // check correctness of estimation and ground truth vector sizes
  if( estimations.size() == 0){
    std::cout <<" CalculateRMSE() - Error - Estimation vector size is Zero";
  }
  else if(estimations.size() != ground_truth.size()){
    std::cout <<" CalculateRMSE() - Error - Estimation vector size not equal to ground truth vector size";
  }
  else{
    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
      VectorXd est = estimations[i];
      VectorXd gnd_tth = ground_truth[i];
      VectorXd vec_diff = est-gnd_tth;
      VectorXd diff_sqrd = vec_diff.array()*vec_diff.array();
      rmse = rmse + diff_sqrd;
    }
    
    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();
  }

  // return the resulting vector
  return rmse;

}
