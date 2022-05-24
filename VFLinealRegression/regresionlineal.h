#ifndef REGRESIONLINEAL_H
#define REGRESIONLINEAL_H

#include "Extraccion/extraer.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string.h>

class RegresionLineal
{
public:
    RegresionLineal(){

    }
    float fCostoOLS(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDes(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iteraciones);
};

#endif /¨/ REGRESIONLINEAL_H
