#include "regresionlineal.h"

#include "Extraccion/extraer.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string.h>

/* Se necesita entrenar el modelo, lo que implica minimizar alguna funcion
 * de costo (para este caso se ha seleccionado para funcion de costo a los
 * minimos cuadrados ordinarios (OLS)), y de esta forma se puede medir la
 * funcion de hipotesis. Una funcion de costo es la foma de penalizar al modelo
 * por cometer un error. Se implementa una funcion del tipo flotante, que toma
 * como entradas "x" y "y" y los valores de theta inicializados (los valores
 * theta se fijan inicialmente en cualquier valor para que al iterar segun un
 * alpha, consiga el menor valor para la funcion de costo. */

float RegresionLineal::fCostoOLS(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd theta)
{
    /* Se almacena la diferencia elevada al cuadrado (funcion de hipotesis que representa el error)*/
    Eigen::MatrixXd diferencia = pow((X *theta -y).array(),2);
    return (diferencia.sum()/(2*X).rows());
}

/* Se necesita proveer al programa una funcion par dar al algoritmo un valor incial para theta,
 * el cual va a cambiar iterativamente hasta que converja al valor minimo de la funcion de costo.
 * basicamente describe el Gradiente Descendiente. La idea es calcula el gradiente para la funcion
 * de costo dado por la derivada parcial. L afuncion tendra un alpha que representa el salto del
 * gradiente. La funcion tiene como "X", "y", "theta", "alpha", y el numero de iteraciones que
 * necesita theta actualizada cada vez para que la funcion converja.*/

std::tuple<Eigen::VectorXd, std::vector<float>> RegresionLineal::GradienteDes(Eigen::MatrixXd X,
                                                                              Eigen::MatrixXd y,
                                                                              Eigen::VectorXd theta,
                                                                              float alpha,
                                                                              int iteraciones){
    /* Se almacena temporalmente los parametros de theta.*/
    Eigen::MatrixXd tempTheta = theta;

    /* Se extrae la cantidad de parametros.*/
    int parametros = theta.rows();

    /* Valores del costo inicial, se actualizara cada vez con los nuevos pesos (pendientes).*/
    std::vector<float> costo;
    costo.push_back(fCostoOLS(X,y,theta));

    /* Para cada iteracion se calcula la funcion de error. Se multiplica cada feature (X)
     * que calcula el error y se almacena en una variable temporal, luego se actualiza theta
     * y se calcula de nuevo la funcion de costo basada en el nuevo avalor de theta.*/

    for (int i=0; i<iteraciones; ++i){
        Eigen::MatrixXd Error = X*theta-y;
        for (int j=0; j<parametros; ++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd tempError = Error.cwiseProduct(X_i);
            tempTheta(j,0) = theta(j,0)-((alpha/X.rows())*(tempError.sum()));
        }
        theta = tempTheta;
        costo.push_back(fCostoOLS(X,y,theta));
    }

    /* Se empaqueta la tupla y se retorna */
    return std::make_tuple(theta, costo);
}
