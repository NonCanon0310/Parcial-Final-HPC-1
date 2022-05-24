/*************************************************

    * Fecha: 23-05-2022

    * Autor:  Salvatore Victorino Arango
    * Materia: HPC-1

    * Tema: Implementacion del algoritmo de Regresion Lineal
    * Requerimientos:
    * 1.-Crear una clase que permita la manipulacion de los datos
    * (extraccion, normalizacion, entre otros) con eigen.
    * 2.-Crear una clase que permita implementar el modelo de Regresion Lineal, con eigen.
**************************************************/

#include "Extraccion/extraer.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <string.h>
#include "regresionlineal.h"

int main(int argc, char *argv[]) {

    /* Se crea un objeto del tipo Extraer
     * para incluir los 3 argumentos que necesita el objeto. */

    Extraer extraerData(argv[1], argv[2], argv[3]);

    /* Se crea un objeto del tipo linearRegresion, sin ningun argumento de salida.*/
    RegresionLineal RL;


    /* Se requiere probar la lectura del fichero y luego se requiere obse rvar el dataset como un objeto
     * de matriz tipo dataFrame */

    std::vector<std::vector<std::string>> dataSET = extraerData.ReadCSV();
    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDATAF = extraerData.CSVtoEigen(dataSET,filas,columnas);

    /*Se imprime la matriz que contiene los datos del dataSET */
    std::cout << MatrizDATAF << std:: endl;
    std::cout << "\nFilas: " <<filas<<std:: endl;
    std::cout << "Columnas: "<<columnas<<std::endl;
    /* se imprime el Promedio, se debe validar */
    std::cout<<"\n"<<extraerData.Promedio(MatrizDATAF) << std::endl;

    /* se crea la matrix para almacenar la normalización*/
    Eigen::MatrixXd matNormal = extraerData.Normalizador(MatrizDATAF);
    /*std::cout<< matNormal <<std::endl;*/

    /* A continuacion se dividen entrenamiento y prueba en conjuntos de datos
     * de entrada (matNormal). */
    Eigen::MatrixXd X_test, y_test, X_train, y_train;

    /* Se dividen los datos y el 80% es para entrenamiento.*/
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> MatrixDividida = extraerData.TrainTestSplit(matNormal,0.8);
    /* Se desempaqueta la tupla.*/
    std::tie(X_train, y_train, X_test, y_test) = MatrixDividida;

    std::cout<<"\n"<< matNormal.rows()<<std::endl;
    std::cout<< X_test.rows()<<std::endl;
    std::cout<< X_train.rows()<<std::endl;

    /* A continuacion se hara el primer modulo de Ml. Se hara una clase regresionLineal
     * con su correspondiente constructor de argumentos de entrada y metodos para el calculo
     * del modelo RL. Se tiene en cuenta que el RL es un metodo estadistico que define
     * la relacion entre la variables independientes con las variables dependientes.
     * La idea principal es definir una linea recta (hiperplano) con sus coeficientes (pendientes)
     * y puntos de corte.
     * Se tienen diferentes metodos para resolver RL para este caso se usara el metodo de los
     * Minimos Cuadrados Ordinarios (OLS), por ser un metodo sencillo y computacionalmente
     * economico. Representa una solucion optima para conjunto de datos no complejos. El dataSet a
     * utilizar es el de vinoRojo el cual tiene 11 variables (multivariable) independientes. Para
     * ello hemos de implementar el algoritmo del gradiente descendiente, cuyo objetivo principal
     * es minimizar la funcion de costo.

    /* Se define un vector de entrenamiento y para prueba inicializador en unos.*/
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensionan las matrices para ser ubicadas en el vector de UNOS: similar al reshape de Numpy. */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta(coeficientes) que se pasara al algoritmo del gradiente descendiente. Basicamente es un vector
     * de ceros del mismo tamaño del entrenamiento, adicionalmente se pasara alpha(ratio de aprendizaje) y el numero de iteraciones.*/
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /* A continuacion se definen las clases de salida que representan los coeficientes y el vector de costo.*/
    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;

    /* Se desempaqueta la tupla como objeto instaciado del gradiente descendiente.*/
    std::tuple<Eigen::VectorXd, std::vector<float>> objetoGradiente = RL.GradienteDes(X_train,y_train,theta,alpha,iteraciones);
    std::tie(thetaSalida,costo) = objetoGradiente;

    /* Se imprimen los coeficientes para cada variable. */
    std::cout<<"\n"<<thetaSalida<<std::endl;

    /* Se imprime para inspeccion ocular los valores de la funcion de costo.*/
    /*for (auto v:costo){
        std::cout<<v<<std::endl;
    }*/

    /* Se almacena la funcion de costo y las variables theta a ficheros.*/
    extraerData.FiletoVector(costo,"costo.txt");
    extraerData.MatrixtoFile(thetaSalida, "theta.txt");

    /* Se calcula el promedio y la desviacion estandar para calcular las prediciones,
     * es decir, se debe normalizar para calcular la metrica.*/

    auto muData = extraerData.Promedio(MatrizDATAF);
    auto muFeatures = muData(0,4);
    auto escalado = MatrizDATAF.rowwise()-MatrizDATAF.colwise().mean();
    auto sigmaData = extraerData.DesvStandar(escalado);
    auto sigmaFeatures = sigmaData(0,4);

    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array() + muFeatures;
    Eigen::MatrixXd y = MatrizDATAF.col(4).topRows(42);

    float R2_score = extraerData.R2_score(y,y_train_hat);
    std::cout<<"\n"<<R2_score*100<<" %"<<std::endl;
    extraerData.MatrixtoFile(y_train_hat, "y_train_hatCpp.txt");

    return EXIT_SUCCESS;
}
