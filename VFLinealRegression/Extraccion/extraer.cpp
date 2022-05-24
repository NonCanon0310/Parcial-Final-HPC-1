/*************************************************
    * Fecha: 23-05-2022
    * Autor:  Salvatore Victorino Arango
    * Materia: HPC-1
    * Tema: Contruccion de la interfaz de la clase extraer
    * Objetivo:
        1.-Crear una clase que permita la manipulacion de los datos
           (extraccion, normalizacion, entre otros) con eigen.
        2.-Crear una clase que permita implementar el modelo de Regresion Lineal, con eigen.
**************************************************/

#include "extraer.h"
#include <iostream>
#include <fstream> //Para el tratamiento de ficheros (csv)
#include <vector>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>

/* Primer función miembro "ReadCSV": lectura de fichero csv. Se presenta como un vector
 * de vectores de tipo string.
 * La idea es leer linea por linea y almacenar cada una en un vector de vectores
 * de tipo string. */

std::vector<std::vector<std::string>>  Extraer::ReadCSV(){ //ReadCSV es el nombre de la clase
    /*Abrir el fichero para lectura SOLAMENTE*/
    std::fstream Fichero(setDatos);

    /*Vector de vectores tipo string a entregar por partede la función*/

    std::vector<std::vector<std::string>> datosString;
    //Se itera a traves de cada linea y se divide el contenido dado por el separador provisto por el constructor*/

    // Almacenar cada linea
    std::string linea = "";
    while(getline(Fichero, linea)){

        /* se crea un vector para almacenar la fila */
        std::vector<std::string> vectorFila;

        /* Se separa segun el delimitador */
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(delimitador));
        datosString.push_back(vectorFila); //inserta el vector en el vector de vectores
    }
    /*se cierra el fichero .csv */
    Fichero.close();
    /*se retorna el vecto de vectores de tipo string*/
    return datosString;
}

/* se implementa la segunda función miembro, la cual tiene como mision
 * trasnformar el vector de vectores del tipo String, en una matrix Eigen.
 * La idea es simular un objeto DATAFRAME de pandas, para poder manipular los datos. */

Eigen::MatrixXd Extraer::CSVtoEigen(std::vector<std::vector<std::string>> SETdatos, int filas, int columnas){
    /*Se hace la pregunta si tiene cabecera o no el vector de vectores del tipo string.
     *Si tiene cabecera, de debe eliminar */

    if(header){
        filas = filas -1;
    }

    /* Se itera sobre cada registro del fichero (dataSet), a la vez que se almacena en una matrixXd de dimensión
     * filas por columnas. Principalmente, almacenará strings (porque llega un vector de vectores del tipo string).
     * La idea es hacer un casting de string a float (Es decir
     * cambiar el tipo de dato). */

    Eigen::MatrixXd MatrizDF(columnas, filas);

    for(int i=0; i<filas; i++){
        for(int j=0; j<columnas; j++){
            MatrizDF(j,i) = atof(SETdatos[i][j].c_str());
        }
    }
    /* Se transpone la matriz dato que viene por columnas por filas, para retornarla */
    return MatrizDF.transpose();
}

/* Función para calcular el promedio */
/* En c++ la herencia del tipo de dato no es directa(sobre tod si es  partir de funciones dadas por otras interfaces/clases/bibliotecas
 * :EIGEN, shrkml, etc).
 * Se declara el tipo en una expresión "decltype" con el fin de tener seguridad de qué tipo de dato retornará la función*/
auto Extraer::Promedio(Eigen::MatrixXd datos) ->
decltype(datos.colwise().mean()){
    return datos.colwise().mean();
}

/* Función para calcular la Desviación Estandar */
/* Para implementar la desviación estadar, datosescalados = xi - x.promedio*/
auto Extraer::DesvStandar(Eigen::MatrixXd datos) ->
decltype((datos.array().square().colwise().sum()/(datos.rows()-1)).sqrt()){
    return (datos.array().square().colwise().sum()/(datos.rows()-1)).sqrt();
}

/*A continuación se procede a implementar la función de normalización. La idea fundamental es que los datos presenten
 una cercana aproximación al promedio, evitando los valores cuyas magnitudes son muy altas o muy bajas
 (por ejemplo los outliers: valores atípicos) */

Eigen::MatrixXd Extraer::Normalizador(Eigen::MatrixXd datos){
    Eigen::MatrixXd datosEscalados = datos.rowwise()-Promedio(datos);
    Eigen::MatrixXd MatrixNormal = datosEscalados.array().rowwise()/ DesvStandar(datosEscalados);
    /* Se retorna la matrix Normalizada */
    return MatrixNormal;
}

/* Para los algoritmos y/o modelos de Machine Learning se necesita dividir los datos en dos grupos.
 * El primer grupo es del Entrenamiento: se recomienda que sea aproximadamente del 80% de los datos.
 * El segundo grupo es para Prueba: Será el resto, es decir el 20%.
 * La idea es crear una función que permita dividir los datos en los grupos de entrenamiento y prueba
 * de forma automática.  Se requiere que la elección de los registros para cada grupo seaq aleatoria.
 * Esto garantiza que el resultado del modelo de ML presente una aceptable precisión.*/

/* La funcion de division a continuaciòn tomarà el porcentaje superior
 * de la matriz dada, para entrenamiento. La parte restante de la matriz
 * dada para pruebas. La funcion devolverà una tupla de cuatro matrices dinamicas,
 * variables dependientes: entrenamiento y prueba, variables independientes:
 * entrenamiento y pruebas. Al utilizar la funcion en el principal,
 * se debe desempaquetar la tubla, para obtener los 4 conjuntos de datos.*/

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>

Extraer::TrainTestSplit(Eigen::MatrixXd DatosNorm, float sizeTrain){

    /* Se crea una variable para obtener el numero de filas totales.*/
    int filasTotal = DatosNorm.rows();

    /* Se crea una variable para obtener el numero de filas de entrenamiento.*/
    int filasTrain = round(filasTotal*sizeTrain);

    /* Se crea una variable para obtener el numero de filas de prueba.*/
    int filasTest = filasTotal-filasTrain;

    /* Se crea la matriz de entrenamiento: Parte superior de la matriz de entrada.*/
    Eigen::MatrixXd Train = DatosNorm.topRows(filasTrain);

    /* Del conjunto de entrenamiento y para este caso en especial (DataSet WineData),
     * todas las columnas de la izquiera son las variables independientes (features)
     * y la ultima columna de la derecha representa la variable dependiente.*/

    /* A continuacion, se declara el conjunto de entrenamiento de las variables independientes X. */
    Eigen::MatrixXd X_train = Train.leftCols(DatosNorm.cols()-1);

    /* A continuacion se declara el conjunto de entrenamiento de las variables dependientes Y. */
    Eigen::MatrixXd y_train = Train.rightCols(1);

    /* A continuacion se declara el conjunto de datos para prueba */
    Eigen::MatrixXd test = DatosNorm.bottomRows(filasTest);

    /* A continuacion se declara el conjunto de prueba de de las variables independientes X. */
    Eigen::MatrixXd X_test = test.leftCols(DatosNorm.cols()-1);

    /* A continuacion se declara el conjunto de prueba de de las variables dependientes Y. */
    Eigen::MatrixXd y_test = test.rightCols(1);

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

/* A continuacion se desarrollaran dos nuevas funciones para convertir de ficheros a vectos,
 * y pasar de una matriz a fichero. La idea principal es almacenar los valores parciales en
 * ficheros por motivos de seguridad, control y seguimiento de la ejecucion del algoritmo de regresion
 * lineal. */

/* Funcion para exportar valores de un fichero a un vector. Esta funcion de tipo vacio recibe
 * un vector que contendra los valores del archivo dado.*/

void Extraer::FiletoVector(std::vector<float>dataVector,std::string fileName){

    /* Se crea un buffer (bus de memoria temporal) como objeto que contiene la Data de un fichero. */
    std::ofstream BufferFichero(fileName);

    /* A continuacion se itera sobre el buffer almacenando cada objeto encontrado, representado
     * por un salto de linea ("\n"). */
    std::ostream_iterator<float>BufferIterator(BufferFichero,"\n");

    /* Se copia la data del iterador (BufferIterador) en el vector. */
    std::copy(dataVector.begin(),dataVector.end(),BufferIterator);
}

/* La siguente funcion representa la conversion de una matriz Eigen a fichero.*/

void Extraer::MatrixtoFile(Eigen::MatrixXd dataMatrix,std::string fileName){
    /* Se crea un buffer (bus de memoria temporal) como objeto que contiene la Data de un fichero. */
    std::ofstream BufferFichero(fileName);

    /*Se pone como condicion que mientras este abierto almacene los datos que estes separados
     * por un salto de linea*/
    if(BufferFichero.is_open()){
        BufferFichero << dataMatrix << "\n";
    }
}

/* Para determinar que tan bueno es nuestro modelo vamos a crear una funcion como metrica de rendimiento.
 * La metrica de rendimiento seleccionada es R2_score.*/

float Extraer::R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y-y_hat).array(),2).sum();
    auto denominador = pow(y.array()-y.mean(),2).sum();

    return (1-(numerador/denominador));
}
