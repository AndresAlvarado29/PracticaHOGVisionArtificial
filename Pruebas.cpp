#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat getHOGDescriptors(const Mat& img) {
    HOGDescriptor hog(
        Size(64, 128), // winSize
        Size(16, 16),  // blockSize
        Size(8, 8),    // blockStride
        Size(8, 8),    // cellSize
        9              // nbins
    );
    vector<float> descriptors;
    hog.compute(img, descriptors);
    return Mat(descriptors).clone();
}

int main() {
    // Cargar el modelo SVM entrenado
    Ptr<SVM> svm = Algorithm::load<SVM>("svm_model.yml");

    // Leer la nueva imagen de prueba
    Mat nuevaImagen = imread("/home/andres/Documents/DatasetPropios/LogosDataset/train/Instagram/c6116757e5dfdc1bf385beb6add47a07.jpg", IMREAD_GRAYSCALE);
    if (nuevaImagen.empty()) {
        cerr << "No se pudo cargar la imagen." << endl;
        return 1;
    }

    // Preprocesar la imagen (redimensionar y calcular descriptores HOG)
    Mat imgRedimensionada;
    Size tamano(64, 128); // Tamaño esperado para el HOG
    resize(nuevaImagen, imgRedimensionada, tamano, INTER_LINEAR);
    Mat descriptors = getHOGDescriptors(imgRedimensionada);

    // Preparar datos para la predicción
    Mat inputMat(1, descriptors.cols, CV_32F);
    for (int j = 0; j < descriptors.cols; ++j) {
        inputMat.at<float>(0, j) = descriptors.at<float>(0, j);
    }

    // Hacer la predicción
    float prediccion = svm->predict(inputMat);

    // Determinar la clase predicha
    vector<string> clases = {"Instagram", "Netflix", "Yahoo", "Youtube"};
    int clasePredicha = static_cast<int>(prediccion);
    string clasePredichaStr = clases[clasePredicha];

    // Mostrar resultados
    cout << "La imagen es predicha como: " << clasePredichaStr << endl;

    return 0;
}
