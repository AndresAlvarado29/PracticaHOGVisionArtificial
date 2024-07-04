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
namespace fs = std::filesystem;

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

int main(int argc, char* args[]) {
    string dataset = "/home/andres/Documents/DatasetPropios/LogosDataset/train";
    vector<string> classes = {"Instagram", "Netflix", "Yahoo", "Youtube"};
    vector<Mat> datosDeEntrenamiento;
    vector<int> labels;

    // Leer imágenes y calcular descriptores HOG
    for (int label = 0; label < classes.size(); ++label) {
        string classPath = dataset + "/" + classes[label];
        for (const auto& entry : fs::directory_iterator(classPath)) {
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);   
            if (!img.empty()) {
                Mat imgRedimensionada;
                Size tamano(64, 128); // Cambiado a 64x128 para coincidir con winSize
                resize(img, imgRedimensionada, tamano, INTER_LINEAR);
                Mat descriptors = getHOGDescriptors(imgRedimensionada);
                datosDeEntrenamiento.push_back(descriptors);
                labels.push_back(label);
            }
        }
    }

    // Verificar si se han leído datos de entrenamiento
    if (datosDeEntrenamiento.empty() || labels.empty()) {
        cerr << "No se encontraron datos de entrenamiento." << endl;
        return 1;
    }

    // Convertir datos de entrenamiento a un formato adecuado para OpenCV
    int descriptorSize = datosDeEntrenamiento[0].cols;
    Mat matDatosDeEntrenamiento(static_cast<int>(datosDeEntrenamiento.size()), descriptorSize, CV_32FC1);

    for (size_t i = 0; i < datosDeEntrenamiento.size(); ++i) {
        for (int j = 0; j < descriptorSize; ++j) {
            matDatosDeEntrenamiento.at<float>(static_cast<int>(i), j) = datosDeEntrenamiento[i].at<float>(0, j);
        }
    }

    Mat labelsMat(static_cast<int>(labels.size()), 1, CV_32SC1);
    for (size_t i = 0; i < labels.size(); ++i) {
        labelsMat.at<int>(static_cast<int>(i), 0) = labels[i];
    }

    // Entrenamiento
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR);
    svm->setType(SVM::C_SVC);
    svm->setC(1);
    svm->train(matDatosDeEntrenamiento, ROW_SAMPLE, labelsMat);

    // Guardar el modelo
    svm->save("svm_model.yml");
    cout << "Modelo entrenado y guardado con éxito." << endl;

    return 0;
}
