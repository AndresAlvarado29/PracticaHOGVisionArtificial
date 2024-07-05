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

Mat getHOGDescriptors(const Mat &img)
{
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

int main(){
    Ptr<SVM> svm = Algorithm::load<SVM>("svm_model.yml");
    string testing = "/home/andres/Documents/DatasetPropios/LogosDataset/test";
    vector<string> classes = {"Instagram", "Netflix", "Yahoo", "Youtube"};
    int windowIndex = 0;
    Size tamanoImagen(300, 300);

    for (int label = 0; label < classes.size(); ++label)
    {
        string classPath = testing + "/" + classes[label];
        for (const auto &entry : fs::directory_iterator(classPath))
        {
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (img.empty())
            {
                cout << "No se pudo cargar la imagen en la ruta: " << entry.path().string() << endl;
            }
            Mat imgRedimensionada;
            Size tamano(64, 128); // Tamaño esperado para el HOG
            resize(img, imgRedimensionada, tamano, INTER_LINEAR);
            Mat descriptors = getHOGDescriptors(imgRedimensionada);
            Mat inputMat(1, descriptors.cols, CV_32F);
            for (int j = 0; j < descriptors.cols; ++j)
            {
                inputMat.at<float>(0, j) = descriptors.at<float>(0, j);
            }

            float prediccion = svm->predict(inputMat);
            int clasePredicha = static_cast<int>(prediccion);
            string clasePredichaStr = classes[clasePredicha];
            Mat imgColor;
            cvtColor(img, imgColor, COLOR_GRAY2BGR);
            resize(imgColor, imgColor, tamanoImagen);
            putText(imgColor, clasePredichaStr, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            string windowName = "Imagen Testing " + to_string(windowIndex);
            imshow(windowName, imgColor);
            cout << "La imagen en " << entry.path().string() << " es predicha como: " << clasePredichaStr << endl;
            windowIndex++;
            waitKey(1);
        }
    }
    cout << endl;
    waitKey(0);
    return 0;
}
