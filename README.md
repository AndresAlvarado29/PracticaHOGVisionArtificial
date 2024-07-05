# PracticaHOGVisionArtificial

En esta practica se creo un dataset con 4 diferentes logos que son de youtube, netflix, instagram y yahoo, las imagenes descargamos de internet 50 images las que se dividieron en 2 carpetas, una de entrenamiento con 45 imagenes y otra de testing con 5 imagenes.

Antes de usar las imagenes se convirtieron de png a jpg para evitar problemas en el entrenamiento aqui el codigo para la tranformacion

        #include <iostream>
        #include <filesystem>
        #include <opencv2/imgcodecs.hpp>
        #include <opencv2/highgui.hpp>

        using namespace std;
        using namespace cv;
        namespace fs = std::filesystem;

        void convertPNGtoJPG(const string& inputDir, const string& outputDir) {
            for (const auto& entry : fs::directory_iterator(inputDir)) {
                if (entry.path().extension() == ".png") {
                    Mat img = imread(entry.path().string(), IMREAD_COLOR); // Leer en color
                    if (!img.empty()) {
                        string outputPath = outputDir + "/" + entry.path().stem().string() + ".jpg";
                        imwrite(outputPath, img);
                        cout << "Converted: " << entry.path().string() << " to " << outputPath << endl;
                    } else {
                        cerr << "Failed to load: " << entry.path().string() << endl;
                    }
                }
            }
        }

        int main() {
            string inputDirectory = "/home/andres/Documents/DatasetPropios/LogosDataset/train/Youtube";
            string outputDirectory = "/home/andres/Documents/DatasetPropios/LogosDataset/modi";

            // Crear el directorio de salida si no existe
            fs::create_directories(outputDirectory);

            convertPNGtoJPG(inputDirectory, outputDirectory);

            return 0;
        }


Despues empezamos con el entrenamiento donde se subio la carpeta de train para entrenerlo como se puede ver en el siguiente codigo


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


Luego del entrenamiento hicimos las predicciones con las imagenes de la carpeta test como se puede ver en la siguiente codigo 


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

por ultimo aqui se pueden ver las imagen resultantes 


![Imagen de las predicciones](./CapturasEvidencias/Screenshot%20from%202024-07-05%2012-21-13.png)

Logos de Youtube

![Imgane de logos de youtube](./CapturasEvidencias/Screenshot%20from%202024-07-05%2012-21-26.png)

Logos de Instagram

![Imgane de logos de instagram](./CapturasEvidencias/Screenshot%20from%202024-07-05%2012-22-37.png)

Logos de Netflix

![Imgane de logos de netflix](./CapturasEvidencias/Screenshot%20from%202024-07-05%2012-21-45.png)

Logos de Yahoo

![Imgane de logos de yahoo](./CapturasEvidencias/Screenshot%20from%202024-07-05%2012-21-39.png)