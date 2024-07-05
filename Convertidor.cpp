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