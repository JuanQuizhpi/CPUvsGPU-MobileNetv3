#include <iostream>
#include <fstream>
#include <chrono>
#include "ObjectDetector.hpp"

int main()
{
    cv::dnn::Net neuralNetwork;
    std::vector<std::string> classes;
    std::string inputDirectory = "../TestFiles";
    std::string outputDirectory = "../Results";

    try {
        neuralNetwork = cv::dnn::readNetFromTensorflow("../SSDMobileNetV3/frozenInterfaceGraph.pb", "../SSDMobileNetV3/frozenInterfaceGraph.pbtxt");
    } catch (cv::Exception &error) {
        std::cout << "Error cargando la red: " << error.msg << std::endl;
        return 1;
    }

    // Leer las clases desde el archivo
    std::ifstream classesFile("../SSDMobileNetV3/classes.txt");
    if (!classesFile.is_open()) {
        std::cerr << "No se pudo abrir el archivo de clases." << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(classesFile, line))
        classes.push_back(line);

    ObjectDetector objectDetector(neuralNetwork, classes);
    objectDetector.setIODirectory(inputDirectory, outputDirectory);

    // Pruebas false = CPU , True = GPU
    std::cout << "Inicio Pruebas" << std::endl;
    objectDetector.configureBackend(true); // Usar CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    int processResultCPU = objectDetector.detectObjects("Office.mp4", ObjectDetector::SourceFileType::Video);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "Tiempo: " << elapsedCPU.count() << " segundos" << std::endl;

    // Calcular FPS para CPU y GPU
    double fpsCPU = (1.0 / elapsedCPU.count()) * 25; // Suponiendo 25 frames por segundo en video

    std::cout << "FPS capturados: " << fpsCPU << std::endl;
}
