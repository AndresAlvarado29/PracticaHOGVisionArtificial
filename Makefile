all:
	g++ Pruebas.cpp --std=c++17 -I/home/andres/opencv_install/librerias/include/opencv4/ -L/home/andres/opencv_install/librerias/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_ml -lopencv_objdetect -o visionPruebas.bin

saludo:
	echo "Hola C++"

run:
	./visionPruebas.bin