all:
	g++ ObjectDetector.cpp ObjectDetection.cpp --std=c++17 -I/home/francisco/opencv_build/opencv/install/include/opencv4/ -L/home/francisco/opencv_build/opencv/install/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_ml -lopencv_dnn -lopencv_objdetect -o vision.bin

saludo:
	echo "Hola C++"

run:
	./vision.bin