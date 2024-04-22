#Water meter number detection with YOLOv5 on Android
This project details the successful implementation of a mobile application for water meter measurement detection, leveraging
the power of YOLOv5 in PyTorch. The project went through an initial phase of exploring options and models, including
TensorFlow and other object detection models. However, the transition to YOLOv5 in PyTorch marked a turning point,
providing more effective model conversion tools. This implementation will allow users to keep track of their water meter
measurements efficiently, improving efficiency and accuracy in water resource management.

###Transfer Learning with YOLOv5
To train the YOLOv5 model with a custom dataset, we proceed in a simple way, as
It is mentioned in the PyTorch example repository android-demo-app/ObjectDetection (PyTorch, 2021). This is
performed with a script that includes the Ultraytics YOLOv5 repository (Ultralytics, 2023). Required: the dataset,
in this case, that of Roboflow, which can be downloaded already prepared in YOLOv5 format; and the source code
from the train.py file, which is already prepared to be used by a command, which requires the file
data.yaml and the dataset.
The command for training would be as follows (the entire command is on a single line):
python train.py --img 640 --batch 16 --epochs 300 --data
data.yaml
--weights yolov5s.pt
As you can see, the command requires a series of arguments that dictate the configuration of inter-
ning. Having –img for the size of the input images, –batch for the number of samples used
per epoch, –epochs for the number of training epochs, –data for the file containing the shape of the
classes of the dataset, and finally, –weights for the version of the base model to use.
Once the model is trained, it is necessary to export it to be able to use it in the Android application.



###Exporting the model to Torchscritp format for using 
To export the trained model, we proceed in a similar way to training. In this case, the
path of the trained model and the export.py file, which is also included in the YOLOv5 repository. Without
However, it is necessary to make a small modification to the export.py file, which consists of changing a line
which is outdated in the YOLOv5 repository, for a correct export to the required format,
in this case optimized torchscript. The modification is done within the export torchscript() function, where there is
I have to add an if and two lines of code as follows:
def export_torchscript(...):
...
f1 = file.with_suffix(’.torchscript.ptl’)
...
if optimize:
optimize_for_mobile(ts)._save_for_lite_interpreter(str(f1),
_extra_files=extra_files)
Once the modification was made, it was exported with the following command:
python export.py --weights runs/train/exp/weights/best.pt --include torchscript
--optimize

In this case, the command requires three input parameters, - -weights which is the trained model and -
-include for the export formats, in this case torchscript and lastly, - -optimize which is the flag
to compress and quantize the trained model so it can be used in the application
Once the process is finished, it is possible to continue with the deployment of the Android application.



## Quick Start

To Test Run the Object Detection Android App, follow the steps below:
1. Install anaconda
2. Install cuda and Pytorch.

##Android app for YOLOv5
The prerequisites for the Android app are:
• PyTorch 1.10.0 and torchvision 0.11.1 (Optional).
• Python 3.8 (Optional).
• Android Pytorch libraries:
– pytorch android lite:1.13.1
– pytorch android torchvision lite:1.13.1
• Android Studio 4.0.1 or later.
For the Android application, only the Pytorch(2021) repository was cloned, since it is practically ready for
be used. However, there are a number of changes required to make it work correctly with the model.
that was previously trained. For this, it is necessary to update the version of PyTorch used by the application
for correct functioning of the optimized model; it is necessary to adjust the shape of the output data of the
model as well as add test images in which we want to test the model.
This requires copying the exported model to the assets folder and making the following changes:


##Update the name of the model to use
The application has 2 modes of use, uploaded images or in real time. You have to update the name in 2
parts of the code.
• To update the model used in the uploaded images, the change is made in the MainActiv- file.
ity.java, the name of the model used is changed to the one that has been put in the export.py file, in this
case best.torchscript.ptl. The change is as follows:

mModule = LiteModuleLoader.load(
MainActivity.assetFilePath
(getApplicationContext()
, "best.torchscript.ptl"));
where everything remains the same except the part inside the quotes, which is where the name of the model goes.
was created previously.
• To update the model used in real-time detection, the change is made in the Object file.
DetectionActivity.java, the change is exactly the same as the previous one, the object is changed in the same way
mModule as follows:
mModule = LiteModuleLoader.load(
MainActivity.assetFilePath
(getApplicationContext()
, "best.torchscript.ptl"));


##Update the model class description file
Para un correcto funcionamiento de la aplicación, es necesario incluir un archivo .txt en el que se listen, separadas
en lineas individuales, las clases del modelo entrenado. Por ejemplo, en este caso el archivo, de nombre
“medidores.txt” para el dataset tiene el siguiente contenido:
0
1
2
3
4
5
6
7
8
9
This file should go in the assets folder of the project. Now yes, the MainActivity.java file is changed to
name of the class file, in this case it looks like this:
BufferedReader br = new BufferedReader(
new InputStreamReader(
getAssets()
.open("meters.txt")
));


##Update the number of model output columns
In the PrePostProcessor.java file, update the number of data output columns, which would be the
number of classes in the dataset plus 5, which are the 4 corner positions of the detection rectangle and the
class that was detected, in this case it would be 15 columns, 10 classes (the digits from 0 to 9) plus the 5 mentioned
previously. In this case, it would be changed from mOutputColumn = 85 to mOutputColumn = 15, the change remains
as follows:
private static int mOutputColumn = 15;


##Add test images
To add images on which to test the model, it is necessary to add them to the assets folder of the
project and add them to the mTestImages array, which contains the image names, defined in the file
MainActivity.java. In this case, 3 images were added (meter1.jpg, meter2.jpg and meter3.jpg), leaving
as follows:
private String[] mTestImages = {"meter1.jpg",
"meter2.jpg","meter3.jpg",
aicook1.jpg", "aicook2.jpg",
"aicook3.jpg", "test1.png",
"test2.jpg", "test3.png"};
Once these changes have been made, it is now possible to use the Android application with the custom model.


##Implementation of reading and regression line
The implementation of a reading of the model was carried out, so that the
model detection. To implement this feature, first the center of each of the
detection rectangles, and were sorted in ascending order with the position value on the horizontal axis. Of
In this mode, the detection will be done by “sweeping” from left to right. Once arranged in this way,
The values of the rectangles are concatenated, obtaining the final reading.
Likewise, at the time of detection, a red line can be observed that passes through the middle of the
detection boxes. This line aims at the future implementation of an algorithm that calculates
the squared error of the detection rectangles, so that an additional layer of filter can be added
to be able to discard those rectangles that are too far from the line of the rest of the detections.
The basis of this algorithm is that the meters always have the numbers lined up next to each other,
so ideally the algorithm would only detect rectangles that follow the same line, which would cross the
center of the counter. Therefore, if we detect a rectangle that is not on this line, very possibly
is not one of the numbers in the counter, and can be discarded.
Unfortunately, due to time constraints, it was not possible to implement the algorithm to calculate the error and
eliminate those rectangles far from the line.


##Acknowledgement
José Chaparro
Omar Escápita
Luis Chacón
David Hernández
Samuel Sánchez
Denzel Muñoz
Anahí Peinado
Ramírez Graciela