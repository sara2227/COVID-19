About:

This work, in particular, evaluates the performance of various DCNN architectures, including VGG16, Resnet50, MobileNet, InceptionResNetV2 for classification of chest X-ray images from covid-19-positive, pneumonia and normal healthy individuals in 3-classes vs 2-classes classification schemes.

Run:

Please run code as this order:
1- Please make the directories as said in req_folders.txt.
2- pip install -r requirements.txt
3- Run train.py file for training and set args parameters:
for example:
    python train.py --data_path data1 --model resnet101 --epochs 1 --nc 3
4- For predict on test data run predict.py and set parameters:
for example:
    python predict.py --model_name '2class_vgg16-01-0.6100.h5' --nc 2 --data_path data1


Also you can use this link for lower size data on my Drive: https://drive.google.com/drive/folders/1wRlyUvE4kA4FCupgOTqvKd6I16wLExWo
