from keras.models import load_model
import glob
import seaborn as sns
import numpy as np
import random
import cv2
import os
import glob
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=True,
    help="path to test images")
ap.add_argument("--model_path", default='./check_points/',
    help="path to model")
ap.add_argument("--model_name",required=True,
    help="model name for predict")
ap.add_argument("--nc",required=True,type=int,
    help="number of classes")
ap.add_argument("--input_dim",default= (256,256,3),
    help="input dimension")
args = ap.parse_args()

model_path = args.model_path
data_path = args.data_path
num_classes = args.nc

#making test dataset:
data_test = []
labels_test = []
cm_labels=[]
# load image files from the dataset
image_files = [f for f in glob.glob(data_path +'/test' +"/**/*", recursive=True) if not os.path.isdir(f)] 
random.seed(42)
random.shuffle(image_files)

# create groud-truth label from the image path
for img in image_files:

    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data_test.append(image)

    label = img.split(os.path.sep)[-2]
    if num_classes == 3:
        cm_labels = [0,1,2]
        if label == "covid":
            label = 0
        elif label == "normal":
            label = 1
        else:
            label = 2
    elif num_classes ==2:
        cm_labels = [0,1]
        if label == "covid":
            label = 0
        else:
            label = 1        
    labels_test.append(label)
    
# pre-processing
data_test = np.array(data_test, dtype="float") / 255.0
print('test_data.shape:',data_test.shape)

model = load_model(model_path + model_name) 
labels_pred = model.predict_classes(data_test, batch_size=40)

cm = confusion_matrix(labels_test, labels_pred, labels=cm_labels)
rep = classification_report(labels_test, labels_pred)

print(cm,'\n',rep)

#plot and save cm
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['covid', 'normal','pneumonia']);
ax.yaxis.set_ticklabels(['covid', 'normal','pneumonia']);
plt.savefig('./figures/'+ 'cm_' + model_name+'.png') 
