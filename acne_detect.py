from __future__ import print_function
import os
from os import listdir
from os.path import join, isfile, splitext
import numpy as np
import pandas as pd
import cntk as C
from PIL import Image
import pickle
import time
import json
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs
import getPatches
import cv2

pretrained_model_name = 'ResNet152_ImageNet_Caffe.model'
pretrained_model_path = 'models'
pretrained_node_name = 'pool5' 

label_mapping = {1: '1-Clear', 2: '2-Almost Clear', 3: '3-Mild', 4: '4-Moderate', 5: '5-Severe'}


base_path = 'C:/Users/sjun0/OneDrive/Desktop/acne_detect/'
img_path = base_path + 'img_path'
result_file = base_path + 'result_file.csv'
patch_path = base_path + 'patch_path'
regression_model_path = base_path + 'models/cntk_regression.dat'
eye_cascade_model = base_path + 'models/haarcascade_eye.xml'

image_height = 224 # Here are the image height and width that the skin patches of the testing selfie are going to be resized to.
image_width  = 224 # They have to be the same as the ResNet-152 model requirement.
num_channels = 3


# get the dimension of each patch of images in the testing image directory
dimension_dict = dict()
imageFiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]

eye_cascade =  cv2.CascadeClassifier(eye_cascade_model)
i = 1
for imagefile in imageFiles:
    image_name = imagefile.split('.')
    dim = getPatches.extract_patches(join(img_path, imagefile), {}, {} , eye_cascade, patch_path, str(i), image_name[0]) #extract_patches function is defined in getPatches.py
    dimension_dict[imagefile] = dim
    i += 1

# define pretrained model location, node name
model_file  = os.path.join(base_path, pretrained_model_path, pretrained_model_name)
loaded_model  = load_model(model_file)
node_in_graph = loaded_model.find_by_name(pretrained_node_name)
output_nodes  = combine([node_in_graph.owner])

node_outputs = C.logging.get_node_outputs(loaded_model)
for l in node_outputs: 
    if l.name == pretrained_node_name:
        num_nodes = np.prod(np.array(l.shape))
        
def extract_features(image_path):   
    img = Image.open(image_path)       
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)  
    
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]    
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2)) 
    
    arguments = {loaded_model.arguments[0]: [hwc_format]}    
    output = output_nodes.eval(arguments)   
    return output

#load the stored regression model
read_model = pd.read_pickle(regression_model_path)
regression_model = read_model['model'][0]
train_regression = pickle.loads(regression_model)

# get the score value for each patch
patch_score = dict()
for file in next(os.walk(patch_path))[2]:
    file_path = os.path.join(patch_path, file)
    # extract features from CNTK pretrained model
    score_features = extract_features (file_path)[0].flatten()
    # score the extracted features using trained regression model
    pred_score_label = train_regression.predict(score_features.reshape(1,-1))
    patch_score[file] = float("{0:.2f}".format(pred_score_label[0]))

# get the max score value among the patches and record the image name
image_patch_scores = {}

for key in patch_score:
    image_id = key.split("_")
    temp = [image_id[1],image_id[2]]
    temp = "-".join(temp)

    image_patch_scores_i = image_patch_scores.get(temp, {"patch_name":[], "patch_score":[]})
    image_patch_scores_i["patch_name"].append(key)
    image_patch_scores_i["patch_score"].append(patch_score[key])
    image_patch_scores[temp] = image_patch_scores_i
    
fp = open(result_file, 'w')
fp.write("Image_Name, Predicted_Label_Avg, fh, lc, rc, nose, chin\n")

for key in image_patch_scores:
    fh = lc = rc = nose = chin = ""
    image_name = key + ".png"
    max_index = np.argmax(image_patch_scores[key]['patch_score'])
    Predicted_Label_Avg = np.mean(image_patch_scores[key]['patch_score'])
    num = 0
    for image in image_patch_scores[key]['patch_name']:
        if("fh" in image_patch_scores[key]['patch_name'][num]):
            fh = image_patch_scores[key]['patch_score'][num]
        elif("lc" in image_patch_scores[key]['patch_name'][num]):
            lc = image_patch_scores[key]['patch_score'][num]
        elif("rc" in image_patch_scores[key]['patch_name'][num]):
            rc = image_patch_scores[key]['patch_score'][num]
        elif("nose" in image_patch_scores[key]['patch_name'][num]):
            nose = image_patch_scores[key]['patch_score'][num]
        elif("chin" in image_patch_scores[key]['patch_name'][num]):
            chin = image_patch_scores[key]['patch_score'][num]
        num = num + 1
        
    fp.write('%s, %.4f, %s, %s, %s, %s, %s\n'%(image_name, Predicted_Label_Avg, fh, lc, rc, nose, chin))

fp.close()
