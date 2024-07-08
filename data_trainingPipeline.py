from __future__ import print_function
import os
import numpy as np
import pandas as pd
import cntk as C
from PIL import Image
import pickle
import time
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs

pretrained_model_name = 'ResNet152_ImageNet_Caffe.model'
pretrained_model_path = './models'
pretrained_node_name = 'pool5'

img_dirs = ['1-Clear', '2-Almost Clear', '3-Mild', '4-Moderate', '5-Severe'] # image labels
data_path = './cropped_images/0926_Landmarks_Frontal_Faces_Selected_fh_cheeks_patches' # image data source

image_height = 224 # the height of resize image
image_width  = 224 # the width of resize image
num_channels = 3 # the RGB image has three chanels
random_seed = 5
train_ratio = 0.8 # this ratio is used for training and validation in the following models

picklefolder_path = os.path.join(data_path, 'pickle') # create a directory pickle to store pickle files for image patches in each
                                                      # label directory. Data of all files in each label directory are dumped into
                                                      # a single pickle file
if not os.path.exists(picklefolder_path):
    os.mkdir(picklefolder_path)

output_path = './models'
if not os.path.exists(output_path):
    os.mkdir(output_path)

regression_model_path = os.path.join(output_path, 'cntk_regression.dat')

# define pretrained model location, node name
model_file  = os.path.join(pretrained_model_path, pretrained_model_name)
loaded_model  = load_model(model_file) # load the pretrained ResNet-152 model.
node_in_graph = loaded_model.find_by_name(pretrained_node_name) #find the node name in the pretrained ResNet-152 model
output_nodes  = combine([node_in_graph.owner])

node_outputs = C.logging.get_node_outputs(loaded_model)
for l in node_outputs:
    if l.name == pretrained_node_name:
        num_nodes = np.prod(np.array(l.shape))

print ('the pretrained model is %s' % pretrained_model_name)
print ('the selected layer name is %s and the number of flatten nodes is %d' % (pretrained_node_name, num_nodes))

def extract_features(image_path):
    img = Image.open(image_path)
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)

    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    arguments = {loaded_model.arguments[0]: [hwc_format]}
    output = output_nodes.eval(arguments)  #extract the features from the pretrained model, and output
    return output

def maybe_pickle(folder_path):
    dataset = np.ndarray(shape=(len(next(os.walk(folder_path))[2]), num_nodes),
                         dtype=np.float16)
    num_image = 0
    for file in next(os.walk(folder_path))[2]:
        image_path = os.path.join(folder_path, file)
        dataset[num_image, :] = extract_features(image_path)[0].flatten()
        num_image = num_image + 1

    pickle_filename = folder_path.split('\\')[-1] + '.pickle'
    pickle_filepath = os.path.join(picklefolder_path, pickle_filename)
    if os.path.isfile(pickle_filepath):
        os.remove(pickle_filepath)
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    return pickle_filename


# Here, we go over each subdirectory corresponding to each label, and dump the data of all images in each
# subdirectory into a single pickle file
start_time = time.time()

pickle_names = []

for f in img_dirs:
    folder_path = os.path.join(data_path, f)
    pickle_names.append(os.path.join(picklefolder_path, maybe_pickle(folder_path)))  # store the pickle file name in pickle_names

print("It takes %s seconds to extract features from skin patch images and dump to pickle files." % (time.time() - start_time))

# This is the function that combines training data in each label subdirectory into the same pickle file, so to the validation data.
def merge_datasets(pickle_files, train_ratio):
    num_classes = len(pickle_files)
    num_datasets = [0]*num_classes
    for i in range(num_classes):
        with open(pickle_files[i], 'rb') as f:
            load_data = pickle.load(f)
            num_datasets[i] = load_data.shape[0]

    total_datasets = np.sum(num_datasets)

    num_train = [int(round(float(x)*train_ratio)) for x in num_datasets]
    num_valid = np.array(num_datasets) - np.array(num_train)

    total_train = np.sum(num_train)
    train_dataset = np.ndarray((total_train, num_nodes), dtype=np.float32)
    train_labels = np.ndarray(total_train, dtype=np.int32)

    total_valid = np.sum(num_valid)
    valid_dataset = np.ndarray((total_valid, num_nodes), dtype=np.float32)
    valid_labels = np.ndarray(total_valid, dtype=np.int32)

    start_trn, start_val = 0, 0
    # the first element in the pickle file is labeled as 1, followd by second element as 2, etc...
    np.random.seed(seed=random_seed)
    for label, pickle_file in enumerate(pickle_files):
        print (label+1)
        print (pickle_file)
        try:
            with open(pickle_file, 'rb') as f:
                data_set = pickle.load(f)
                np.random.shuffle(data_set) #shuffle the data in each pickle file

                train_data = data_set[0:num_train[label], :] # the first batch goes to training data
                train_dataset[start_trn:(start_trn+num_train[label]), :] = train_data
                train_labels[start_trn:(start_trn+num_train[label])] = label+1
                start_trn += num_train[label]

                valid_data = data_set[num_train[label]:num_datasets[label], :]
                valid_dataset[start_val:(start_val+num_valid[label]), :] = valid_data
                valid_labels[start_val:(start_val+num_valid[label])] = label+1
                start_val += num_valid[label]

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return train_dataset, train_labels, valid_dataset, valid_labels

# merge all dataset together and divide it into training and validation
train_dataset, train_labels, valid_dataset, valid_labels = merge_datasets(pickle_names, train_ratio)
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)

# add regression model which has three hidden layers (1024, 512, 256).
# It may take around 30 minutes to train the model.
# Default hyperparameters are used here:
# L2 penalty: 0.0001
# Solver: adam
# batch_size: 'auto', = min(200, n_samples) = 200 since n_samples > 200
# learning_rate: 'constant'
# learning_rate_init: 0.001
# max_iter: 200. 200 iterations.
# verbose: False. Turn it to True if you want to see the training progress.
from sklearn.neural_network import MLPRegressor
clf_regr = MLPRegressor(hidden_layer_sizes=(1024, 512, 256), activation='relu', random_state=random_seed)
clf_regr.fit(train_dataset, train_labels) #Start training the regression model using the training data

# Predict the labels of images in the validation dataset
pred_labels_regr = clf_regr.predict(valid_dataset)

# Calculate RMSE on the validation dataset
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_regr = sqrt(mean_squared_error(pred_labels_regr, valid_labels))
print ('the RMSE of regression NN is %f' % rmse_regr)


# Store regression model
regr_model = pickle.dumps(clf_regr)
regression_store= pd.DataFrame({"model":[regr_model]})
regression_store.to_pickle(regression_model_path)
