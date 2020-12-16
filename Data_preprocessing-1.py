#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:58:51 2020

@author: ddman
"""

import pandas as pd
import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
import random
import sklearn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from time import sleep

all_music_data = pd.read_csv("../salami-data-public/metadata/SALAMI_iTunes_library.csv", sep=',')
all_music_data_2 = pd.read_csv("./salami_youtube_pairings.csv", sep=",")

# Concat the dataframe salami-iTunes_library.csv and salami_youtube_pairings.csv
new_music_data = pd.merge(all_music_data, all_music_data_2, how="inner")

# find all .mp3 file which could be downloaded from youtube and sotre into the list (music_list)
# All the .mp3 file is downloaded from youtube by "youtube dl"
music_list = []
all_music = os.listdir("./downloaded_audio/")
for i in range(len(all_music)):
    if ".mp3" == os.path.splitext(all_music[i])[-1]:
        music_list.append(all_music[i])
    else :
        music_list = music_list

# combine the number and mp3 file name
music_path = []
for i in range(len(new_music_data["salami_id"])):
    for j in range(len(music_list)):
        if os.path.splitext(music_list[j])[-2][-11:] == new_music_data["youtube_id"][i]:
            temp_list = []
            temp_list.append(new_music_data["salami_id"][i])
            temp_list.append(music_list[j])
            music_path.append(temp_list)
        else:
            music_path = music_path
            
music_id_with_file = pd.DataFrame(music_path, columns=['salami_id','file_name'])

# merge the music_id_with_file and new_music_data
salami_data = pd.merge(new_music_data, music_id_with_file, how="inner")

# Output the salami_data first !!!
salami_data.to_csv("New_salami_dataframe.csv", sep="\t")

# calculate the data's the descriptive statistics (Before the cracked into the 3 seconds patches)
genre_statistics_original = salami_data['Genre'].value_counts()
genre_statistics_original.to_csv("genre_original.csv", "\t")
print(len(genre_statistics_original))

# calculate the data after the cracked into the 3 seconds patches

# interaction the music filepath and the salami id
music_all_path = []
for i in range(len(salami_data["salami_id"])):
    for j in range(len(music_path)):
        if music_path[j][0] == list(salami_data["salami_id"])[i]:
            music_all_path.append(music_path[j])
        else:
            music_all_path = music_all_path

# crab all the path of labels into the list
temp_all = os.listdir("../renew-salami-dataset/annotations/")
for i in range(len(temp_all)):
    temp_all[i] = int(temp_all[i])
temp_all.sort() # number in the annotation
temp_all = pd.DataFrame(temp_all, columns=['salami_id'])
salami_data = pd.merge(salami_data, temp_all, how="inner")
salami_structure_label = []
for i in range(len(salami_data['salami_id'])):
    salami_id_label = []
    if os.path.isfile("../renew-salami-dataset/annotations/{}/parsed/textfile1_functions.txt".format(salami_data['salami_id'][i])):
        salami_id_label.append(salami_data['salami_id'][i])
        salami_id_label.append(pd.read_csv("../renew-salami-dataset/annotations/{}/parsed/textfile1_functions.txt".format(salami_data['salami_id'][i]), sep="\t",  header=None))
        salami_structure_label.append(salami_id_label)
    else:
        salami_id_label.append(salami_data['salami_id'][i])
        salami_id_label.append(pd.read_csv("../renew-salami-dataset/annotations/{}/parsed/textfile2_functions.txt".format(salami_data['salami_id'][i]), sep="\t",  header=None))
        salami_structure_label.append(salami_id_label)
salami_structure_label = pd.DataFrame(salami_structure_label, columns=["salami_id","labels"]) 
salami_structure_label['labels'][0]

# calculate the number categories of labels
final_all_path_and_labels = []    
for i in range(len(salami_structure_label)):
    final_all_path_and_labels.append(salami_structure_label["labels"][i])

all_labels = []
for i in range(len(final_all_path_and_labels)):
    for j in range(len(final_all_path_and_labels[i][1])):
        all_labels.append(final_all_path_and_labels[i][1][j])

pd.DataFrame(all_labels).value_counts().to_csv("original_sum_of_categories.csv", sep="\t") # Save the sum of categories of labels data

# The labels we want (1)
rest_labels = ['Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out',\
                'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition',\
                'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro']

rest_all_labels = []
for i in range(len(all_labels)):
    for j in range(len(rest_labels)):
        if all_labels[i] == rest_labels[j]:
            rest_all_labels.append(all_labels[i])
        else :
            rest_all_labels = rest_all_labels

pd.DataFrame(rest_all_labels).value_counts().to_csv("new_sum_of_categories.csv", sep="\t")

# 
#=======================================================================================================================================
# rest_labels = ['Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out',\
#                'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition',\
#                'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro']
# # read the audio and transform to Mel-spectrograms and cut it into patches 
# patch_with_labels_list = []
# for i in range(len(final_all_path_and_labels)):
#     y, sr = librosa.load("./downloaded_audio/{}".format(final_all_path_and_labels[i][1]),sr=16000)
#     # mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
#     # Get the sample point in timestamps
#     s_array = librosa.time_to_samples(np.array(final_all_path_and_labels[i][2][0]), sr=16000)
#     # Check the labels and deal with it!!! 
#     for j in range(len(final_all_path_and_labels[i][2][1])):
#         if final_all_path_and_labels[i][2][1][j] == "no_function":
#             temp_ = 1
#             while temp_ < 10 :
#                 if final_all_path_and_labels[i][2][1][j-1] != "no_function":
#                     final_all_path_and_labels[i][2][1][j] = final_all_path_and_labels[i][2][1][j-1]
#                     temp_ += 10
#                 else : 
#                     j -= 1
#         else :
#             final_all_path_and_labels[i][2][1][j] = final_all_path_and_labels[i][2][1][j]
#     # Use label with the audio(y)
#     for k in range(len(final_all_path_and_labels[i][2][1])-1):
#         l_list = []
#         l_list.append(final_all_path_and_labels[i][2][1][k])
#         if s_array[k] != s_array[k+1] and s_array[k+1] <= np.shape(y)[0]:
#             mel_spec = librosa.feature.melspectrogram(y=y[s_array[k]:s_array[k+1]], sr=sr)
#             l_list.append(mel_spec)
#             patch_with_labels_list.append(l_list)
#         else :
#             patch_with_labels_list = patch_with_labels_list
#     print(i)

#===========================================================================================================================================================

# Then we delete the short segment and deal with all same labels (ex: outro and Outro),
# and calculate the all_labels_and_freq again. (rest_labels-(1))

# step 1 : crack the data to the segments 
seg = []
for i in range(len(final_all_path_and_labels)):
    path = final_all_path_and_labels[i][1]
    seg_ = final_all_path_and_labels[i][2]
    # segment the msuic timestamps data
    for j in range(len(seg_)-2):
        temp_seg = []
        temp_seg.append(path)
        temp_seg.append(np.array(seg_[0][j+0:j+2]))
        temp_seg.append(seg_[1][j+0])
        seg.append(temp_seg)
# step 2 : eliminate some labels (ex: no_function, count-in, vocal etc.) and the short parts
rest_labels = ['Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out',\
               'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition',\
               'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro']
all_seg = []
for i in range(len(seg)):
    for label in rest_labels:
        if seg[i][2] == label :
            if (seg[i][1][1]-seg[i][1][0]) > 3 : # short segments are eliminated
                all_seg.append(seg[i])
            else :
                all_seg = all_seg
        else :
            all_seg = all_seg

# step 3 : add the same labels together (ex: outro and Outro, Secondary_theme and Secondary_Theme) 
for i in range(len(all_seg)):
    if all_seg[i][2] == 'silence' :
        all_seg[i][2] = 'Silence'
    elif all_seg[i][2] == 'outro' :
        all_seg[i][2] = 'Outro'
    elif all_seg[i][2] == 'Secondary_theme' :
        all_seg[i][2] = 'Secondary_Theme'
    else :
        all_seg[i][2] = all_seg[i][2]
        
print(len(all_seg))


for i in tqdm(range(len(all_seg))):
    if len(all_seg[i]) == 3 :
        y, sr = librosa.load("./downloaded_audio/{}".format(all_seg[i][0]),sr=16000)
        t_array = []
        for j in range(len(all_seg)):
            if all_seg[i][0] == all_seg[j][0]:
                t_array.append(j)
            else :
                t_array = t_array
        for k in range(len(t_array)):
            s_array = librosa.time_to_samples(np.array(all_seg[t_array[k]][1]), sr=16000)
            if len(y) >= s_array[1]:
                mel_spec = librosa.feature.melspectrogram(y=y[s_array[0]:s_array[1]], sr=sr)
                all_seg[t_array[k]].append(mel_spec)
            else :
                all_seg = all_seg
    else :
        all_seg = all_seg
    print("progress:{}%".format(round((i/len(all_seg))*100, 3)))

#=======================================================================================================================================

# fix the wrong segment (ex: len(all_seg)!=4)
new_seg = []
for i in range(len(all_seg)):
    if len(all_seg[i]) != 4:
        new_seg = new_seg
    else:
        new_seg.append(all_seg[i])



#=======================================================================================================================================
# l_list = []
# for i in range(len(patch_with_labels_list)):
#     l_list.append(patch_with_labels_list[i][0])
# len(np.unique(l_list)) # all category number

# for i in range(len(patch_with_labels_list)):
#     print(np.shape(patch_with_labels_list[i][1]))

# long = len(patch_with_labels_list)

# # find all under the 3 seconds patch and delete it !!!
# over_short = []
# for i in range(long):
#     if np.shape(patch_with_labels_list[i][1])[1] < 94 :
#         over_short.append(i)
#     else :
#         over_short = over_short

# for i in over_short:
#     print(np.shape(patch_with_labels_list[i][1]))
#=========================================================================================================================================
# fill all the long enough patches into the new list and crack it into 3 seconds patches
p_l = []
for i in range(len(new_seg)):
    if np.shape(new_seg[i][3])[1] >= 94 :
        # crack it into 3 seconds patches
        for j in range(0, np.shape(new_seg[i][3])[1], 32):
            l = []
            if np.shape(new_seg[i][3])[1] >= 94+j :
                temp_array = new_seg[i][3][0:128,0+j:94+j]
                l.append(new_seg[i][2])
                l.append(temp_array)
                p_l.append(l)
            else :
                p_l = p_l
    else:
        p_l = p_l
# aa = 0
# for i in range(len(p_l)):
#     aa += len(p_l[i][1])
# print(aa) # Total the patches number
# to_l = []
# for i in range(len(p_l)):
#     to_l.append(p_l[i][0])
# len(np.unique(to_l)) # Total the patches labels number
# targets_num = pd.get_dummies(pd.Series(np.unique(to_l))) # Get the labels to the number(arrays)

# shuffle and split the data set to test set and train set
random.shuffle(p_l)
training_p_l, testing_p_l = sklearn.model_selection.train_test_split(p_l, train_size=len(p_l)-10000, test_size=10000)

# set the labels and data (training)
train_d = []
train_t = []
for i in range(len(training_p_l)):
    train_d.append(training_p_l[i][1])
    train_t.append(training_p_l[i][0])
train_d = np.array(train_d)
train_t = np.array(train_t)

# set the labels and data (testing)
test_d = []
test_t = []
for i in range(len(training_p_l)):
    test_d.append(training_p_l[i][1])
    test_t.append(training_p_l[i][0])
test_d = np.array(test_d)
test_t = np.array(test_t)

# set the labels to the number
targets_num_train = pd.get_dummies(pd.Series(np.unique(train_t)))
targets_num_test = pd.get_dummies(pd.Series(np.unique(test_t)))

# Set the labels and data well !!!
train_targets = []
for i in range(len(train_t)):
    train_targets.append(np.argmax(np.array(targets_num_train[train_t[i]])))
train_targets = np.array(train_targets)
test_targets = []
for i in range(len(test_t)):
    test_targets.append(np.argmax(np.array(targets_num_test[test_t[i]])))
test_targets = np.array(test_targets)

x_val = train_d[:10000]
y_val = train_targets[:10000]

train_d = train_d[10000:]
train_targets = train_targets[10000:]

training_dataset = tf.data.Dataset.from_tensor_slices((train_d, train_targets))
testing_dataset = tf.data.Dataset.from_tensor_slices((test_d, test_targets))

BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 100

training_dataset = training_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)

#=======================================================================
nb_layers = 4  # number of convolutional layers
nb_filters = [64, 128, 128, 128]  # filter sizes
kernel_size = (3, 3)  # convolution kernel size
activation = 'elu'  # activation function to use after each layer
pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
             (4, 2)]  # size of pooling area

# shape of input data (frequency, time, channels)
inputs = tf.keras.Input(shape=[128,94,1])
input_shape = (inputs.shape[1], inputs.shape[2], inputs.shape[3])
frequency_axis = 1
time_axis = 2
channel_axis = 3

# Create sequential model and normalize along frequency axis
output_1 = layers.BatchNormalization(axis=frequency_axis, input_shape=input_shape)(inputs)

# First convolution layer specifies shape
output_1 = layers.Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                         data_format="channels_last",
                         input_shape=input_shape)(output_1)

output_1 = layers.Activation(activation)(output_1)
output_1 = layers.BatchNormalization(axis=channel_axis)(output_1)
output_1 = layers.MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0])(output_1)
output_1 = layers.Dropout(0.1)(output_1)

# Add more convolutional layers
for layer in range(nb_layers - 1):
    # Convolutional layer
    output_1 = layers.Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                     padding='same')(output_1)
    output_1 = layers.Activation(activation)(output_1)
    output_1 = layers.BatchNormalization(
        axis=channel_axis)(output_1)  # Improves overfitting/underfitting
    output_1 = layers.MaxPooling2D(pool_size=pool_size[layer + 1],
                           strides=pool_size[layer + 1])(output_1)  # Max pooling
    output_1 = layers.Dropout(0.1)(output_1)

    # Reshaping input for recurrent layer
# (frequency, time, channels) --> (time, frequency, channel)
output_1 = layers.Permute((time_axis, frequency_axis, channel_axis))(output_1)
resize_shape = output_1.shape[2] * output_1.shape[3]
output_1 = layers.Reshape((output_1.shape[1], resize_shape))(output_1)

# recurrent layer
output_1 = layers.GRU(32, return_sequences=True)(output_1)
output_1 = layers.GRU(32, return_sequences=False)(output_1)
output_1 = layers.Dropout(0.3)(output_1)

# Output layer
output_1 = layers.Dense(39)(output_1)
output_1 = layers.Activation("softmax")(output_1)

model = tf.keras.Model(inputs, output_1)
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_d, train_targets, epochs=1000, batch_size=16, validation_data=(x_val, y_val))

model.evaluate(testing_dataset)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val loss'], loc='lower left')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()


plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#=======================================================================
i = 2
inputs = tf.keras.Input(shape=[128,94,1])

temp1 = layers.Conv2D(i*10, (1,100), padding='same', activation='relu')(inputs)
temp1 = layers.MaxPooling2D((4,temp1.shape[2]))(temp1)

temp2 = layers.Conv2D(i*6, (3,100), padding='same', activation='relu')(inputs)
temp2 = layers.MaxPooling2D((4,temp2.shape[2]))(temp2)

temp3 = layers.Conv2D(i*3, (5,100), padding='same', activation='relu')(inputs)
temp3 = layers.MaxPooling2D((4,temp3.shape[2]))(temp3)

temp4 = layers.Conv2D(i*3, (7,100), padding='same', activation='relu')(inputs)
temp4 = layers.MaxPooling2D((4,temp4.shape[2]))(temp4)

temp5 = layers.Conv2D(i*15, (1,75), padding='same', activation='relu')(inputs)
temp5 = layers.MaxPooling2D((4,temp5.shape[2]))(temp5)

temp6 = layers.Conv2D(i*10, (3,75), padding='same', activation='relu')(inputs)
temp6 = layers.MaxPooling2D((4,temp6.shape[2]))(temp6)

temp7 = layers.Conv2D(i*5, (5,75), padding='same', activation='relu')(inputs)
temp7 = layers.MaxPooling2D((4,temp7.shape[2]))(temp7)

temp8 = layers.Conv2D(i*5, (7,75), padding='same', activation='relu')(inputs)
temp8 = layers.MaxPooling2D((4,temp8.shape[2]))(temp8)

temp9 = layers.Conv2D(i*15, (1,25), padding='same', activation='relu')(inputs)
temp9 = layers.MaxPooling2D((4,temp9.shape[2]))(temp9)

temp10 = layers.Conv2D(i*10, (3,25), padding='same', activation='relu')(inputs)
temp10 = layers.MaxPooling2D((4,temp10.shape[2]))(temp10)

temp11 = layers.Conv2D(i*5, (5,25), padding='same', activation='relu')(inputs)
temp11 = layers.MaxPooling2D((4,temp11.shape[2]))(temp11)

temp12 = layers.Conv2D(i*5, (7,25), padding='same', activation='relu')(inputs)
temp12 = layers.MaxPooling2D((4,temp12.shape[2]))(temp12)

temp = layers.Concatenate()([temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp7,temp8,temp9,temp10,temp11,temp12])
temp = layers.Reshape((temp.shape[3],temp.shape[1],temp.shape[2]))(temp)
temp = layers.Conv2D(i*16, (8,1), activation='relu')(temp)
temp = layers.MaxPooling2D((4,1))(temp)
temp = layers.Flatten()(temp)
temp = layers.AlphaDropout(0.5)(temp)
temp = layers.Dense(100, activation='relu')(temp)
temp = layers.AlphaDropout(0.5)(temp)
outputs = layers.Dense(39, activation='softmax')(temp)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(training_dataset, epochs=300)

model.evaluate(testing_dataset)

#=====================================================================================================================
i = 2
inputs = tf.keras.Input(shape=[128,94,1])
frequency_axis = 1
time_axis = 2
channel_axis = 3

temp1 = layers.Conv2D(i*10, (1,100), padding='same', activation='relu')(inputs)
temp1 = layers.MaxPooling2D((4,temp1.shape[2]))(temp1)

temp2 = layers.Conv2D(i*6, (3,100), padding='same', activation='relu')(inputs)
temp2 = layers.MaxPooling2D((4,temp2.shape[2]))(temp2)

temp3 = layers.Conv2D(i*3, (5,100), padding='same', activation='relu')(inputs)
temp3 = layers.MaxPooling2D((4,temp3.shape[2]))(temp3)

temp4 = layers.Conv2D(i*3, (7,100), padding='same', activation='relu')(inputs)
temp4 = layers.MaxPooling2D((4,temp4.shape[2]))(temp4)

temp5 = layers.Conv2D(i*15, (1,75), padding='same', activation='relu')(inputs)
temp5 = layers.MaxPooling2D((4,temp5.shape[2]))(temp5)

temp6 = layers.Conv2D(i*10, (3,75), padding='same', activation='relu')(inputs)
temp6 = layers.MaxPooling2D((4,temp6.shape[2]))(temp6)

temp7 = layers.Conv2D(i*5, (5,75), padding='same', activation='relu')(inputs)
temp7 = layers.MaxPooling2D((4,temp7.shape[2]))(temp7)

temp8 = layers.Conv2D(i*5, (7,75), padding='same', activation='relu')(inputs)
temp8 = layers.MaxPooling2D((4,temp8.shape[2]))(temp8)

temp9 = layers.Conv2D(i*15, (1,25), padding='same', activation='relu')(inputs)
temp9 = layers.MaxPooling2D((4,temp9.shape[2]))(temp9)

temp10 = layers.Conv2D(i*10, (3,25), padding='same', activation='relu')(inputs)
temp10 = layers.MaxPooling2D((4,temp10.shape[2]))(temp10)

temp11 = layers.Conv2D(i*5, (5,25), padding='same', activation='relu')(inputs)
temp11 = layers.MaxPooling2D((4,temp11.shape[2]))(temp11)

temp12 = layers.Conv2D(i*5, (7,25), padding='same', activation='relu')(inputs)
temp12 = layers.MaxPooling2D((4,temp12.shape[2]))(temp12)

temp = layers.Concatenate()([temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp7, temp8, temp9, temp10, temp11, temp12])

resize_shape = temp.shape[2] * temp.shape[3]
# temp = layers.Reshape((temp.shape[1], resize_shape))(temp)
temp = layers.Reshape((temp.shape[3],temp.shape[1],temp.shape[2]))(temp)
temp = layers.Conv2D(i*16, (8,1), activation='relu')(temp)
output_1 = layers.MaxPooling2D((4,1))(temp)
output_1 = layers.Reshape((output_1.shape[1],output_1.shape[2]*output_1.shape[3]))(output_1)

# recurrent layer
temp = layers.GRU(128, return_sequences=True)(output_1)
temp = layers.GRU(128, return_sequences=False)(output_1)
temp = layers.Dropout(0.3)(temp)

outputs = layers.Dense(39, activation='softmax')(temp)

model = tf.keras.Model(inputs, outputs)
model.summary()
# model.set_weights(weights['arr_2'])
# config = model.get_config()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(training_dataset, epochs=100)

model.evaluate(testing_dataset)

model.save("first_trial",save_format='tf')