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
#========================================================================================================================================
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

# merge the same labels (ex: outro and Outro)
for i in range(len(rest_all_labels)):
    if rest_all_labels[i] == 'silence':
        rest_all_labels[i] = 'Silence'
    elif rest_all_labels[i] == 'outro':
        rest_all_labels[i] = 'Outro'
    elif rest_all_labels[i] == 'Secondary_theme':
        rest_all_labels[i] = 'Secondary_Theme'
    else :
        rest_all_labels[i] = rest_all_labels[i]

pd.DataFrame(rest_all_labels).value_counts().to_csv("new_sum_of_categories.csv", sep="\t")
#========================================================================================================================================
# The labels we want (2)
rest_labels_2 = ['Verse', 'Chorus', 'Outro', 'Intro', 'Bridge', 'Intrumental',\
                'Pre-Verse', 'Pre-Chorus', 'outro']

rest_all_labels_2 = []
for i in range(len(all_labels)):
    for j in range(len(rest_labels_2)):
        if all_labels[i] == rest_labels_2[j]:
            rest_all_labels_2.append(all_labels[i])
        else :
            rest_all_labels_2 = rest_all_labels_2

for i in range(len(rest_all_labels_2)):
    if rest_all_labels_2[i] == 'outro':
        rest_all_labels_2[i] = 'Outro'
    else :
        rest_all_labels_2[i] = rest_all_labels_2[i]

pd.DataFrame(rest_all_labels_2).value_counts().to_csv("new_sum_of_categories_2.csv", sep="\t")
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