# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import csv
import numpy as np
from scipy import ndimage

def load_images(folder_fundus, folder_fundus_2, img_size):
  fundus_list_res = []#fundus images
  fundus_list_2_res = []#ROI images   
  file_name_list = sorted(os.listdir(folder_fundus))#img file name --> 0001/0002 ....
  #print(len(file_name_list))  
  for file_fundus in file_name_list:

    img_fundus = cv2.imread(os.path.join(folder_fundus,file_fundus))#Read fundus Image
    img_fundus_2 = cv2.imread(os.path.join(folder_fundus_2,file_fundus))#Read ROI image
    
    b,g,r = cv2.split(img_fundus)      
    img_fundus = cv2.merge([r,g,b])
    img_fundus_res = cv2.resize(img_fundus, (img_size,img_size))#resize
    img_fundus_res = (img_fundus_res/255.0).astype("float32")

    b,g,r = cv2.split(img_fundus_2)      
    img_fundus_2 = cv2.merge([r,g,b])
    img_fundus_2_res = cv2.resize(img_fundus_2, (img_size,img_size))#resize
    img_fundus_2_res = (img_fundus_2_res/255.0).astype("float32")

    fundus_list_res.append(img_fundus_res)#resized
    fundus_list_2_res.append(img_fundus_2_res)#resized
  return fundus_list_res, fundus_list_2_res

def load_fundus_images(path,folder_fundus, folder_fundus_2, conj,img_size):#return image and label arrays 
  os.chdir(path)
  if (conj==1):#test split

    fundus_test,ROI_test = load_images(folder_fundus, folder_fundus_2, img_size)
    return(np.array(fundus_test), np.array(ROI_test))

  if (conj==0):#train split
    label_file_path = os.path.join(path, "train.csv")#gamma labels
    #copy from csv to a list
    data_list = []

    with open(label_file_path,'r') as f:  
      lines=csv.reader(f)  
      for key, line in enumerate(lines):  
          data_list.append(line)
    
    non_list = []
    early_list = []
    mid_adv_list = []
    labels_list = []

    for item in data_list:#copy labels to a list
      if (item[1]=='1'):
        non_list.append(item[0])
        labels_list.append(0)
      if (item[2]=='1'):
        early_list.append(item[0])
        labels_list.append(1)
      if (item[3]=='1'):
        mid_adv_list.append(item[0])
        labels_list.append(2)

    fundus_train,ROI_train = load_images(folder_fundus, folder_fundus_2, img_size)

    return(np.array(fundus_train),np.array(ROI_train),np.array(labels_list))
    
#Read OCT scans
def read_file(folder_oct):
  oct_list = []
  for slic in range(0,256):#256 slices
    file_oct = str(slic) + '_image.jpg'
    img_oct = cv2.imread(os.path.join(folder_oct,file_oct))
    img_oct = cv2.bilateralFilter(img_oct,15,75,75)#d, sigma color, sigma coordinate
    #img_oct = cv2.cvtColor(img_oct, cv2.COLOR_BGR2GRAY)
    b,g,r = cv2.split(img_oct)      
    img_oct = cv2.merge([r,g,b])
    img_oct = (img_oct/255.0).astype("float32")
    oct_list.append(img_oct)
  return np.array(oct_list, copy = False)

def resize_volume(img, IMAGE_SIZE, depth):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = depth
    desired_width = IMAGE_SIZE
    desired_height = IMAGE_SIZE
    desired_channel = 3
    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
    current_channel = img.shape[-1]
    
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    channel = current_channel / desired_channel
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    channel_factor = 1 / channel
  
    img = ndimage.zoom(img, (depth_factor,  width_factor, height_factor, channel_factor), order=1)
    return img

def process_scan(path, IMAGE_SIZE, depth):
    """Read and resize volume"""
    # Read scan
    volume = read_file(path)
    volume = resize_volume(volume, IMAGE_SIZE, depth)#depth, Image_size
    return volume

def dir_octs(path_dataset,folder_octs, conj):#path to volumes
  path_dataset_oct = os.path.join(path_dataset,folder_octs)
  
  if(conj==0):#train split
    scan_paths = [
        os.path.join(os.getcwd(), path_dataset_oct, '000'+str(x), '000'+str(x))
        #for x in os.listdir(path_dataset)
        for x in range(1,100) if x<10
    ]
    scan_paths = scan_paths+ [
        os.path.join(os.getcwd(), path_dataset_oct, '00'+str(x), '00'+str(x))
        #for x in os.listdir(path_dataset)
        for x in range(1,100) if x>=10 and x<100
    ]

    scan_paths = scan_paths+ [
        os.path.join(os.getcwd(), path_dataset_oct, '0'+str(x), '0'+str(x))
        #for x in os.listdir(path_dataset)
        for x in range(1,101) if x>=100
    ]

  if(conj==1):#test split
    scan_paths = [
        os.path.join(os.getcwd(), path_dataset_oct, '0'+str(x), '0'+str(x))
        #for x in sorted(os.listdir(path_dataset))
        for x in range(101,201)
    ]
  return scan_paths

def converte(item):
  if item<10:
    item = "000"+ str(item)
  elif item<100:
    item = "00"+ str(item)
  else:
    item = str(item)
    item = "0"+ item
  return item

def padroniza_resultado(y):#one hot encoding
    results = []
    for i in range(len(y)):
        row = []
        item = converte(i+101)
        row.append(item)
        if(y[i]==0):
            row.append(1)
            row.append(0)
            row.append(0)
        if(y[i]==1):
            row.append(0)
            row.append(1)
            row.append(0)
        if(y[i]==2):
            row.append(0)
            row.append(0)
            row.append(1)
        results.append(row)
    return results