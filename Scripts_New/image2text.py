from resizeimage import resizeimage
from PIL import Image,ImageDraw
from skimage import exposure
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
import sys
import pickle
from create_tiles import tile_save_path, symbol_path
from copy import deepcopy

#folder_levels = "../Edited_New/"
folder_levels = "../Original_Levels/"

orig_csv_save_path = "../levels_CSV_New/"
trans_csv_save_path = "../levels_transposed_New/"

temp_path = "../level1specs/temp.png"
default_block = "-"
flag_char = "F"
stats_file = "./stats.txt"
item_block_map={
    "?":"Q",
    "X":"P"
}


path = tile_save_path
im_array = []
with open(symbol_path,"rb") as f:
    symbols_array = pickle.load(f)

for i in range(0,len(symbols_array)):
	name = path + str(i) + ".png"
	im_array.append(name)

#separate items from other symbols
items_array=[x for i,x in enumerate(symbols_array) if x=='p']
items_im_array=[im_array[i] for i,x in enumerate(symbols_array) if x=='p']
symbols_array=[x for i,x in enumerate(symbols_array) if x!='p']
im_array=[im_array[i] for i,x in enumerate(symbols_array) if x!='p']

    
levels = os.listdir(folder_levels)
levels = [x for x in levels if '.gif' in x]
#for level in levels:
level = 'mario-1-1.gif'


#####################################
im = Image.open(folder_levels+level)

def process_im_array(im_array):
    for i in range(0,len(im_array)):
        M=Image.open(im_array[i])
        match = resizeimage.resize_cover(M, [16, 16])
        match.save(im_array[i])
        match=cv2.imread(im_array[i])
        match=cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
        #match = exposure.equalize_adapthist(match)
        im_array[i]=match
    return im_array
    
def imgCompare(imageA, symbols_array, im_array,items=False):
    ssim_array = []
    xyz= resizeimage.resize_cover(imageA, [16, 16])
    xyz.save(temp_path)
    xyz=cv2.imread(temp_path)
    xyz=cv2.cvtColor(xyz, cv2.COLOR_BGR2GRAY)
    if items:
        xyz = cv2.Canny(xyz,0,10)
    #xyz = exposure.equalize_adapthist(xyz)
    for match in im_array:
        if items:
            match = cv2.Canny(match,0,10)
        ssim_ = structural_similarity(xyz,match)
        ssim_array.append(ssim_)

    max_val = max(ssim_array)
    max_symbol = default_block
    if(max_val >= 0.7):
        max_symbol = symbols_array[ssim_array.index(max(ssim_array))]
    return max_symbol

def insert_in_horiz_array(array, char, index):
    if len(array)<=index:
        array.append([])
    array[index].append(char)

def update_char_with_item(char,prev_item):
    if prev_item!=default_block and char in item_block_map:
        char=item_block_map[char]
    elif prev_item!=default_block:
        print("Unknown solid for item")#TODO: add this as info logger
    prev_item=default_block
    return char, prev_item

def process_image(im):
    (maxX,maxY) = im.size
    maxX=int(maxX/16)*16
    maxY=int(maxY/16)*16

    image_matrix = []
    image_matrix_horiz = []
    flag=False
    n_cols = 0
    for x in range (16,maxX,16):
        local_column = []
        counter=0
        prev_item=default_block
        for y in range (8,maxY,16):
            if (x+16 > maxX or y+16 > maxY):
                break
            img2 =im.crop((x,y ,x+16, y+16))
            char = imgCompare(img2, symbols_array, im_array)
            item = imgCompare(img2, items_array ,items_im_array,True)
            char,prev_item = update_char_with_item(char,prev_item)
            prev_item = item

            if char == flag_char:#crop image after flag is reached
                flag=True
            local_column.append(char)
            insert_in_horiz_array(image_matrix_horiz, char, counter)
            counter+=1
            #end
        if flag:
            break
        n_cols+=1
        image_matrix.append(local_column)
        #end
    return image_matrix, image_matrix_horiz, n_cols

im_array = process_im_array(im_array)
items_im_array = process_im_array(items_im_array)
x, y, n_cols = process_image(im)
file_ = level.split(".")[0]+"_trans.txt"
file_horiz = level.split(".")[0]+".txt"
with open(trans_csv_save_path + file_,"w") as f:
    for col in x:
        f.write("".join(col)+"\n")

with open(orig_csv_save_path + file_horiz,"w") as f:
    for col in y:
        f.write("".join(col)+"\n")

with open(stats_file,"w") as f:
    f.write("Average Number of Columns : "+str(n_cols))