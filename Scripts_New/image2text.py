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
from copy import deepcopy
from init_params import *

item_block_map={
    "?":"Q",
    "X":"P"
}
tile_h = 16
tile_w = 16

path = tile_save_path
im_array = []
with open(symbol_path,"rb") as f:
    symbols_array = pickle.load(f)

for i in range(0,len(symbols_array)):
	name = path + str(i) + ".png"
	im_array.append(name)


def process_im_array(im_array):
    for i in range(0,len(im_array)):
        M=Image.open(im_array[i])
        match = resizeimage.resize_cover(M, [tile_h, tile_w])
        match.save(im_array[i])
        match=cv2.imread(im_array[i])
        match=cv2.cvtColor(match, cv2.COLOR_BGR2GRAY)
        #match = exposure.equalize_adapthist(match)
        im_array[i]=match
    return im_array
    
def imgCompare(imageA, symbols_array, im_array,items=False):
    ssim_array = []
    xyz= resizeimage.resize_cover(imageA, [tile_h, tile_w])
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

def get_enemy_locations(img_rgb,enemies_array,enemies_im_array):
    img_rgb.save(temp_path)
    img_rgb = cv2.imread(temp_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    locations = []
    for i, template in enumerate(enemies_im_array):
        #template = cv.imread('../tiles/30.png',0)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            locations.append([get_tile_x_from_coord(pt[0]), get_tile_x_from_coord(pt[1]),enemies_array[i]])
    return locations
    
def update_text_array_with_enemies(locations, image_matrix, image_matrix_horiz):
    for loc in locations:
        x,y,e=loc
        if image_matrix[x][y] == default_block:
            image_matrix[x][y]=e
            image_matrix_horiz[y][x]=e
    return image_matrix, image_matrix_horiz

def remove_castle_columns(image_matrix, image_matrix_horiz, castle_cols):
    castle_cols=list(set(castle_cols))
    castle_cols.sort()
    for i in reversed(castle_cols):
        image_matrix.pop(i)
        for j in range(len(image_matrix_horiz)):
            image_matrix_horiz[j].pop(i)
    return image_matrix, image_matrix_horiz

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
    maxX=int(maxX/tile_h)*tile_h
    maxY=int(maxY/tile_w)*tile_w

    image_matrix = []
    image_matrix_horiz = []
    flag=False
    castle_cols=[]
    n_cols = 0
    for x in range (16,maxX,tile_h):
        local_column = []
        counter=0
        prev_item=default_block
        for y in range (8,maxY,tile_w):
            if (x+tile_h > maxX or y+tile_w > maxY):
                break
            img2 =im.crop((x,y ,x+tile_h, y+tile_w))
            char = imgCompare(img2, symbols_array, im_array)
            item = imgCompare(img2, items_array ,items_im_array,True)
            char,prev_item = update_char_with_item(char,prev_item)
            prev_item = item

            if char == flag_char:#crop image after flag is reached
                flag=True
            if char == castle_char:
                castle_cols.append(n_cols)
            local_column.append(char)
            insert_in_horiz_array(image_matrix_horiz, char, counter)
            counter+=1
            #end
        if flag: 
            break
        n_cols+=1
        image_matrix.append(local_column)
        #end
    locations = get_enemy_locations(im, enemies_array, enemies_im_array)
    image_matrix, image_matrix_horiz = update_text_array_with_enemies(locations, image_matrix, image_matrix_horiz)
    for i, tile_char in enumerate(long_tile_array):
        template_path = long_tile_im_array[i]
        locations, location_tiles = get_long_tile_locations(im, template_path, tile_char)
        image_matrix, image_matrix_horiz = update_text_for_long_tile(locations, location_tiles, image_matrix, image_matrix_horiz)
    
    image_matrix, image_matrix_horiz = remove_castle_columns(image_matrix, image_matrix_horiz, castle_cols)
    return image_matrix, image_matrix_horiz, n_cols

def get_tile_x_from_coord(x):
    return round((x)/tile_h-1)

def get_tile_y_from_coord(y):
    return round((y-8)/tile_w)

def area(x1,y1,x2,y2):
    return (x2-x1)*(y2-y1)
def cal_iou(box1,box2):
    x1,y1,x2,y2=box1
    x1_,y1_,x2_,y2_=box2
    a=area(x1,y1,x2,y2)
    a_=area(x1_,y1_,x2_,y2_)
    x1=max(x1,x1_)
    y1=max(y1,y1_)
    x2=min(x2,x2_)
    y2=min(y2,y2_)
    intersection = area(x1,y1,x2,y2)
    return intersection/(a+a_-intersection)
def checkiou(boxes, new_box):
    max_iou=0
    for box in boxes:
        iou=cal_iou(new_box,box)
        max_iou=max(max_iou,iou)
    return max_iou

def get_long_tile_locations(img_rgb, template_path, tile_char):
    img_rgb.save(temp_path)
    img_rgb = cv2.imread(temp_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path,0)
    im = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    ret, mask = cv2.threshold(im[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_SQDIFF_NORMED,mask=mask)
    threshold = 0.01
    loc = np.where( res <= threshold)
    ys = list(set(loc[::-1][-1]))
    boxes = []
    box_tiles = []
    for y in ys:
        xs = [x_ for x_,y_ in zip(*loc[::-1]) if y_==y]
        xmin=np.min(xs)
        xmax=np.max(xs)
        if checkiou(boxes, [xmin,y, xmax + w, y + h]) > .9:
            continue
        #cv2.rectangle(img_rgb, (xmin,y), (xmax + w, y + h), (0,0,255), 1)
        boxes.append([xmin,y, xmax + w, y + h])
        box_tiles.append(tile_char)
    return boxes, box_tiles

def update_text_for_long_tile(locations, location_tiles, image_matrix, image_matrix_horiz):
    for i, loc in enumerate(locations):
        x1,y1,x2,y2=loc
        tile_char = location_tiles[i]
        x1,x2 = get_tile_x_from_coord(x1), get_tile_x_from_coord(x2)
        y1,y2 = get_tile_y_from_coord(y1), get_tile_y_from_coord(y2)
        for x in range(x1,x2):
            for y in range(y1,y2):
                if image_matrix[x][y]==default_block:
                    image_matrix[x][y] = tile_char
                    image_matrix_horiz[y][x] = tile_char
    return image_matrix, image_matrix_horiz

#separate items from other symbols
items_array=[x for i,x in enumerate(symbols_array) if x=='p']
items_im_array=[im_array[i] for i,x in enumerate(symbols_array) if x=='p']
enemies_array=[x for i,x in enumerate(symbols_array) if x=='E']
enemies_im_array=[im_array[i] for i,x in enumerate(symbols_array) if x=='E']
long_tile_array = [x for i,x in enumerate(symbols_array) if x=='=']
long_tile_im_array = [im_array[i] for i,x in enumerate(symbols_array) if x=='=']
im_array=[im_array[i] for i,x in enumerate(symbols_array) if x not in ['p', 'E',"="]]
symbols_array=[x for i,x in enumerate(symbols_array) if x not in ['p', 'E',"="]]
im_array = process_im_array(im_array)
enemies_im_array = process_im_array(enemies_im_array)
items_im_array = process_im_array(items_im_array)


levels = os.listdir(folder_levels)
games = os.listdir(folder_levels)
n_cols_array = []

games = [x for x in games if os.path.isdir(folder_levels+x) and x in games_to_use]
for game in games:
    levels = os.listdir(folder_levels+game)
    levels = [x for x in levels if '.gif' in x or '.png' in x]
    
    if not os.path.exists(trans_csv_save_path +"/"+game):
        os.makedirs(trans_csv_save_path +"/"+game)
    if not os.path.exists(orig_csv_save_path +"/"+game):
        os.makedirs(orig_csv_save_path +"/"+game)

    for level in levels:
        file_ = level.split(".")[0]+"_trans.txt"
        file_horiz = level.split(".")[0]+".txt"
        
        if file_ in os.listdir(trans_csv_save_path +"/"+game):
            continue

        im = Image.open(folder_levels+"/"+game+"/"+level)
        x, y, n_cols = process_image(im)
        n_cols_array.append(n_cols)

        with open(trans_csv_save_path +"/"+game+"/"+ file_,"w") as f:
            for col in x:
                f.write("".join(col)+"\n")

        with open(orig_csv_save_path +"/"+game+"/"+ file_horiz,"w") as f:
            for col in y:
                f.write("".join(col)+"\n")

with open(stats_file,"w") as f:
    f.write("Average Number of Columns : "+str(np.mean(n_cols_array)))