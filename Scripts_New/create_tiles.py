from resizeimage import resizeimage
from PIL import Image,ImageDraw
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
import sys
from PIL import Image,ImageDraw
import sys
import os
import pickle
from init_params import * 

#make lines on sheet
def make_lines_on_sheet(path):
    im = Image.open(path)
    maxX,maxY = im.size
    print("sheet size:",maxX,maxY)
    X = 16
    Y = maxY
    draw = ImageDraw.Draw(im)
    while X <= maxX:
        draw.line((X, 0, X, maxY), fill = 'white')
        X += 16

    while Y >= 0:
        draw.line((0, Y, maxX, Y), fill='white')
        Y -= 16
    del draw
    new_path = ".".join(path.split(".")[:-1])+'-edited.png'
    im.save(new_path)
    #im.show()
    return new_path, (maxX,maxY)

def get_tile_matrix(sheet,maxX, maxY, xcut=1e4,ycut=1e4):
    xcut=min(xcut,maxX)
    ycut=min(ycut,maxY)
    img =Image.open(sheet)
    tile_array = []
    for y in range (0,maxY,16):
        array_ = []
        for x in range (0,maxX,16):
                if (x > xcut or y > ycut):
                        break
                img2 =img.crop((x,y ,x+16, y+16))
                array_.append(img2)
        tile_array.append(array_)
    return tile_array

sheet_, (maxX,maxY) = make_lines_on_sheet(sheet)
print("saved path:",sheet_)
tile_array = get_tile_matrix(sheet,maxX,maxY)

enemies_sheet_, (maxX,maxY) = make_lines_on_sheet(enemies_sheet)
print("enemies saved path:",enemies_sheet)
enemies_tile_array = get_tile_matrix(enemies_sheet,maxX,maxY)

items_sheet_, (maxX,maxY) = make_lines_on_sheet(items_sheet)
print("items saved path:",items_sheet_)
items_tile_array = get_tile_matrix(items_sheet,maxX,maxY)

#figure out each tile
char2tile = []
tile_types = ['solid','enemy','destructible block', 
              'question mark with coin',
             'question mark with power',
             'coin',
             'bullet shooter top', 'bullet shooter column',
             'left pipe','right pipe', 'top left pipe','top right pipe',
             'empty'
              #,'castle','flag'
             ]
{
    "tiles" : {
        "X" : ["solid","ground"],
        "S" : ["solid","breakable"],
        "P" : ["solid","power-up"],
        "H" : ["solid","moving","platform"],
        "-" : ["passable","empty"],
        "?" : ["solid","question block", "coin"],
        "Q" : ["solid","question block", "power-up"],
        "E" : ["enemy","damaging","hazard","moving"],
        "<" : ["solid","top-left pipe","pipe"],
        ">" : ["solid","top-right pipe","pipe"],
        "[" : ["solid","left pipe","pipe"],
        "]" : ["solid","right pipe","pipe"],
        "o" : ["coin","collectable","passable"],
        "p" : ["power","collectable","passable"],
        "B" : ["Cannon top","cannon","solid","hazard"],
        "b" : ["Cannon bottom","cannon","solid"],
        
        "F" : ["Flag","solid","cutoff"],
        "C" : ["Castle","empty","cutoff"]
    }
}
level_type = ['above ground', 'under ground', 'underwater']
#maybe consider castle as background as well - crop everything after flag
#problem with castle tiles which are same a destructible tiles
#missing flag tile
blocks=[
    [tile_array[0][0],"X"],
    [tile_array[0][1],"S"],
    [tile_array[0][3],"X"],
    [tile_array[0][24],"?"],
    [tile_array[0][25],"?"],
    [tile_array[0][26],"?"],
    [tile_array[1][0],"X"],
    [tile_array[1][1],"X"],
    [tile_array[1][2],"X"],
    [tile_array[27][3],"-"],
    [tile_array[8][5],"X"],
    [tile_array[8][6],"X"],
    [tile_array[8][7],"X"],
    [tile_array[0][15],"X"],
    #[tile_array[1][5],"X"],

    [items_tile_array[8][4], "="],
    
    [items_tile_array[6][0],'o'],
    [items_tile_array[6][1],'o'],
    [items_tile_array[7][0],'o'],
    [items_tile_array[7][1],'o'],
    [items_tile_array[6][9],'o'],
    
]
pipes=[
    [tile_array[8][0],"<"],
    [tile_array[8][1],">"],
    [tile_array[9][0],"["],
    [tile_array[9][1],"]"],
    
    [tile_array[8][2],"^"],
    [tile_array[8][3],"~"],
    [tile_array[9][2],"v"],
    [tile_array[9][3],"_"],
    [tile_array[8][4],"l"],
    [tile_array[9][4],"L"]
]
cutoff_tile = [
    [tile_array[8][16], "F"],
    [tile_array[0][11], "C"],
    [tile_array[1][11], "C"],
]

enemies = [
    [enemies_tile_array[1][0],"E"],
    [enemies_tile_array[1][6],"E"],
    [enemies_tile_array[1][9],"E"],
    [enemies_tile_array[3][6],"E"],
    [enemies_tile_array[3][9],"E"],
    #[enemies_tile_array[1][12],"E"],
    #[enemies_tile_array[1][13],"E"],
    #[enemies_tile_array[0][16],"E"],
    
    [enemies_tile_array[1][46],"E"]
    
]

items = [
    [items_tile_array[0][0],'p'],
    [items_tile_array[1][0],'p'],
    [items_tile_array[2][0],'p'],
    [items_tile_array[3][0],'p'],
    [items_tile_array[0][11],'p'],
    [items_tile_array[0][10],'p']
]
char2tile=char2tile+blocks+pipes+cutoff_tile+enemies+items

##Save these tiles
counter = 0
tile_char_array = []
for [tile, char] in char2tile:
    tile.save(tile_save_path+str(counter)+".png")
    tile_char_array.append(char)
    counter+=1
with open(symbol_path,"wb") as f:
    pickle.dump(tile_char_array,f, protocol=pickle.HIGHEST_PROTOCOL)
    