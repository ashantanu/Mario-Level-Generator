from PIL import Image,ImageDraw
import sys
import os

folder_path = "../Original_Levels/"
save_folder_path = "../Edited/"
def draw_lines_on_level(folder_path, save_folder_path):
    for level_name in os.listdir(folder_path):
        if '.gif' not in level_name:
            continue
        im = Image.open(folder_path + level_name)
        maxX,maxY = im.size
        X = 16
        Y = 215+1
        draw = ImageDraw.Draw(im)
        while X <= maxX:
            draw.line((X, 0, X, maxY), fill = 255)
            X += 16

        while Y >= 0:
            draw.line((0, Y, maxX, Y), fill=255)
            Y -= 16
        del draw
        im.save(save_folder_path + os.path.splitext(level_name)[0] + "-edited.gif")
        #im = Image.open("Edited/edited-1.gif")
        #im.show()

if __name__ == "__main__":
    folder_path = "../Original_Levels/"
    save_folder_path = "../Edited_New/"
    draw_lines_on_level(folder_path, save_folder_path)
    


