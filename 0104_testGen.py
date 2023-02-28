import os
import sys
import glob
from PIL import Image
import numpy
import random

def get_label_annos(rootpath, filelist):
    annos = []
    for filename in filelist:
        filepath = rootpath + filename + '.txt'
        with open(filepath, 'r') as f:
            lines = f.readlines()
        labelcontext = []
        for line in lines:
            context = line.strip().split(' ')
            classname = context[0]
            # if classname == 'DontCare': continue
            if classname == 'Misc': continue
            tmp = list(map(float,context[4:8]))
            tmp = list(map(int,tmp))
            labelcontext.append(tmp)
        annos.append(labelcontext)
    return annos


def get_background_bbox2d(imgsize, bbox2d):
    def checkiou(bbox1, bbox2):
        zx = abs(bbox1[0]+bbox1[2]-bbox2[0]-bbox2[2])
        x = abs(bbox1[0]-bbox1[2]) + abs(bbox2[0]-bbox2[2])
        zy = abs(bbox1[1]+bbox1[3]-bbox2[1]-bbox2[3])
        y = abs(bbox1[1]-bbox1[3]) + abs(bbox2[1]-bbox2[3])
        if zx < x and zy < y: return True
        else: return False

    backgroundbbox = []
    num_bbox2d = len(bbox2d)
    queue_imgindex = [x for x in range(num_bbox2d)]
    for rec in bbox2d:
        targetsize = [rec[2]-rec[0], rec[3]-rec[1]]
        random.shuffle(queue_imgindex)
        ISOK = False
        for index in queue_imgindex:
            plannumlist = [x for x in range(8)]
            random.shuffle(plannumlist)
            for plannum in plannumlist:
                # right
                if plannum == 0:
                    x1 = bbox2d[index][2]
                    y1 = bbox2d[index][1]
                # bottom
                elif plannum == 1:
                    x1 = bbox2d[index][0]
                    y1 = bbox2d[index][3]
                # left
                elif plannum == 2:
                    x1 = bbox2d[index][0] - targetsize[0]
                    y1 = bbox2d[index][1]
                # up
                elif plannum == 3:
                    x1 = bbox2d[index][0]
                    y1 = bbox2d[index][1] - targetsize[1]
                # up right
                elif plannum == 4:
                    x1 = bbox2d[index][2]
                    y1 = bbox2d[index][1] - targetsize[1]
                # bottom right
                elif plannum == 5:
                    x1 = bbox2d[index][2]
                    y1 = bbox2d[index][3]
                # bottom left
                elif plannum == 6:
                    x1 = bbox2d[index][0] - targetsize[0]
                    y1 = bbox2d[index][3]
                # up left
                elif plannum == 7:
                    x1 = bbox2d[index][0] - targetsize[0]
                    y1 = bbox2d[index][1] - targetsize[1]
                x2 = x1 + targetsize[0]
                y2 = y1 + targetsize[1]
                # check correct
                if min(x1,y1,x2,y2) < 0: continue
                if x2 > imgsize[0] or y2 > imgsize[1]: continue
                for q in queue_imgindex:
                    ISOK = True
                    if checkiou([x1,y1,x2,y2], bbox2d[q]) == True: 
                        ISOK = False
                        break
                if ISOK == True: 
                    backgroundbbox.append([x1,y1,x2,y2])
                    break
            if ISOK == True: break
    return backgroundbbox    

def get_image_roi(imagerootpath, bbox2dlist, filelist, outpath):

    def cut_and_save(srcimg, bbox2d, outpath):
        index = 0
        for rec in bbox2d:
            maxlen = max(rec[3]-rec[1], rec[2]-rec[0])
            if maxlen <= 15: continue
            part = srcimg.crop((rec[0],rec[1],rec[2],rec[3]))            
            # blankimg = Image.new('RGB', (maxlen,maxlen))
            # blankimg.paste(part, (0,0))
            part.save(f'{outpath}_{index}_{rec[0]}.png')
            index += 1
    
    # select one image
    outforepath = outpath + '1/'
    if not os.path.exists(outforepath): os.makedirs(outforepath)
    else: os.system(f'rm {outforepath}*.png')
    outbackpath = outpath + '2/'
    if not os.path.exists(outbackpath): os.makedirs(outbackpath)
    else: os.system(f'rm {outbackpath}*.png')

    for index in range(len(filelist)):
        filename = filelist[index]
        imagepath = imagerootpath + filename + '.png'
        img = Image.open(imagepath)
        # cut foreground
        cut_and_save(img, bbox2dlist[index], outforepath+filename)
        # cut background
        bkbbox2d = get_background_bbox2d(img.size, bbox2dlist[index])
        # print(bkbbox2d)
        cut_and_save(img, bkbbox2d, outbackpath+filename)
        # if index == 10: break

def main():
    # PATH info
    KITTI_PATH = './data/kitti_3d'
    MODE = sys.argv[1] if len(sys.argv)>1 else 'train'

    SPLIT_FILE = f'{KITTI_PATH}/split/{MODE}.txt'
    LABEL_PATH = f'{KITTI_PATH}/training/label_2/'
    IMAGE_PATH = f'{KITTI_PATH}/training/image_2/'

    OUTPUT_PATH = f'{KITTI_PATH}/classification/backfore/{MODE}/'
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

    # Get file list
    with open(SPLIT_FILE, 'r') as f:
        filelist = f.readlines()
    filelist = [x.strip() for x in filelist]
    
    # Get label annotations
    annos = get_label_annos(LABEL_PATH, filelist)

    # Cut and save foreground and background
    get_image_roi(IMAGE_PATH, annos, filelist, OUTPUT_PATH)

if __name__ == '__main__':
    main()
