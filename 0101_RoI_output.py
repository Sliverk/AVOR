import os
import sys
import glob
import cv2


dataset_path = '/home/zz/locald/dataset/kitti_3d/'
split = sys.argv[1] if len(sys.argv)>1 else 'val'

label_path = os.path.join(dataset_path,'training/label_2/')
image_path = os.path.join(dataset_path,'training/image_2/')

split_file = f'./split/{split}.txt'
output_path = os.path.join(dataset_path, f'classification/{split}')

num_classes = 6

for i in range(num_classes):
    if not os.path.exists(os.path.join(output_path,f'{i+1}')) : os.mkdir(os.path.join(output_path,f'{i+1}'))

def label_anaylse(labelpathlist):
    label_dict = {}
    for label in labelpathlist:
        with open(label, 'r') as f:
            lines = f.readlines()
        for ln in lines:
            value = ln.strip().split(' ')
            if not value[0] in label_dict.keys():
                label_dict[value[0]] = 1
            else:
                label_dict[value[0]] += 1            
    print(label_dict)
    # {'Pedestrian': 4487, 'Truck': 1094, 'Car': 28742, 'Cyclist': 1627, 'DontCare': 11295, 'Misc': 973, 'Van': 2914, 'Tram': 511, 'Person_sitting': 222}


def main():
    # label_anaylse(label_path)
    
    # 1 person: 'Pedestrian': 4487, 'Person_sitting': 222
    # 2 cyclist: 'Cyclist': 1627
    # 3 car: 'Car': 28742
    # 4 truck: 'Truck': 1094
    # 5 van: 'Van': 2914
    # 6 tram: 'Tram': 511
    class_dict = {'Pedestrian':1, 'Person_sitting':1, 'Cyclist': 2, 'Car': 3, 'Truck': 4, 'Van': 5, 'Tram': 6}
    def read_label(labelpath):
        with open(labelpath, 'r') as f:
            lines = f.readlines()
        labelcontext = []
        for line in lines:
            context = line.strip().split(' ')
            classname = context[0]
            if classname == 'DontCare': continue
            if classname == 'Misc': continue
            context[4:8] = list(map(float,context[4:8]))
            context[4:8] = map(int,context[4:8])
            context[0] = class_dict[context[0]]
            labelcontext.append([context[0]]+context[4:8])
        return labelcontext

    def cutRoI(filename, records, output):
        sourceimg = cv2.imread(image_path+filename+'.png')
        # print(image_path+filename+'.png')
        index = 1
        for rec in records:
            partimg = sourceimg[rec[2]:rec[4],rec[1]:rec[3]]
            cv2.imwrite(output+f'/{rec[0]}/'+filename+f'_{index}.jpg', partimg)
            index += 1
        

    with open(split_file, 'r') as f:
        splitlines = f.readlines()
    
    for line in splitlines:
        filename = line.strip()
        labelpath = label_path + filename + '.txt'
        rec = read_label(labelpath)
        cutRoI(filename, rec, output_path)
        # break





if __name__ == '__main__':
    main()
    print('Done')

