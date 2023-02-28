# AVOR dataset is based on KITTI 3D Object Detection Dataset

Autonomous Vehicle Object Recognition dataset for training Cross-Modal Verification ([CMV](https://github.com/Sliverk/LITIS_CrossModal)) model .


## Step 1. Prepare KITTI 3D Object Detection Dataset

URL : [Link to KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Unzip *data_split.zip* and copy it to the dataset root folder.

## Step 2. Run 0105_GenAVOR.py

```bash
python3 0105_GenAVOR.py [ |train|val|test|trainval]
```

