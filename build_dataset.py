import config
import os
from imutils import paths
import shutil


for split in (config.TRAIN, config.TEST, config.VAL):
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
    imagePaths = list(paths.list_images(p))

    for imagePath in imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])]

        outDirPath = os.path.sep.join([config.BASE_PATH, split, label])

        if not os.path.exists(outDirPath):
            os.makedirs(outDirPath)

        p =os.path.sep.join([outDirPath, filename])
        shutil.copy(imagePath, p)