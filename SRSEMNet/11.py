import imageio
import numpy as np
from PIL import Image
i =imageio.imread("/home/wind/Documents/LESRCNN/dataset/SEMPIC/SEMPIC_train_HR/000.png")
q= imageio.imread("/home/wind/Documents/DIV2K/DIV2K_train_HR/0001.png")
print(i.shape[::])
print(q.shape[::])