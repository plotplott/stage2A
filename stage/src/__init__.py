import src.segmentationUtils.Analogy_Computing as Analogy_computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Segmentation as Segmentation
import src.segmentationUtils.Segmentation_Calculator as Segmentation_Calculator
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy as sp
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
from multiprocessing import Pool
import math


# Import libraries

D=[]
D.append([1,2,3,4])
D.append([i for i in range(1,400,2)])
D.append([np.exp(i) for i in range(1,400,2)])
D.append([np.sqrt(i) for i in range(1,400,2)])

# Creating dataset
data = [D[0],D[1],D[2],D[3]]
fig = plt.figure(figsize=(10, 7))

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = ax.boxplot(data)

# show plot
#plt.show()