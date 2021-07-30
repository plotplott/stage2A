
import src.segmentationUtils.Analogy_Computing as Analogy_computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Segmentation as Segmentation
import src.segmentationUtils.Segmentation_Calculator as Segmentation_Calculator

import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from os import path
from scipy import ndimage
from skimage import measure

file = "../Data/lissage_after/95from90.png"
print(file)



if not os.path.exists(file):
    raise ValueError(str("file "+file +" doesn't exists"))

mat_to_contour = Pic_Treatments.get_image(file)
row,col = mat_to_contour.shape[0],mat_to_contour.shape[1]
for i in range(row):
    mat_to_contour[0][i] = 1
    mat_to_contour[row-1][i]=1
    mat_to_contour[i][0]=1
    mat_to_contour[i][row-1]=1

#L= Analogy_Computing.test_contouring(mat_to_contour)

contours = measure.find_contours(mat_to_contour,0.99)
print(type(contours[0]),len(contours[0]))
for i in contours[1]:
    print(i)

# Find contours at a constant value of 0.8
#contours = measure.find_contours(r, 0.8)

# Display the image and plot all contours found
fig, ax = plt.subplots()

for contour in contours:
    print(contour.shape)
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
#plt.show()
plt.clf()

mat1 = np.zeros((512,512))

for i in contours[1]:
    x = round(i[0])
    y = round(i[1])
    #y = row - y
    #print(type(i),i[0],i[1],i,x,y)
    mat1[int(x)][int(y)] = 1

#more efficient this way
start = time.time()
x = np.argwhere(mat1 >0).maximum()
print(x,np.sum(mat1),time.time()-start)

mat2 = Pic_Treatments.dim3(mat1)
Pic_Openings.sauvegardeImage("./lol1.png",mat2)

for i in range(row):
    args = np.argwhere(mat1[:,i] > 0)
    if args.shape[0] != 0:
        print(args)
        debut, fin = args.minimum(), args.maximum()
        print(debut,fin)
        for j in range(debut+1,fin):
            mat1[j][i] = 1

    args = np.argwhere(mat1[i,:] > 0)
    if args.shape[0] != 0:
        debut, fin = args.minimum(), args.maximum()
        print(debut,fin)
        for j in range(debut+1,fin):
            mat1[i][j] = 1


#start = time.time()
#print(mat1.tolist().count(1),time.time()-start)
mat2 = Pic_Treatments.dim3(mat1)
Pic_Openings.sauvegardeImage("./lol.png",mat2)


mat3 = np.zeros((512,512))
for i in contours[1]:
    x = round(i[0])
    y = round(i[1])
    #y = row - y
    #print(type(i),i[0],i[1],i,x,y)
    mat1[int(x)][int(y)] = 1

row,col = mat3.shape[0],mat3.shape[1]

for i in range(row):

    ligne = np.argwhere(mat3[:,i] > 0)
    if args.shape[0] != 0:
        for j in ligne:
            print("hfekmfkljelkjfeffm")
            print(j[0])
            debut, fin = args.minimum(), args.maximum()
            print(debut,fin)
mat3 = Pic_Treatments.dim3(mat1)
Pic_Openings.sauvegardeImage("./lol2.png",mat3)




'''
import cv2


img = cv2.imread("../Data/lissage_after/95from90.png", cv2.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 50
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

'''
"""
L_class=[]
for k in L:
    k_class = Analogy_Computing.Contour(k)
    L_class.append(k_class)

print(len(L_class))
my_contour = L_class[1]
print(my_contour.getContour())

mat1 = my_contour.fromContourToMatBord()
mat2 = Pic_Treatments.dim3(mat1)
Pic_Openings.sauvegardeImage("./lol.png",mat2)

"""


