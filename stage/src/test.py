from skimage import feature
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import src.segmentationUtils.Analogy_Computing as Analogy_computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Segmentation as Segmentation
import src.segmentationUtils.Segmentation_Calculator as Segmentation_Calculator
import os
import time
import scipy as sp
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
from multiprocessing import Pool
#import cv2

pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136

sous_repertoire = "sheldon"
tracing = False
dicing = True
dealing = False
reu = True

CNN_OVATION_PATH = "../Data/coupeGap10/tumeur/"
CNN_PATH = "../Data/coupeCnn/tumeur/"
EXPERT_PATH = "../Data/coupeExpert/tumeur/"

CNN_TO_CNN_PATH = "../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/"
CNN_TO_OVATION_PATH = "../Data/basic/" + sous_repertoire + "/cnn_ovation/tumeur/"
OVATION_TO_CNN_PATH = "../Data/basic/" + sous_repertoire + "/ovation_cnn/tumeur/"
OVATION_TO_OVATION_PATH = "../Data/basic/" + sous_repertoire + "/ovation_ovation/tumeur/"

if not os.path.exists(CNN_TO_CNN_PATH):
    os.makedirs(CNN_TO_CNN_PATH)
if not os.path.exists(CNN_TO_OVATION_PATH):
    os.makedirs(CNN_TO_OVATION_PATH)
if not os.path.exists(OVATION_TO_CNN_PATH):
    os.makedirs(OVATION_TO_CNN_PATH)
if not os.path.exists(OVATION_TO_OVATION_PATH):
    os.makedirs(OVATION_TO_OVATION_PATH)


def to_trace(k):
    indice = [k + i for i in range(-5, 6) if k + i >= min and k + i <= max]

    expert_pivot = Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(EXPERT_PATH + str(k) + ".png"))
    expert_pivot = Pic_Treatments.dim1(expert_pivot)
    cnn_pivot = Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(CNN_PATH + str(k) + ".png"))
    cnn_pivot = Pic_Treatments.dim1(cnn_pivot)

    for i in indice:
        img_liss_cnn_cnn = Pic_Openings.ouvrirImage(CNN_PATH+str(i)+".png")
        img_liss_cnn_cnn = Pic_Treatments.niveauGris(img_liss_cnn_cnn)
        img_liss_cnn_cnn = Pic_Treatments.dim1(img_liss_cnn_cnn)
        im_red = Analogy_computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=cnn_pivot,cnn2=img_liss_cnn_cnn,methodType="sheldon")
        new_img = Pic_Treatments.dim3(im_red)
        plt.imshow(im_red)


        Pic_Openings.sauvegardeImage(nomFic= CNN_TO_CNN_PATH+str(i)+".png", image=new_img)
        print(i)
    return 1


def to_calcul(i):
    expert_i = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(EXPERT_PATH + str( i) + ".png")))
    cnn_i = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(CNN_PATH + str(i) + ".png")))
    ovation_i = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(CNN_OVATION_PATH + str(i) + ".png")))

    imC = Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../Data/basic/c/cnn_cnn/tumeur/"+str(i)+".png"))
    imSheldon = Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../Data/basic/sheldon/cnn_cnn/tumeur/"+str(i)+".png"))


    a = Frame_Performances.indice_double(expert_i,ovation_i)
    Dice = [[], [], [],[]]  # oracle,cnn,ovation,lissage simple
    Jacc = [[], [], [],[]]
    Dice[0].append(a[0])
    Jacc[0].append(a[1])
    b = Frame_Performances.indice_double(expert_i, cnn_i)
    Dice[1].append(b[0])
    Jacc[1].append(b[1])
    c = Frame_Performances.indice_double(expert_i, imC)
    Dice[2].append(c[0])
    Jacc[2].append(c[1])
    d = Frame_Performances.indice_double(expert_i,imSheldon)
    Dice[3].append(d[0])
    Jacc[3].append(d[1])

    return Dice, Jacc

if __name__ == '__main__':
    start = time.time()
    print(os.cpu_count())
    p = Pool(os.cpu_count()-2)

    if tracing:
        for rep in p.map(to_trace,pivots):
            1

    gold = [i for i in range(1,137)]

    D = [[], [], [], []]  # oracle,cnn,ovation,contour
    J = [[], [], [], []]  # oracle, cnn,ovation,contour
    if dicing:
        for rep in p.map(to_calcul,gold):
            D[0] += rep[0][0]
            D[1] += rep[0][1]
            D[2] += rep[0][2]
            D[3] += rep[0][3]

        print(len(gold),len(D[0]),len(D[1]),len(D[2]),len(D[3]))
        plt.clf()
        plt.title("Comparaison Dice method C/sheldon")
        plt.plot(gold, D[0], label="Ovation")
        plt.plot(gold, D[1], label="CNN")
        plt.plot(gold, D[2], label="c")
        plt.plot(gold, D[3], label="sheldon")
        for k in pivots:
            plt.axvline(x=k, color='red')
        plt.legend()
        plt.show()
        plt.clf()

    if dealing:
        im1 = Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../Data/basic/c/cnn_cnn/tumeur/99.png"))
        a,b,c,d = Pic_Treatments.get_rectangle_coords_minimal(im1)
        im1 = Pic_Treatments.get_compressed_image_from_rectangle_coords(image=im1,line_min=a,line_max=b,col_min=c,col_max=d)
        im2 = Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../Data/basic/c/cnn_cnn/tumeur/99.png"))
        a,b,c,d = Pic_Treatments.get_rectangle_coords_minimal(im2)
        im2 = Pic_Treatments.get_compressed_image_from_rectangle_coords(image=im2,line_min=a,line_max=b,col_min=c,col_max=d)
        im1_2 = Pic_Treatments.fusion_double_image(im1,im2)


        im3 = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeExpert/tumeur/101.png")))
        im4 = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeCnn/tumeur/101.png")))
        im5 = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeCnn/tumeur/99.png")))
        a, b, c, d = Pic_Treatments.get_rectangle_coords_minimal(im3)
        im3 = Pic_Treatments.get_compressed_image_from_rectangle_coords(image=im3, line_min=a, line_max=b, col_min=c,col_max=d)
        a, b, c, d = Pic_Treatments.get_rectangle_coords_minimal(im4)
        im4 = Pic_Treatments.get_compressed_image_from_rectangle_coords(image=im4, line_min=a, line_max=b, col_min=c,col_max=d)
        a, b, c, d = Pic_Treatments.get_rectangle_coords_minimal(im5)
        im5 = Pic_Treatments.get_compressed_image_from_rectangle_coords(image=im5, line_min=a, line_max=b, col_min=c,col_max=d)

        fig = plt.figure(figsize=(4,4))
        columns = 3
        rows = 2

        for i in range(1,columns*rows+1):
            fig.add_subplot(rows,columns,i)
            if i == 1:
                img = Pic_Treatments.dim3(im3)
            elif i ==2:
                img = Pic_Treatments.dim3(im4)
            elif i == 3:
                img = Pic_Treatments.dim3(im5)
            elif i == 4:
                img = Pic_Treatments.dim3(im1)
            elif i == 5:
                img = im1_2
            else:
                img = Pic_Treatments.dim3(im2)

            plt.imshow(img)

        plt.show()

    if reu:
        b = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeExpert/tumeur/101.png")))
        a = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeCnn/tumeur/101.png")))
        c = Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../Data/coupeCnn/tumeur/99.png")))
        d = Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../Data/basic/c/cnn_cnn/tumeur/99.png"))
        Pic_Treatments.plot_a_b_c_d(imA=a,imB=b,imC=c,imD=d)

    end = time.time()
    print("delay", end - start)