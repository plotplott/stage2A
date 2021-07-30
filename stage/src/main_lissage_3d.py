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

param = "de"
pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136

sous_repertoire = "lissage_3D"

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

pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136


def get_trio(k):
    '''
    choper une liste a 3 elements: centre, contour 1, contour 2
    :param k: numero du fichier voulu
    :return: [image centre, contour 1, contour 2]
    '''
    im1 = Pic_Treatments.get_image("../Data/basic/lissage_base_red/ovation_ovation/tumeur/"+str(k)+".png")
    im2 = Pic_Treatments.get_image("../Data/basic/lissage_base_red/ovation_ovation/tumeur/"+str(k-1)+".png")
    im3 = Pic_Treatments.get_image("../Data/basic/lissage_base_red/ovation_ovation/tumeur/"+str(k+1)+".png")

    new_img = Analogy_computing.do_lissage(im1,im2,im3,dimension="3D")
    Pic_Treatments.dim3(new_img)
    Pic_Openings.sauvegardeImage(nomFic=OVATION_TO_OVATION_PATH+str(k)+".png",image=new_img)
    return new_img


def to_trace(k):
    indice = [k + i for i in range(-5,6) if k+i >= min and k+i <=max]
    print(indice)
    expert_pivot = Pic_Treatments.get_image(EXPERT_PATH + str(k) + ".png")
    cnn_pivot = Pic_Treatments.get_image(CNN_PATH + str(k) + ".png")
    ovation_pivot = Pic_Treatments.get_image(CNN_OVATION_PATH + str(k) + ".png")
    for i in indice:

        img_liss_cnn_cnn = Pic_Treatments.get_image(CNN_PATH+str(i)+".png")
        img_liss_ovation_ovation = Pic_Treatments.get_image(CNN_OVATION_PATH+str(i)+".png")

        im_red = Analogy_computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=cnn_pivot,cnn2=img_liss_cnn_cnn)

        ct = Analogy_computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=cnn_pivot,cnn2=img_liss_cnn_cnn,methodType="lissage_base_red",im_red=im_red)
        ct2 = Analogy_computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=ovation_pivot,cnn2=img_liss_ovation_ovation,methodType="lissage_base_red",im_red=im_red)

        new_img = Pic_Treatments.dim3(ct)
        Pic_Openings.sauvegardeImage(nomFic= CNN_TO_CNN_PATH+str(i)+".png", image=new_img)

        new_img=Pic_Treatments.dim3(ct2)
        Pic_Openings.sauvegardeImage(nomFic=OVATION_TO_OVATION_PATH+str(i)+".png",image=new_img)


def to_calcul(i):
    img_2D = Pic_Treatments.get_image("../Data/basic/contouring/lissage_base_red/tumeur/"+str(i)+".png")
    img_3D = Pic_Treatments.get_image("../Data/basic/contouring/lissage_3D/tumeur/"+str(i)+".png")
    expert_i = Pic_Treatments.get_image(EXPERT_PATH + str( i) + ".png")
    cnn_i = Pic_Treatments.get_image(CNN_PATH + str( i) + ".png")
    ovation_i = Pic_Treatments.get_image(CNN_OVATION_PATH + str( i) + ".png")

    img_liss_cnn_cnn = Pic_Treatments.get_image(CNN_TO_CNN_PATH+str(i)+".png")
    img_liss_ovation_ovation = Pic_Treatments.get_image(OVATION_TO_OVATION_PATH + str( i) + ".png")
    img_3D = img_liss_cnn_cnn
    img2 = Pic_Treatments.get_image("../Data/basic/contouring/cnn_cnn/tumeur/"+str(i)+".png")
    a = Frame_Performances.indice_double(expert_i, img_3D)
    Dice = [[], [], [],[]]  # oracle,cnn,ovation,lissage simple
    Jacc = [[], [], [],[]]
    Dice[0].append(a[0])
    Jacc[0].append(a[1])
    b = Frame_Performances.indice_double(expert_i, cnn_i)
    Dice[1].append(b[0])
    Jacc[1].append(b[1])
    c = Frame_Performances.indice_double(expert_i, ovation_i)
    Dice[2].append(c[0])
    Jacc[2].append(c[1])
    d = Frame_Performances.indice_double(expert_i,img2)
    Dice[3].append(d[0])
    Jacc[3].append(d[1])

    return Dice, Jacc

if __name__ == '__main__':
    start = time.time()
    print(os.cpu_count())
    p = Pool(8)

    for rep in p.map(to_trace,pivots):
        1

    gold = [i for i in range(78, 107) if i not in pivots]
    print(gold)
    gold_bar = [i for i in range(2,136)]

    do_trio = True
    if do_trio:
        for rep in p.map(get_trio,gold_bar):
            1


    D = [[], [], [],[]]  # oracle,cnn,ovation,contour
    J = [[], [], [],[]]  # oracle, cnn,ovation,contour
    for rep in p.map(to_calcul,gold):
        D[0] += rep[0][0]
        D[1] += rep[0][1]
        D[2] += rep[0][2]
        D[3] += rep[0][3]
        J[0] += rep[1][0]
        J[1] += rep[1][1]
        J[2] += rep[1][2]
        J[3] += rep[1][3]
    x = gold
    print(len(x),len(D[0]),len(D[1]),len(D[2]),len(J[0]),len(J[1]),len(J[2]))
    plt.clf()
    plt.title("Comparaison Dice methode contouring")
    plt.plot(x,D[0],label="Contouring")
    plt.plot(x,D[1],label="CNN")
    plt.plot(x,D[2],label="Ovation")
    plt.plot(x,D[3],label="Lissage")

    for k in gold_bar:
        plt.axvline(x=k, color='red')
    plt.legend()
    plt.show()
    plt.clf()

    plt.title("Comparaison Jacc methode contouring")
    plt.plot(x, J[0], label="Contouring")
    plt.plot(x, J[1], label="CNN")
    plt.plot(x, J[2], label="Ovation")
    plt.plot(x, J[3], label="Lissage")

    for k in gold_bar:
        plt.axvline(x=k, color='red')
    plt.legend()
    plt.show()

    end = time.time()
    print("delay",end-start)
