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


#but: faire 2 methode : cnn_ovation: expert :: cnn_ovation : x
#                       cnn : expert :: cnn_ovation
#                       a chaque fois on calcule dice/jaccard
# L'idee est d'obtenir un jolie DICE/JACCARD pour les differentes propositions faites ici

ss_rep = "SK"

CNN_OVATION_PATH = "../Data/coupeGap10/tumeur/"
CNN_PATH = "../Data/coupeCnn/tumeur/"
EXPERT_PATH = "../Data/coupeExpert/tumeur/"

CNN_TO_CNN_PATH = "../Data/basic/"+ss_rep+"/cnn_cnn/tumeur/"
CNN_TO_OVATION_PATH = "../Data/basic/"+ss_rep+"/cnn_ovation/tumeur/"
OVATION_TO_CNN_PATH = "../Data/basic/"+ss_rep+"/ovation_cnn/tumeur/"
OVATION_TO_OVATION_PATH = "../Data/basic/"+ss_rep+"/ovation_ovation/tumeur/"


#pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
pivots = [35, 46, 57, 68, 79, 90]
#pivots = [79,90]

min, max = 1, 136
start = time.time()


def do_pivot(pivot):
    print(pivot)
    tot = []
    x = []
    dd1 = []
    dd2 = []
    dd3 = []
    dd4 = []
    yy1 = []
    yy2 = []
    yy3 = []
    yy4 = []

    ovat_expert_d=[]
    ovat_expert_j=[]

    for k in range(-5, 6, 1):
        if pivot + k >= min and pivot + k <= max:
            x.append(pivot + k)
            expert = Pic_Openings.ouvrirImage(nomFic=EXPERT_PATH + str(pivot + k) + ".png")
            expert = Pic_Treatments.niveauGris(expert)
            expert = Pic_Treatments.dim1(expert)

            real_ovation = Pic_Openings.ouvrirImage(nomFic=CNN_OVATION_PATH + str(pivot + k) + ".png")
            real_ovation = Pic_Treatments.niveauGris(real_ovation)
            real_ovation = Pic_Treatments.dim1(real_ovation)

            real_cnn = Pic_Openings.ouvrirImage(nomFic=CNN_PATH + str(pivot + k) + ".png")
            real_cnn = Pic_Treatments.niveauGris(real_cnn)
            real_cnn = Pic_Treatments.dim1(real_cnn)

            cnnCnn = Pic_Openings.ouvrirImage(nomFic=CNN_TO_CNN_PATH + str(pivot + k) + "from" + str(pivot) + ".png")
            cnnCnn = Pic_Treatments.niveauGris(cnnCnn)
            cnnCnn = Pic_Treatments.dim1(cnnCnn)

            cnnOvation = Pic_Openings.ouvrirImage(nomFic=CNN_TO_OVATION_PATH + str(pivot + k) + "from" + str(pivot) + ".png")
            cnnOvation = Pic_Treatments.niveauGris(cnnOvation)
            cnnOvation = Pic_Treatments.dim1(cnnOvation)

            ovationCnn = Pic_Openings.ouvrirImage(nomFic=OVATION_TO_CNN_PATH + str(pivot + k) + "from" + str(pivot) + ".png")
            ovationCnn = Pic_Treatments.niveauGris(ovationCnn)
            ovationCnn = Pic_Treatments.dim1(ovationCnn)

            ovationOvation = Pic_Openings.ouvrirImage(nomFic=OVATION_TO_OVATION_PATH + str(pivot + k) + "from" + str(pivot) + ".png")
            ovationOvation = Pic_Treatments.niveauGris(ovationOvation)
            ovationOvation = Pic_Treatments.dim1(ovationOvation)

            d1 = Frame_Performances.indice_dice(expert, cnnCnn)
            j1 = Frame_Performances.indice_jaccard(expert, cnnCnn)

            d2 = Frame_Performances.indice_dice(expert, cnnOvation)
            j2 = Frame_Performances.indice_jaccard(expert, cnnOvation)

            d3 = Frame_Performances.indice_dice(expert, ovationCnn)
            j3 = Frame_Performances.indice_jaccard(expert, ovationCnn)

            d4 = Frame_Performances.indice_dice(expert, ovationOvation)
            j4 = Frame_Performances.indice_jaccard(expert, ovationOvation)

            oed = Frame_Performances.indice_dice(expert,real_ovation)
            oej = Frame_Performances.indice_jaccard(expert,real_ovation)

            dd1.append(d1)
            dd2.append(d2)
            dd3.append(d3)
            dd4.append(d4)

            yy1.append(j1)
            yy2.append(j2)
            yy3.append(j3)
            yy4.append(j4)

            ovat_expert_d.append(oed)
            ovat_expert_j.append(oej)

    return x,[dd1,dd2,dd3,dd4],[yy1,yy2,yy3,yy4],[ovat_expert_d,ovat_expert_j]


if __name__ == '__main__':
    start = time.time()
    print(os.cpu_count())
    p = Pool(8)
    x = []
    dice = [[],[],[],[],[]]
    jacc = [[],[],[],[],[]]
    for rep in p.map(do_pivot,pivots):
        print(rep)
        x += rep[0]
        dice[0]+=rep[1][0]
        dice[1]+=rep[1][1]
        dice[2]+=rep[1][2]
        dice[3]+=rep[1][3]
        dice[4]+=rep[3][0] # ligne pour expert / ovation
        jacc[0]+=rep[2][0]
        jacc[1]+=rep[2][1]
        jacc[2]+=rep[2][2]
        jacc[3]+=rep[2][3]
        jacc[4]+=rep[3][1]#ligne pour expert/ovation


    dice = [ [dice[0][i],dice[1][i],dice[2][i],dice[3][i],dice[4][i]] for i in range(len(dice[0])) ]
    jacc = [ [jacc[0][i],jacc[1][i],jacc[2][i],jacc[3][i],jacc[4][i]] for i in range(len(jacc[0])) ]

    print("delay", time.time() - start)
    plt.clf()
    plt.ylabel("Valeur du Dice")
    plt.xlabel("Numero de Segmentation")
    ss_rep = "SK"
    plt.title("Indice methode {} en fonction du numero de la segmentation".format(ss_rep))
    for k in pivots:
        plt.axvline(x=k)
    lineObjects = plt.plot(x, dice)
    plt.legend(iter(lineObjects), ("a=Cnn|b=Cnn", "a=Cnn|b=Ovat", "a=Ovat|b=Cnn","a=Ovat|b=Ovat","ovationPure"))
    plt.show()


    plt.clf()
    plt.ylabel("Valeur du Jacc")
    plt.xlabel("Numero de Segmentation")
    plt.title("Indice methode {} en fonction du numero de la segmentation".format(ss_rep))
    for k in pivots:
        plt.axvline(x=k)
    lineObjects = plt.plot(x,jacc)
    plt.legend(iter(lineObjects), ("a=Cnn|b=Cnn", "a=Cnn|b=Ovat", "a=Ovat|b=Cnn", "a=Ovat|b=Ovat","ovationPure"))
    plt.show()

    print("delay", time.time() - start)

