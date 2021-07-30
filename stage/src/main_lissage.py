import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Segmentation as Segmentation
import src.segmentationUtils.Segmentation_Calculator as Segmentation_Calculator
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import time
from multiprocessing import Pool

param = "de"
pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
DATA_PATH = "../Data/"
tumeur = "/tumeur/"
path_to_store = "../Data/lissage_base/"
min,max = 1,136

def first_try(pivot = 90):

    start = time.time()

    imagePivotExpert = Pic_Openings.ouvrirImage(DATA_PATH + "coupeExpert" + tumeur + str(pivot) + ".png")
    matPivotExpert = Pic_Treatments.niveauGris(imagePivotExpert)
    matPivotExpert = Pic_Treatments.dim1(matPivotExpert)

    imagePivotCnn = Pic_Openings.ouvrirImage(DATA_PATH + "coupeCnn" + tumeur + str(pivot) + ".png")
    matPivotCnn = Pic_Treatments.niveauGris(imagePivotCnn)
    matPivotCnn = Pic_Treatments.dim1(matPivotCnn)


    for k in range(-5,6,1):

        if k != 0 and pivot+k > min and pivot + k < max:
            #but calculer les images autour du pivot et les enregistrer dans red pour voir
            imageCurrentCnn = Pic_Openings.ouvrirImage(DATA_PATH + "coupeCnn" + tumeur + str(pivot + k) + ".png")
            matCurrentCnn = Pic_Treatments.niveauGris(imageCurrentCnn)
            matCurrentCnn = Pic_Treatments.dim1(imageCurrentCnn)

            matLissage = Analogy_Computing.cnnTab_to_anaTab_(matPivotExpert, matPivotCnn, matCurrentCnn, methodType="lissage_base")
            imageLissage = Pic_Treatments.dim3(matLissage)
            Pic_Openings.sauvegardeImage(path_to_store + str(pivot + k) + "from" + str(pivot) + ".png", imageLissage)

    end = time.time()
    print("action took:",end-start,"sec")

    return 1
'''
for k in pivots:
    first_try(pivot=k)
'''

def second_try(pivot = 90):
    start = time.time()

    imagePivotExpert = Pic_Openings.ouvrirImage(DATA_PATH + "coupeExpert" + tumeur + str(pivot) + ".png")
    matPivotExpert = Pic_Treatments.niveauGris(imagePivotExpert)
    matPivotExpert = Pic_Treatments.dim1(matPivotExpert)

    imagePivotCnn = Pic_Openings.ouvrirImage(DATA_PATH + "coupeCnn" + tumeur + str(pivot) + ".png")
    matPivotCnn = Pic_Treatments.niveauGris(imagePivotCnn)
    matPivotCnn = Pic_Treatments.dim1(matPivotCnn)

    for k in range(-5, 6, 1):

        if k != 0 and pivot + k > min and pivot + k < max:
            # but calculer les images autour du pivot et les enregistrer dans red pour voir
            imageCurrentCnn = Pic_Openings.ouvrirImage(DATA_PATH + "coupeCnn" + tumeur + str(pivot + k) + ".png")
            matCurrentCnn = Pic_Treatments.niveauGris(imageCurrentCnn)
            matCurrentCnn = Pic_Treatments.dim1(imageCurrentCnn)


            imageCurrentRed = Pic_Openings.ouvrirImage(DATA_PATH + "red/" + str(pivot + k) + "from" + str(pivot) + ".png")
            matCurrentRed = Pic_Treatments.dim1(imageCurrentRed)

            matLissage = Analogy_Computing.cnnTab_to_anaTab_(matPivotExpert, matPivotCnn, matCurrentCnn, methodType="lissage_base", im_red=matCurrentRed)
            imageLissage = Pic_Treatments.dim3(matLissage)
            Pic_Openings.sauvegardeImage(DATA_PATH + "lissage_after/04-03-03/" + str(pivot + k) + "from" + str(pivot) + ".png", imageLissage)


    end = time.time()
    print("action took:",end-start,"sec")
'''
x=[i for i in range(10)]
x2 = [ [i,i+2,i+1] for i in range(10)]
print(x2)


plt.clf()
lineObjects = plt.plot(x, x2)
plt.legend(iter(lineObjects), ('foo', 'bar', 'baz'))

plt.show()
'''

sous_repertoire = "lissage_base_red"

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

def to_trace(k):
    indice = [k + i for i in range(-5,6) if k+i >= min and k+i <=max]
    print(indice)
    expert_pivot = Pic_Treatments.get_image(EXPERT_PATH + str(k) + ".png")
    cnn_pivot = Pic_Treatments.get_image(CNN_PATH + str(k) + ".png")
    ovation_pivot = Pic_Treatments.get_image(CNN_OVATION_PATH + str(k) + ".png")
    for i in indice:

        img_liss_cnn_cnn = Pic_Treatments.get_image(CNN_PATH+str(i)+".png")
        img_liss_ovation_ovation = Pic_Treatments.get_image(CNN_OVATION_PATH+str(i)+".png")

        im_red = Analogy_Computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=cnn_pivot,cnn2=img_liss_cnn_cnn)

        ct = Analogy_Computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=cnn_pivot,cnn2=img_liss_cnn_cnn,methodType="lissage_base_red",im_red=im_red)
        ct2 = Analogy_Computing.cnnTab_to_anaTab_(expert=expert_pivot,cnn1=ovation_pivot,cnn2=img_liss_ovation_ovation,methodType="lissage_base_red",im_red=im_red)
        new_img = Pic_Treatments.dim3(ct)

        Pic_Openings.sauvegardeImage(nomFic= CNN_TO_CNN_PATH+str(i)+".png", image=new_img)
        new_img=Pic_Treatments.dim3(ct2)
        Pic_Openings.sauvegardeImage(nomFic=OVATION_TO_OVATION_PATH+str(i)+".png",image=new_img)


def to_calcul(i):
    img = Pic_Treatments.get_image("../Data/basic/contouring/cnn_cnn/tumeur/"+str(i)+".png")
    expert_i = Pic_Treatments.get_image(EXPERT_PATH + str( i) + ".png")
    cnn_i = Pic_Treatments.get_image(CNN_PATH + str( i) + ".png")
    ovation_i = Pic_Treatments.get_image(CNN_OVATION_PATH + str( i) + ".png")

    img_liss_cnn_cnn = Pic_Treatments.get_image(CNN_TO_CNN_PATH+str(i)+".png")
    img_liss_ovation_ovation = Pic_Treatments.get_image(OVATION_TO_OVATION_PATH + str( i) + ".png")
    img = img_liss_cnn_cnn
    img2 = Pic_Treatments.get_image("../Data/basic/contouring/cnn_cnn/tumeur/"+str(i)+".png")
    a = Frame_Performances.indice_double(expert_i, img)
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

    #for rep in p.map(to_trace,pivots):

    gold = [i for i in range(78, 107) if i not in pivots]
    print(gold)
    gold_bar = [79, 90, 101]

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
