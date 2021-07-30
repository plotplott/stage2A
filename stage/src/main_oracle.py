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



"""
    But: realiser une courbe montrant l'eventuelle avancée de l'oracle
"""
param = "de"
pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136


sous_repertoire = "oracle"
CNN_OVATION_PATH = "../Data/coupeGap10/tumeur/"
CNN_PATH = "../Data/coupeCnn/tumeur/"
EXPERT_PATH = "../Data/coupeExpert/tumeur/"
CNN_TO_CNN_PATH = "../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/"
if not os.path.exists(CNN_TO_CNN_PATH):
    os.makedirs(CNN_TO_CNN_PATH)

def to_do(num):
    cnn_pivot = Pic_Openings.ouvrirImage(CNN_PATH+str(num)+".png")
    cnn_pivot = Pic_Treatments.niveauGris(cnn_pivot)
    cnn_pivot = Pic_Treatments.dim1(cnn_pivot)

    cnn_expert = Pic_Openings.ouvrirImage(EXPERT_PATH+str(num)+".png")
    cnn_expert = Pic_Treatments.niveauGris(cnn_expert)
    cnn_expert = Pic_Treatments.dim1(cnn_expert)

    return cnn_pivot,cnn_expert

def to_do2(k):
    kp1 = k+1

    cnn_pivot1 = Pic_Openings.ouvrirImage(CNN_PATH + str(k) + ".png")
    cnn_pivot1 = Pic_Treatments.niveauGris(cnn_pivot1)
    cnn_pivot1 = Pic_Treatments.dim1(cnn_pivot1)
    cnn_expert1 = Pic_Openings.ouvrirImage(EXPERT_PATH + str(k) + ".png")
    cnn_expert1 = Pic_Treatments.niveauGris(cnn_expert1)
    cnn_expert1 = Pic_Treatments.dim1(cnn_expert1)
    cnn_pivot2 = Pic_Openings.ouvrirImage(CNN_PATH + str(kp1) + ".png")
    cnn_pivot2 = Pic_Treatments.niveauGris(cnn_pivot2)
    cnn_pivot2 = Pic_Treatments.dim1(cnn_pivot2)
    cnn_expert2 = Pic_Openings.ouvrirImage(EXPERT_PATH + str(kp1) + ".png")
    cnn_expert2 = Pic_Treatments.niveauGris(cnn_expert2)
    cnn_expert2 = Pic_Treatments.dim1(cnn_expert2)

    for i in range(1,11):
        current_cnn = Pic_Openings.ouvrirImage(CNN_PATH + str(k)+".png")
        current_cnn = Pic_Treatments.niveauGris(current_cnn)
        current_cnn = Pic_Treatments.dim1(current_cnn)
        current_expert = Pic_Openings.ouvrirImage(EXPERT_PATH + str(k)+".png")
        current_expert = Pic_Treatments.niveauGris(current_expert)
        current_expert = Pic_Treatments.dim1(current_expert)
        image = Analogy_computing.do_oracle(pivot1_a = cnn_pivot1,pivot1_b = cnn_expert1 ,pivot2_a = cnn_pivot2,pivot2_b = cnn_expert2,cible_a=current_cnn,cible_expert=current_expert)
        image = Pic_Treatments.dim3(image)
        Pic_Openings.sauvegardeImage(nomFic=CNN_TO_CNN_PATH+str(k+i)+".png",image=image)

    return 1

def to_trace(num_pivot):
    '''
    But: entre pivot k et pivot k+1, on calcule dice entre oracle/expert et cnn/expert et ovation/expert
    :param pivot:
    :return: 2 listes [dice(o,e),dice(cnn,e),dice(ova/e)] et [jac(o,e),..]
    '''
    Dice = [[],[],[],[]] #oracle , cnn, ovation , methode lissage base red
    Jacc = [[],[],[],[]]
    for i in range(1,11):

        expert_i = Pic_Treatments.get_image(EXPERT_PATH + str(num_pivot + i)+".png")
        cnn_i = Pic_Treatments.get_image(CNN_PATH+str(num_pivot+i)+".png")
        ovation_i = Pic_Treatments.get_image(CNN_OVATION_PATH+str(num_pivot+i)+".png")
        oracle_i = Pic_Treatments.get_image(CNN_TO_CNN_PATH+str(num_pivot+i)+".png")

        method_lissage_i = Pic_Treatments.get_image("../Data/basic/lissage_base_red/cnn_cnn/tumeur/"+str(num_pivot+i)+".png")

        """
        Dice[0].append(Frame_Performances.indice_dice(expert_i,oracle_i))
        Jacc[0].append(Frame_Performances.indice_jaccard(expert_i,oracle_i))

        Dice[1].append(Frame_Performances.indice_dice(expert_i,cnn_i))
        Jacc[1].append(Frame_Performances.indice_jaccard(expert_i,cnn_i))

        Dice[2].append(Frame_Performances.indice_dice(expert_i,ovation_i))
        Jacc[2].append(Frame_Performances.indice_jaccard(expert_i,ovation_i))

        """
        a = Frame_Performances.indice_double(expert_i,oracle_i)
        Dice[0].append(a[0])
        Jacc[0].append(a[1])
        b = Frame_Performances.indice_double(expert_i,cnn_i)
        Dice[1].append(b[0])
        Jacc[1].append(b[1])
        c=Frame_Performances.indice_double(expert_i,ovation_i)
        Dice[2].append(c[0])
        Jacc[2].append(c[1])
        d = Frame_Performances.indice_double(expert_i,method_lissage_i)
        Dice[3].append(d[0])
        Jacc[3].append(d[1])

    return Dice,Jacc


if __name__ == '__main__':
    start = time.time()
    print(os.cpu_count())
    p = Pool(8)

    # on cherche a obtenir un ensemble de vecteur que l'on pourra tracer: abscisse et ordonnees
    D=[[],[],[],[]] # oracle,cnn,ovation
    J=[[],[],[],[]] #oracle, cnn,ovation

    #on commence par l'ordonne en passant par to_trace en parallelisation
    #L = [pivots[k] for k in range(len(pivots)) if k+1 < len(pivots)]
    L = [24, 35, 46, 57, 68, 79, 90, 101]

    for rep in p.map(to_trace,L):
        D[0] += rep[0][0]
        D[1] += rep[0][1]
        D[2] += rep[0][2]
        D[3] += rep[0][3]
        J[0] += rep[1][0]
        J[1] += rep[1][1]
        J[2] += rep[1][2]
        J[3] += rep[1][2]

    #maintenant il nous faut nos absicess
    x = [ i for i in range(min,max+1) if i not in pivots and i> L[0] and i < L[-1]+11]
    print(len(x),len(D[0]),len(D[1]),len(D[2]),len(J[0]),len(J[1]),len(J[2]))
    plt.clf()
    plt.title("Comparaison Dice methode oracle")
    plt.plot(x,D[0],label="Oracle")
    plt.plot(x,D[1],label="CNN")
    plt.plot(x,D[2],label="Ovation")
    plt.plot(x,D[3],label="Lissage_cnn")
    plt.legend()
    plt.show()
    plt.clf()
    plt.title("Comparaison Jacc methode oracle")
    plt.plot(x, J[0], label="Oracle")
    plt.plot(x, J[1], label="CNN")
    plt.plot(x, J[2], label="Ovation")
    plt.plot(x,J[3],label="Lissage_cnn")
    plt.legend()
    plt.show()


    plt.clf()



    # Import libraries
    # Creating dataset
    data = [D[0],D[1],D[2],D[3]]
    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(data)

    plt.legend()
    # show plot
    plt.show()

    end = time.time()
    print("delay",end-start)