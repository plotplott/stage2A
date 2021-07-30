from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import os

import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import time

pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
DATA_PATH = "../Data/"
tumeur = "/rein/"
path_to_store = "../Data/lissage_after/rein/"
min,max = 1,136

def courbePoids():
    liste = [i/10 for i in range(11)]
    n = len(liste)

    centre = []
    bord = []
    diagonale = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (liste[i]+liste[j]+liste[k] == 1):
                    centre.append(liste[i])
                    bord.append(liste[j])
                    diagonale.append(liste[k])

    for i in range(len(centre)):
        # effectuer tous les calculs d'images
        for k in pivots:
            second_try([centre[i], bord[i], diagonale[i]],pivot=k)
'''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(centre, bord, diagonale, 'gray')
    plt.show()
'''
def second_try(poids, pivot = 90):
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
            imageCurrentCnn = Pic_Treatments.niveauGris(imageCurrentCnn)
            matCurrentCnn = Pic_Treatments.dim1(imageCurrentCnn)


            imageCurrentRed = Pic_Openings.ouvrirImage(DATA_PATH + "red/" + str(pivot + k) + "from" + str(pivot) + ".png")
            matCurrentRed = Pic_Treatments.dim1(imageCurrentRed)

            matLissage = cnnTab_to_anaTab_(matPivotExpert, matPivotCnn, matCurrentCnn, poids, im_red=matCurrentRed)
            imageLissage = Pic_Treatments.dim3(matLissage)
            repertoire = DATA_PATH + "lissage_after/rein/" + str(poids[0]) + "-" + str(poids[1]) + "-" + str(poids[2]) + "/"
            if not os.path.exists(repertoire):
                os.makedirs(repertoire)
            Pic_Openings.sauvegardeImage(repertoire + str(pivot + k) + "from" + str(pivot) + ".png", imageLissage)


    end = time.time()
    print("action took:",end-start,"sec")

    return 1

def cnnTab_to_anaTab_(expert, cnn1, cnn2, poids, im_red=None):

    size = expert.shape
    row,col = size[0],size[1]

    my_segmentation = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            a = cnn1[i][j]
            b = expert[i][j]
            c= cnn2[i][j]
            if not Analogy_Computing.isUnknown(a, b, c):
                my_segmentation[i][j] = Analogy_Computing.compute_d_known(a,b,c)
            else:
                # ici on veut utiliser la fonction compute lissage_base
                # pour ce faire, il nous faut la matrice 3x3 associe
                value = 0.5
                if 0 < i < row-1:
                    if 0< j < col-1:
                        # ici il nous faut reconstruire la matrice 3x3 qu'on va utiliser
                        # on fait appel a une fonction externe pour ca
                        mat3x3 = Analogy_Computing.get_mat3x3(im_red,i,j)
                        value = Analogy_Computing.compute_d_lissage(mat3x3, poids)
                my_segmentation[i][j] = value

    return my_segmentation

#courbePoids()

for k in pivots:
    second_try([0.5, 0.3, 0.2], pivot=k)
