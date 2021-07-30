import src.segmentationUtils.Analogy_Computing as Analogy_computing

import numpy as np

#Analogy_computing.red_calculation("classic", "c")

def taille_tumeur(matrice):
    l,c = matrice.shape
    nb = 0
    for i in range(l):
        for j in range(c):
            if (matrice[i][j] == 1):
                nb += 1
    return (nb/(l*c))