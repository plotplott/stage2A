import numpy as np
from multiprocessing import Pool
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import time

CNN_OVATION_PATH = "../Data/coupeGap10/tumeur/"
CNN_PATH = "../Data/coupeCnn/tumeur/"
EXPERT_PATH = "../Data/coupeExpert/tumeur/"

CNN_TO_CNN_PATH = "../Data/basic/red/cnn_cnn/tumeur/"
CNN_TO_OVATION_PATH = "../Data/basic/red/cnn_ovation/tumeur/"
OVATION_TO_CNN_PATH = "../Data/basic/red/ovation_cnn/tumeur/"
OVATION_TO_OVATION_PATH = "../Data/basic/red/ovation_ovation/tumeur/"

pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]


def do_union(mats):
    param = "less"
    matrice1 = mats[0]
    matrice2 = mats[1]
    matrice1 = Pic_Treatments.niveauGris(matrice1)
    matrice1 = Pic_Treatments.dim1(matrice1)

    matrice2 = Pic_Treatments.niveauGris(matrice2)
    matrice2 = Pic_Treatments.dim1(matrice2)

    size = matrice1.shape
    new_mat = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if matrice1[i][j] == 1 or matrice2[i][j] == 1:
                new_mat[i][j] = 1
            elif matrice1[i][j] == 0.5 or matrice2[i][j] == 0.5:  # on a un truc du genre 0.5 / 0
                if param == "less":
                    pass
                else:  # param = more
                    new_mat[i][j] = 1

    return new_mat,3

def do_intersect(int):
    L = []
    for pivot in pivots:
        # entrainment a, b
        b1 = Pic_Openings.ouvrirImage(nomFic=EXPERT_PATH + str(pivot) + ".png")
        a1 = Pic_Openings.ouvrirImage(nomFic=CNN_PATH + str(pivot) + ".png")
        L.append([a1,b1])
    return L

start = time.time()
L = do_intersect(1)
print("delay",time.time()-start)


if __name__ == '__main__':
    start = time.time()
    p = Pool(5)
    for k in p.map(do_union, L):
        print(k[1])
    print("delay", time.time() - start)