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
import src.segmentationUtils.main_functions as main_functions

if __name__ == '__main__':
    os.system('python segmentationUtils/main_functions.py')

pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136

sous_repertoire = "lissage_base_red"
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


projet = {"CNN_PATH":"../Data/coupeCnn/tumeur/","EXPERT_PATH":"../Data/coupeExpert/tumeur/","OVATION_PATH":"../Data/coupeGap10/tumeur/"}

fermeture = {"verticale":"../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/verticale/pure/"}
fermeture["horizontale"]= "../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/horizontale/pure/"
fermeture["verti_hori"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/verti_hori/pure/"
fermeture["diago_mont"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diago_mont/pure/"
fermeture["diago_desc"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diago_desc/pure/"
fermeture["diagos"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diagos/pure/"
fermeture["base"] = "../Data/basic/"+sous_repertoire+"/cnn_cnn/tumeur/"
fermeture["rectangulaire"]="../Data/basic/"+sous_repertoire+"/cnn_cnn/tumeur/rectangulaire/pure/"

croise = {"verticale":"../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/verticale/croise/"}
croise["horizontale"]= "../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/horizontale/croise/"
croise["verti_hori"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/verti_hori/croise/"
croise["diago_mont"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diago_mont/croise/"
croise["diago_desc"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diago_desc/croise/"
croise["diagos"]="../Data/basic/" + sous_repertoire + "/cnn_cnn/tumeur/diagos/croise/"
croise["base"] = "../Data/basic/"+sous_repertoire+"/cnn_cnn/tumeur/"
croise["rectangulaire"]="../Data/basic/"+sous_repertoire+"/cnn_cnn/tumeur/rectangulaire/croise/"

for k in fermeture:
    #print(fermeture[k])
    Pic_Openings.if_not_exist(fermeture[k])
    Pic_Openings.if_not_exist(croise[k])

def to_trace(k):
    indice = [k + i for i in range(-5, 6) if k + i >= min and k + i <= max]
    for k in indice:
        im = Pic_Openings.ouvrirImage(fermeture["base"]+str(k)+".png")
        im = Pic_Treatments.dim1(im)

        vert = Analogy_computing.do_fermeture_convexe(mat = im,param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert),nomFic=fermeture["verticale"]+str(k)+".png")
        vert_croise = Frame_Performances.do_intersect(vert,im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert_croise),nomFic=croise["verticale"]+str(k)+".png")



        hor = Analogy_computing.do_fermeture_convexe(mat = im,param="horizontale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(hor),nomFic=fermeture["horizontale"]+str(k)+".png")
        hori_croise = Frame_Performances.do_intersect(hor, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(hori_croise),nomFic=croise["horizontale"] + str(k) + ".png")

        vert_hor = Analogy_computing.do_fermeture_convexe(mat = vert,param="horizontale")
        vert_hor = Analogy_computing.do_fermeture_convexe(mat = vert_hor,param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert_hor),nomFic=fermeture["verti_hori"]+str(k)+".png")
        vert_hori_croise = Frame_Performances.do_intersect(vert_hor, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert_hori_croise),nomFic=croise["verti_hori"] + str(k) + ".png")

        diago_mont = Analogy_computing.do_fermeture_convexe(mat = im,param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont),nomFic=fermeture["diago_mont"]+str(k)+".png")
        diagoM_croise = Frame_Performances.do_intersect(diago_mont, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diagoM_croise),nomFic=croise["diago_mont"] + str(k) + ".png")

        diago_desc = Analogy_computing.do_fermeture_convexe(mat = im,param="diago_descendante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_desc),nomFic=fermeture["diago_desc"]+str(k)+".png")
        diagoD_croise = Frame_Performances.do_intersect(diago_desc, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diagoD_croise),nomFic=croise["diago_desc"] + str(k) + ".png")

        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat = diago_mont,param="diago_descendante")
        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat = diago_mont_desc,param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont_desc),nomFic=fermeture["diagos"]+str(k)+".png")
        diagoMD_croise = Frame_Performances.do_intersect(diago_mont_desc, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diagoMD_croise),nomFic=croise["diagos"] + str(k) + ".png")

        rect = Analogy_computing.do_fermeture_convexe(mat = im,param="rectangulaire")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(rect),nomFic=fermeture["rectangulaire"]+str(k)+".png")
        rect_croise = Frame_Performances.do_intersect(rect, im)
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(rect_croise), nomFic=croise["rectangulaire"] + str(k) + ".png")

    return 1




if __name__ == '__main__':
    start = time.time()
    print(os.cpu_count())
    p = Pool(os.cpu_count()-2)
    if tracing:
        for rep in p.map(to_trace,pivots):
            1

    print("delay time:",time.time()-start)