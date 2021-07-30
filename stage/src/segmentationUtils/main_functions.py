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
import sqlite3
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
from multiprocessing import Pool
DB_PATH = "../../Data/mySql.db"
pivots = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
min, max = 1, 136




#######################################################################################################################
#   Realisation de fermeture convexes
#   but: faire les fermetures convexes sur cnn/expert horizontale/verticale/les deux diagonales montante/descendante/les deux
#   on entrepose le resultar dans les dossiers avec de nouveaux dossiers crees
#######################################################################################################################
param = "lissage_base_red"


# le repertoire ou on stocke les differentes fermetures convexes liees au cnn
fermeture = {"verticale":"../../Data/coupeCnn/tumeur/verticale/"}
fermeture["horizontale"]= "../../Data/coupeCnn/tumeur/horizontale/"
fermeture["verti_hori"]="../../Data/coupeCnn/tumeur/verti_hori/"
fermeture["diago_mont"]="../../Data/coupeCnn/tumeur/diago_montante/"
fermeture["diago_desc"]="../../Data/coupeCnn/tumeur/diago_descendante/"
fermeture["diagos"]="../../Data/coupeCnn/tumeur/diagos/"
fermeture["base"] = "../../Data/coupeCnn/tumeur/"
fermeture["rectangulaire"]="../../Data/coupeCnn/tumeur/rectangle/"

# le repertoire ou on stocke les differentes fermetures convexes liees a l'expert
fermeture2 = {"verticale":"../../Data/coupeExpert/tumeur/verticale/"}
fermeture2["horizontale"]= "../../Data/coupeExpert/tumeur/horizontale/"
fermeture2["verti_hori"]="../../Data/coupeExpert/tumeur/verti_hori/"
fermeture2["diago_mont"]="../../Data/coupeExpert/tumeur/diago_montante/"
fermeture2["diago_desc"]="../../Data/coupeExpert/tumeur/diago_descendante/"
fermeture2["diagos"]="../../Data/coupeExpert/tumeur/diagos/"
fermeture2["base"] = "../../Data/coupeExpert/tumeur/"
fermeture2["rectangulaire"]="../../Data/coupeExpert/tumeur/rectangle/"

#le repertoire ou on stocke le produit de fc(a):fc(b)::fc(c):x
fermeture3 = {"verticale":"../../Data/basic/"+param+"/cnn_cnn/tumeur/verticale/pure/"}
fermeture3["horizontale"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/horizontale/pure/"
fermeture3["verti_hori"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/verti_hori/pure/"
fermeture3["rectangulaire"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/rectangulaire/pure/"
fermeture3["diago_mont"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diago_montante/pure/"
fermeture3["diago_desc"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diago_descendante/pure/"
fermeture3["diagos"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diagos/pure/"
fermeture3["base"] = "../../Data/basic/"+param+"/cnn_cnn/tumeur/"

#le repertoire ou on stocke le produit de l'intersection fcd et d
fermeture4 = {"verticale":"../../Data/basic/"+param+"/cnn_cnn/tumeur/verticale/croise/"}
fermeture4["horizontale"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/horizontale/croise/"
fermeture4["verti_hori"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/verti_hori/croise/"
fermeture4["rectangulaire"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/rectangulaire/croise/"
fermeture4["diago_mont"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diago_montante/croise/"
fermeture4["diago_desc"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diago_descendante/croise/"
fermeture4["diagos"]="../../Data/basic/"+param+"/cnn_cnn/tumeur/diagos/croise/"

def support_fermeture_cnn_expert(pivot):
    '''
    but: creer les fermetures convexes autour du pivot pour les images cnn/expert
    '''
    indice = [pivot + i for i in range(-5, 6) if pivot + i >= min and pivot + i <= max]

    for k in indice:
        #fermeture du cnn
        cnn = Pic_Openings.ouvrirImage("../../Data/coupeCnn/tumeur/"+str(k)+".png")
        cnn = Pic_Treatments.niveauGris(cnn)
        cnn = Pic_Treatments.dim1(cnn)

        vert = Analogy_computing.do_fermeture_convexe(mat=cnn, param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert), nomFic=fermeture["verticale"] + str(k) + ".png")
        hor = Analogy_computing.do_fermeture_convexe(mat=cnn, param="horizontale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(hor), nomFic=fermeture["horizontale"] + str(k) + ".png")
        vert_hor = Analogy_computing.do_fermeture_convexe(mat=vert, param="horizontale")
        vert_hor = Analogy_computing.do_fermeture_convexe(mat=vert_hor, param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert_hor),nomFic=fermeture["verti_hori"] + str(k) + ".png")
        diago_mont = Analogy_computing.do_fermeture_convexe(mat=cnn, param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont),nomFic=fermeture["diago_mont"] + str(k) + ".png")
        diago_desc = Analogy_computing.do_fermeture_convexe(mat=cnn, param="diago_descendante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_desc),nomFic=fermeture["diago_desc"] + str(k) + ".png")
        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat=diago_mont, param="diago_descendante")
        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat=diago_mont_desc, param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont_desc),nomFic=fermeture["diagos"] + str(k) + ".png")
        rect = Analogy_computing.do_fermeture_convexe(mat=cnn, param="rectangulaire")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(rect),nomFic=fermeture["rectangulaire"] + str(k) + ".png")

        #fermeture de l'expert
        expert = Pic_Openings.ouvrirImage("../../Data/coupeExpert/tumeur/" + str(k) + ".png")
        expert  = Pic_Treatments.niveauGris(expert)
        expert  = Pic_Treatments.dim1(expert)

        vert = Analogy_computing.do_fermeture_convexe(mat=expert, param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert), nomFic=fermeture2["verticale"] + str(k) + ".png")
        hor = Analogy_computing.do_fermeture_convexe(mat=expert, param="horizontale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(hor), nomFic=fermeture2["horizontale"] + str(k) + ".png")
        vert_hor = Analogy_computing.do_fermeture_convexe(mat=vert, param="horizontale")
        vert_hor = Analogy_computing.do_fermeture_convexe(mat=vert_hor, param="verticale")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(vert_hor),nomFic=fermeture2["verti_hori"] + str(k) + ".png")
        diago_mont = Analogy_computing.do_fermeture_convexe(mat=expert, param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont),nomFic=fermeture2["diago_mont"] + str(k) + ".png")
        diago_desc = Analogy_computing.do_fermeture_convexe(mat=expert, param="diago_descendante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_desc),nomFic=fermeture2["diago_desc"] + str(k) + ".png")
        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat=diago_mont, param="diago_descendante")
        diago_mont_desc = Analogy_computing.do_fermeture_convexe(mat=diago_mont_desc, param="diago_montante")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(diago_mont_desc),nomFic=fermeture2["diagos"] + str(k) + ".png")
        rect = Analogy_computing.do_fermeture_convexe(mat=expert, param="rectangulaire")
        Pic_Openings.sauvegardeImage(image=Pic_Treatments.dim3(rect), nomFic=fermeture2["rectangulaire"] + str(k) + ".png")


def raisonnement_sur_fermeture_segment(pivot):
    '''
    but: realiser le raisonnement par analogie selon fc(a):fc(b)::fc(c): fcD
    on veut recuperer fcD obtenu par analogie

    on le stock dans le dossier basic/cnn_cnn/tumeur/rectangle ou verticale ect
    :param pivot:
    :return:
    '''
    a_verti = Pic_Openings.ouvrirImage(fermeture["verticale"]+str(pivot)+".png")
    a_verti = Pic_Treatments.dim1(a_verti)
    a_hori = Pic_Openings.ouvrirImage(fermeture["horizontale"]+str(pivot)+".png")
    a_hori = Pic_Treatments.dim1(a_hori)
    a_hori_verti = Pic_Openings.ouvrirImage(fermeture["verti_hori"] + str(pivot) + ".png")
    a_hori_verti = Pic_Treatments.dim1(a_hori_verti)

    b_verti = Pic_Openings.ouvrirImage(fermeture2["verticale"] + str(pivot) + ".png")
    b_verti = Pic_Treatments.dim1(b_verti)
    b_hori = Pic_Openings.ouvrirImage(fermeture2["horizontale"] + str(pivot) + ".png")
    b_hori = Pic_Treatments.dim1(b_hori)
    b_hori_verti = Pic_Openings.ouvrirImage(fermeture2["verti_hori"] + str(pivot) + ".png")
    b_hori_verti = Pic_Treatments.dim1(b_hori_verti)

    indice = [pivot + i for i in range(-5, 6) if pivot + i >= min and pivot + i <= max]
    for k in indice:
        c_verti = Pic_Openings.ouvrirImage(fermeture["verticale"] + str(k) + ".png")
        c_verti = Pic_Treatments.dim1(c_verti)
        c_hori = Pic_Openings.ouvrirImage(fermeture["horizontale"] + str(k) + ".png")
        c_hori = Pic_Treatments.dim1(c_hori)
        c_hori_verti = Pic_Openings.ouvrirImage(fermeture["verti_hori"] + str(k) + ".png")
        c_hori_verti = Pic_Treatments.dim1(c_hori_verti)

        d_verti = Analogy_computing.cnnTab_to_anaTab_(expert = b_verti,cnn1=a_verti,cnn2=c_verti,methodType="lissage_base_red",im_red=c_verti)

        d_hori= Analogy_computing.cnnTab_to_anaTab_(expert=b_hori, cnn1=a_hori, cnn2=c_hori,methodType="lissage_base_red", im_red=c_hori)

        d_hori_verti = Analogy_computing.cnnTab_to_anaTab_(expert=b_hori_verti, cnn1=a_hori_verti, cnn2=c_hori_verti,methodType="lissage_base_red", im_red=c_hori_verti)

        d_verti = Pic_Treatments.dim3(d_verti)
        d_hori = Pic_Treatments.dim3(d_hori)
        d_hori_verti = Pic_Treatments.dim3(d_hori_verti)

        Pic_Openings.sauvegardeImage(nomFic=fermeture3["verticale"]+str(k)+".png",image=d_verti)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["horizontale"]+str(k)+".png",image=d_hori)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["verti_hori"]+str(k)+".png",image=d_hori_verti)

    return 1


def raisonnement_sur_fermeture_diago(pivot):
    '''
        but: realiser le raisonnement par analogie fc(a):fc(b)::fc(c): fcD
        on veut recuperer fcD obtenu par analogie

            ici: fc est la fermeture diago montante/descendante et les deux
        on le stock dans le dossier basic/cnn_cnn/tumeur/rectangle ou verticale ect

        :param pivot:
        :return:
        '''
    a_verti = Pic_Openings.ouvrirImage(fermeture["diago_mont"]+str(pivot)+".png")
    a_verti = Pic_Treatments.dim1(a_verti)
    a_hori = Pic_Openings.ouvrirImage(fermeture["diago_desc"]+str(pivot)+".png")
    a_hori = Pic_Treatments.dim1(a_hori)
    a_hori_verti = Pic_Openings.ouvrirImage(fermeture["diagos"] + str(pivot) + ".png")
    a_hori_verti = Pic_Treatments.dim1(a_hori_verti)

    b_verti = Pic_Openings.ouvrirImage(fermeture2["diago_mont"] + str(pivot) + ".png")
    b_verti = Pic_Treatments.dim1(b_verti)
    b_hori = Pic_Openings.ouvrirImage(fermeture2["diago_desc"] + str(pivot) + ".png")
    b_hori = Pic_Treatments.dim1(b_hori)
    b_hori_verti = Pic_Openings.ouvrirImage(fermeture2["diagos"] + str(pivot) + ".png")
    b_hori_verti = Pic_Treatments.dim1(b_hori_verti)

    indice = [pivot + i for i in range(-5, 6) if pivot + i >= min and pivot + i <= max]
    for k in indice:
        c_verti = Pic_Openings.ouvrirImage(fermeture["diago_mont"] + str(k) + ".png")
        c_verti = Pic_Treatments.dim1(c_verti)
        c_hori = Pic_Openings.ouvrirImage(fermeture["diago_desc"] + str(k) + ".png")
        c_hori = Pic_Treatments.dim1(c_hori)
        c_hori_verti = Pic_Openings.ouvrirImage(fermeture["diagos"] + str(k) + ".png")
        c_hori_verti = Pic_Treatments.dim1(c_hori_verti)

        #d_verti_red = Analogy_computing.cnnTab_to_anaTab_(expert = b_verti,cnn1=a_verti,cnn2=c_verti,methodType="none")
        d_verti = Analogy_computing.cnnTab_to_anaTab_(expert = b_verti,cnn1=a_verti,cnn2=c_verti,methodType="lissage_base_red",im_red=c_verti)

        #d_hori_red = Analogy_computing.cnnTab_to_anaTab_(expert=b_hori, cnn1=a_hori, cnn2=c_hori, methodType="none")
        d_hori= Analogy_computing.cnnTab_to_anaTab_(expert=b_hori, cnn1=a_hori, cnn2=c_hori,methodType="lissage_base_red", im_red=c_hori)

        #d_hori_verti_red = Analogy_computing.cnnTab_to_anaTab_(expert=b_hori_verti, cnn1=a_hori_verti, cnn2=c_hori_verti, methodType="none")
        d_hori_verti = Analogy_computing.cnnTab_to_anaTab_(expert=b_hori_verti, cnn1=a_hori_verti, cnn2=c_hori_verti,methodType="lissage_base_red", im_red=c_hori_verti)


        d_verti = Pic_Treatments.dim3(d_verti)
        d_hori = Pic_Treatments.dim3(d_hori)
        d_hori_verti = Pic_Treatments.dim3(d_hori_verti)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["diago_mont"]+str(k)+".png",image=d_verti)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["diago_desc"]+str(k)+".png",image=d_hori)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["diagos"]+str(k)+".png",image=d_hori_verti)

    return 1


def raisonnement_sur_fermeture_rectangle(pivot):
    '''
    but: realiser le raisonnement par analogie fc(a):fc(b)::fc(c): fcD
    on veut recuperer fcD obtenu par analogie

        ici: fc est la fermeture rectangle
    on le stock dans le dossier basic/cnn_cnn/tumeur/rectangle ou verticale ect

    :param pivot:
    :return:
    '''

    a_verti = Pic_Openings.ouvrirImage(fermeture["rectangulaire"]+str(pivot)+".png")
    a_verti = Pic_Treatments.dim1(a_verti)
    b_verti = Pic_Openings.ouvrirImage(fermeture2["rectangulaire"] + str(pivot) + ".png")
    b_verti = Pic_Treatments.dim1(b_verti)

    indice = [pivot + i for i in range(-5, 6) if pivot + i >= min and pivot + i <= max]
    for k in indice:
        c_verti = Pic_Openings.ouvrirImage(fermeture["rectangulaire"] + str(k) + ".png")
        c_verti = Pic_Treatments.dim1(c_verti)
        d_verti = Analogy_computing.cnnTab_to_anaTab_(expert=b_verti, cnn1=a_verti, cnn2=c_verti,methodType="lissage_base_red", im_red=c_verti)
        d_verti = Pic_Treatments.dim3(d_verti)
        Pic_Openings.sauvegardeImage(nomFic=fermeture3["rectangulaire"]+str(k)+".png",image=d_verti)

    return 1


def suport_realisation_croisement(d,name,k):
    '''
    function uniquement utile pour factoriser du code, a voir l'usage dans realisation croisement

    on ouvre en memoire l'image issu du raisonnement par analogie sur la fermeture convexe
    portant le nom name

    on realise l'interesection
    puis on l'enregistre comme il faut dans les dossiers fermetures4

    :param d:
    :param name:
    :param k:
    :return:
    '''
    dv = Pic_Openings.ouvrirImage(fermeture3[name] + str(k) + ".png")
    dv = Pic_Treatments.dim1(dv)

    dv = Frame_Performances.do_intersect(dv, d)
    dv = Pic_Treatments.dim3(dv)
    Pic_Openings.sauvegardeImage(image =dv ,nomFic=fermeture4[name] + str(k) + ".png")

    return 1

def realisation_croisement(pivot):
    '''
    But: realiser le croisement entre fcD et l'image d associe
    on le fait pour tous type de fermetures convexes
    on stocke dans "../../Data/basic/"+param+"/cnn_cnn/tumeur/diagos/croise/"

    :param pivot:
    :return:
    '''

    indice = [pivot + i for i in range(-5, 6) if pivot + i >= min and pivot + i <= max]
    for k in indice:
        d = Pic_Openings.ouvrirImage(fermeture3["base"]+str(k)+".png")
        d = Pic_Treatments.dim1(d)
        for name in fermeture2:
            if name != "base":
                suport_realisation_croisement(d,name=name,k=k)

    return 1

def dicing_segment(pivot,min=min,max=max):
    '''
    but: obtenir le dice de la methode
    :param pivot:
    :param min: optionnel
    :param max: optionnel
    :return:
    '''
    indice = [pivot + i for i in range(-5,6) if pivot + i >= min and pivot+i <= max]

    expert_i = [ Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(fermeture2["base"]+str(i)+".png"))) for i in indice]
    cnn_i = [ Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(fermeture["base"]+str(i)+".png"))) for i in indice]
    ovation_i = [ Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../../Data/coupeGap10/tumeur/"+str(i)+".png"))) for i in indice]
    im_ref_i = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture3["base"]+str(i)+".png")) for i in indice]
    im_hor = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["horizontale"]+str(i)+".png")) for i in indice]
    im_vert = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["verticale"]+str(i)+".png")) for i in indice]
    im_verti_hori = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["verti_hori"]+str(i)+".png")) for i in indice]
    im_diago_mont = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["diago_mont"]+str(i)+".png")) for i in indice]
    im_diago_desc = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["diago_desc"]+str(i)+".png")) for i in indice]
    im_diagos = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["diagos"]+str(i)+".png")) for i in indice]
    im_rectangle = [ Pic_Treatments.dim1(Pic_Openings.ouvrirImage(fermeture4["rectangulaire"]+str(i)+".png")) for i in indice]


    Result = [[],[],[],[],[],[],[],[],[],[],[]] # numero, cnn, ovation, ref, hor, vert, both, diago_mont,desc,mont+desc,rectangle
    Result[0] = indice
    Result[1] = [Frame_Performances.indice_dice(expert_i[i],cnn_i[i]) for i in range(len(indice))]
    Result[2] = [Frame_Performances.indice_dice(expert_i[i],ovation_i[i]) for i in range(len(indice))]
    Result[3] = [Frame_Performances.indice_dice(expert_i[i],im_ref_i[i]) for i in range(len(indice))]
    Result[4] = [Frame_Performances.indice_dice(expert_i[i],im_hor[i]) for i in range(len(indice))]
    Result[5] = [Frame_Performances.indice_dice(expert_i[i],im_vert[i]) for i in range(len(indice))]
    Result[6] = [Frame_Performances.indice_dice(expert_i[i],im_verti_hori[i]) for i in range(len(indice))]
    Result[7] = [Frame_Performances.indice_dice(expert_i[i],im_diago_mont[i]) for i in range(len(indice))]
    Result[8] = [Frame_Performances.indice_dice(expert_i[i],im_diago_desc[i]) for i in range(len(indice))]
    Result[9] = [Frame_Performances.indice_dice(expert_i[i],im_diagos[i]) for i in range(len(indice))]
    Result[10] = [Frame_Performances.indice_dice(expert_i[i],im_rectangle[i]) for i in range(len(indice))]

    return Result


def stockage_db_convexe(sortie_dicing_segment,db = DB_PATH):
    '''
    but: stocker sur la db nos precieux dice calcule pour pas les perdre
    :param sortie_dicing_segment:
    :return:
    '''
    if len(sortie_dicing_segment) != 11:
        raise ValueError("Erreure interne")
    sqliteConnection = sqlite3.connect(db)
    sqlite_query = ""
    cursor = sqliteConnection.cursor()
    for k in range(len(sortie_dicing_segment[0])):
        indice = sortie_dicing_segment[0][k]

        cnn = sortie_dicing_segment[1][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'cnn'" + ',' + str(cnn) + ')'
        cursor.execute(sqlite_query)

        ovation = sortie_dicing_segment[2][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'ovation'" + ',' + str(ovation) + ')'
        cursor.execute(sqlite_query)


        ref = sortie_dicing_segment[3][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'ref'" + ',' + str(ref) + ')'
        cursor.execute(sqlite_query)

        hor = sortie_dicing_segment[4][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'horizontale'" + ',' + str(hor) + ')'
        cursor.execute(sqlite_query)

        vert = sortie_dicing_segment[5][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'verticale'" + ',' + str(vert) + ')'
        cursor.execute(sqlite_query)


        both = sortie_dicing_segment[6][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'verti_hori'" + ',' + str(both) + ')'
        cursor.execute(sqlite_query)

        mont = sortie_dicing_segment[7][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'montante'" + ',' + str(mont) + ')'
        cursor.execute(sqlite_query)

        desc = sortie_dicing_segment[8][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'descendante'" + ',' + str(desc) + ')'
        cursor.execute(sqlite_query)

        diagos = sortie_dicing_segment[9][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'diagos'" + ',' + str(diagos) + ')'
        cursor.execute(sqlite_query)

        rect = sortie_dicing_segment[10][k]
        sqlite_query = 'INSERT INTO convexe' \
                       ' (num, type ' \
                       ',dice)  VALUES  ('
        sqlite_query += str(indice) + ',' + "'rectangle'" + ',' + str(rect) + ')'
        cursor.execute(sqlite_query)



        sqliteConnection.commit()
    cursor.close()
    return 1


def extraction_db_convexe(pivot,min = 1,max = 136):
    print(pivot)
    db = DB_PATH
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()

    Result = [[],[],[],[],[],[],[],[],[],[],[]] # numero, cnn, ovation, ref, hor, vert, both, diago_mont,desc,mont+desc,rectangle

    indices = [pivot + i for i in range(-5,6) if pivot + i >= min and pivot+i <= max]
    for indice in indices:
        sqlite_query = ""
        sqlite_query = """select type,dice from convexe where num like "{}";""".format(str(indice))
        cursor.execute(sqlite_query)
        record = cursor.fetchall()
        Result[0].append(indice)
        for k in record:
            if k[0] == "cnn":
                Result[1].append(float(k[1]))
            elif k[0] == "ovation":
                Result[2].append(float(k[1]))
            elif k[0] == "verti_hori":
                Result[6].append(float(k[1]))
            elif k[0] == "ref":
                Result[3].append(float(k[1]))
            elif k[0] == "horizontale":
                Result[4].append(float(k[1]))
            elif k[0] == "verticale":
                Result[5].append(float(k[1]))
            elif k[0] == "montante":
                Result[7].append(float(k[1]))
            elif k[0] == "descendante":
                Result[8].append(float(k[1]))
            elif k[0] == "diagos":
                Result[9].append(float(k[1]))
            elif k[0] == "rectangle":
                Result[10].append(float(k[1]))
    cursor.close()
    return Result



def fermeture_cnn_expert():

    for k in fermeture:
        Pic_Openings.if_not_exist(fermeture[k])
    for k in fermeture2:
        Pic_Openings.if_not_exist(fermeture2[k])
    for k in fermeture3:
        Pic_Openings.if_not_exist(fermeture3[k])

    ##############################
    # matrice de controle
    ##############################
    tracer_fermeture_cnn_expert = not True  # faut il generer les fermetures convexes des cnn_expert ?
    raisonnement_sur_analogie_segment = not True  # faut il generer l'image issu du raisonnment par analogie sur les fermetures ?
    raisonnement_sur_analogie_diago = not True  # faut il generer l'image issu du raisonnment par analogie sur les fermetures ?
    raisonnement_sur_analogie_rectangle = not True  # faut il generer l'image issu du raisonnment par analogie sur les fermetures ?
    do_croisement = not True  # veut tu realiser le croisement d'image
    storage_dice =  not True  # veut tu storer du dice dans la db
    do_tracing_segment =  not True  # veut tu des beaux graphiques ?

    if __name__ == '__main__':
        print(os.cpu_count())
        start = time.time()

        if tracer_fermeture_cnn_expert:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(support_fermeture_cnn_expert, pivots):
                1

        if raisonnement_sur_analogie_segment:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(raisonnement_sur_fermeture_segment, pivots):
                1

        if raisonnement_sur_analogie_diago:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(raisonnement_sur_fermeture_diago, pivots):
                1

        if raisonnement_sur_analogie_rectangle:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(raisonnement_sur_fermeture_rectangle, pivots):
                1

        if do_croisement:
            for k in fermeture4:
                Pic_Openings.if_not_exist(fermeture4[k])
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(realisation_croisement, pivots):
                1

        if storage_dice:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(dicing_segment, pivots):
                stockage_db_convexe(rep)

        if do_tracing_segment:
            p = Pool(os.cpu_count() - 2)

            def to_plot(Result):

                mon_dico = {"CNN": Result[1], "OVASSION": Result[2]}
                mon_dico["ref"] = Result[3]
                mon_dico["hor"] = Result[4]
                mon_dico["vert"] = Result[5]
                mon_dico["hor+vert"] = Result[6]
                mon_dico["mont"] = Result[7]
                mon_dico["desc"] = Result[8]
                mon_dico["mont+desc"] = Result[9]
                mon_dico["rectangle"] = Result[10]
                mon_dico["num"]=Result[0]

                plt.clf()
                plt.title("Dice sur le raisonnement par fermeture convexe horizontale/verticale")
                plt.xlabel("Numero Segmentation")
                plt.ylabel("Indice Dice")
                plt.plot(Result[0], Result[1], label="cnn", color="r")
                plt.plot(Result[0], Result[2], label="ovassion", color="y")
                plt.plot(Result[0], Result[3], label="ref")
                plt.plot(Result[0], Result[4], label="hor", marker=".", ls='')
                plt.plot(Result[0], Result[5], label="vert", marker=".", ls='')
                plt.plot(Result[0], Result[6], label="hor+vert", marker=".", ls='')
                plt.grid()
                plt.legend()
                plt.show()
                """
                boxplotElements = pyplot.boxplot([[1, 2, 3, 4, 5, 13],
                                  [6, 7, 8, 10, 10, 11, 12],
                                  [1, 2, 3]], sym = 'g*', whis = 1.2,
                                 widths = [1, 0.5, 0.2], positions = [1, 3, 4],
                                 patch_artist = True)
pyplot.gca().xaxis.set_ticklabels(['A', 'B', 'C'])
for element in boxplotElements['medians']:
    element.set_color('blue')
    element.set_linewidth(4)
for element in boxplotElements['boxes']:
    element.set_edgecolor('magenta')
    element.set_facecolor('yellow')
    element.set_linewidth(3)
    element.set_linestyle('dashed')
    element.set_fill(True)
    element.set_hatch('/')
for element in boxplotElements['whiskers']:
    element.set_color('red')
    element.set_linewidth(2)
for element in boxplotElements['caps']:
    element.set_color('cyan')
pyplot.ylim(0, 14)
pyplot.title('boxplot avec configuration complete')
                
                """


                plt.clf()
                diff1 = [ Result[4][i] - Result[3][i] for i in range(len(Result[4])) ]
                diff2 = [ Result[5][i] - Result[3][i] for i in range(len(Result[4])) ]
                diff3 = [ Result[6][i] - Result[3][i] for i in range(len(Result[4])) ]
                diff4 = [ Result[1][i] - Result[3][i] for i in range(len(Result[4])) ]

                plt.title("Comparaison Fermeture/Référence")
                plt.xlabel("Numero Segmentation")
                plt.ylabel("Ecart Dice")
                #plt.plot(Result[0], diff4,label="cnn",marker="o",ls="")
                plt.plot(Result[0],diff1,marker=".", ls='',label="hor")
                plt.plot(Result[0], diff2, marker=".", ls='',label="vert")
                plt.plot(Result[0], diff3, marker=".", ls='',label="hor+vert")
                plt.plot(Result[0],[0 for i in range(len(Result[0]))],color="black")
                plt.ylim(-0.015,0.010)
                plt.grid()
                plt.legend()
                plt.show()
                # partie diagramme à moustache

                plt.clf()
                boxplotElements = plt.boxplot([Result[2],Result[1],Result[3],Result[4],Result[5],Result[6]], sym='g*',
                                                 widths=[0.5, 0.5, 0.5,0.5,0.5,0.5],positions = [1, 2,3,4,5,6],
                                                 patch_artist=True)
                plt.gca().xaxis.set_ticklabels(["OVASSION","CNN","Ref","mont","desc","mont+desc"])
                for element in boxplotElements['medians']:
                    element.set_color('blue')
                    element.set_linewidth(4)
                for element in boxplotElements['boxes']:
                    element.set_edgecolor('magenta')
                    element.set_facecolor('yellow')
                    element.set_linewidth(3)
                    element.set_linestyle('dashed')
                    element.set_fill(True)
                    element.set_hatch('/')
                for element in boxplotElements['whiskers']:
                    element.set_color('magenta')
                    element.set_linewidth(2)
                for element in boxplotElements['caps']:
                    element.set_color('magenta')
                plt.ylim(0.6,1)
                plt.title('boxplot resultats fermeture convexe Segment')
                plt.show()





                plt.clf()
                plt.title("Dice sur le raisonnement par fermeture convexe diagonales")
                plt.xlabel("Numero Segmentation")
                plt.ylabel("Indice Dice")
                plt.plot(Result[0], Result[1], label="cnn", color="r")
                plt.plot(Result[0], Result[2], label="ovassion", color="y")
                plt.plot(Result[0], Result[3], label="ref")
                plt.plot(Result[0], Result[7], label="montante", marker=".", ls='')
                plt.plot(Result[0], Result[8], label="descendante", marker=".", ls='')
                plt.plot(Result[0], Result[9], label="ont+desc", marker=".", ls='')
                plt.grid()
                plt.legend()
                plt.show()




                plt.clf()
                plt.title("Dice sur le raisonnement par fermeture convexe rectangulaire")
                plt.xlabel("Numero Segmentation")
                plt.ylabel("Indice Dice")
                plt.plot(Result[0], Result[1], label="cnn", color="r")
                plt.plot(Result[0], Result[2], label="ovassion", color="y")
                plt.plot(Result[0], Result[3], label="ref")
                plt.plot(Result[0], Result[10], label="rectangle", marker=".", ls='')
                plt.grid()
                plt.legend()
                plt.show()

                return 1

            # pour remplir les listes
            Result = [[], [], [], [], [], [], [], [], [], [],[]]  # numero, cnn, ovation, ref, hor, vert, both, diago_mont,desc,mont+desc,rectangle
            for rep in p.map(extraction_db_convexe, pivots):
                Result = [Result[i] + rep[i] for i in range(len(rep))]
            to_plot(Result)

            # pour remplir les listes
            pivots_vbis = pivots[3:-3]
            Result = [[], [], [], [], [], [], [], [], [], [],[]]  # numero, cnn, ovation, ref, hor, vert, both, diago_mont,desc,mont+desc,rectangle
            for rep in p.map(extraction_db_convexe, pivots_vbis):
                Result = [Result[i] + rep[i] for i in range(len(rep))]
            to_plot(Result)

        print("delay time:", time.time() - start)
    return 1

fermeture_cnn_expert()

#######################################################################################################################
#           Tracer lissage 3_d
#
lissage_3D = True
#
#######################################################################################################################

if lissage_3D:

    liste_coeff = []
    liste_coeff.append([0.4, 0.3, 0.2, 0.1])
    liste_coeff.append([0, 0.4, 0.3, 0.3])
    liste_coeff.append([0.25, 0.25, 0.25, 0.25])

    def tracer_lissage_3D(pivot):
        cnn = Pic_Openings.ouvrirImage("../../Data/coupeCnn/tumeur/" + str(pivot) + ".png")
        cnn = Pic_Treatments.niveauGris(cnn)
        cnn = Pic_Treatments.dim1(cnn)
        expert = Pic_Openings.ouvrirImage("../../Data/coupeExpert/tumeur/" + str(pivot) + ".png")
        expert = Pic_Treatments.niveauGris(expert)
        expert = Pic_Treatments.dim1(expert)


        # il est necessaire de formater le saving path
        saving_path = "../../Data/basic/lissage_3D/{}/cnn_cnn/tumeur/"

        indice_raiso = [pivot + i for i in range(-5, 6) if pivot + i >= min+1 and pivot + i <= max-1]
        for k in range(len(indice_raiso)):

            cnn_k = cnn_k = Pic_Openings.ouvrirImage("../../Data/coupeCnn/tumeur/" + str(indice_raiso[k]) + ".png")
            cnn_k = Pic_Treatments.niveauGris(cnn_k)
            cnn_k = Pic_Treatments.dim1(cnn_k)

            cnn_k_gauche = Pic_Openings.ouvrirImage("../../Data/coupeCnn/tumeur/" + str(indice_raiso[k]-1) + ".png")
            cnn_k_gauche = Pic_Treatments.niveauGris(cnn_k_gauche)
            cnn_k_gauche = Pic_Treatments.dim1(cnn_k_gauche)

            cnn_k_droite = Pic_Openings.ouvrirImage("../../Data/coupeCnn/tumeur/" + str(indice_raiso[k] + 1) + ".png")
            cnn_k_droite = Pic_Treatments.niveauGris(cnn_k_droite)
            cnn_k_droite = Pic_Treatments.dim1(cnn_k_droite)

            for i in liste_coeff:
                sav = saving_path.format(str(i[0])+"_"+str(i[1])+"_"+str(i[2])+"_"+str(i[3]))
                print(sav)
                new_im = Analogy_computing.cnnTab_to_anaTab_(expert=expert,cnn1=cnn,cnn2=cnn_k,methodType='lissage_3D',im_droite=cnn_k_droite,im_gauche=cnn_k_gauche)
                new_im = Pic_Treatments.dim3(new_im)
                Pic_Openings.if_not_exist(sav)
                Pic_Openings.sauvegardeImage(nomFic=sav+str(indice_raiso[k])+".png",image=new_im)
        return 1

    def lissage_3D_dincing(pivot):
        # il est necessaire de formater le saving path
        saving_path = "../../Data/basic/lissage_3D/{}/cnn_cnn/tumeur/"

        indice_raiso = [pivot + i for i in range(-5, 6) if pivot + i >= min+1 and pivot + i <= max-1]
        expert_i = [Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(fermeture2["base"] + str(i) + ".png"))) for i in indice_raiso]
        cnn_i = [Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage(fermeture["base"] + str(i) + ".png"))) for i in indice_raiso]
        ovation_i = [Pic_Treatments.dim1(Pic_Treatments.niveauGris(Pic_Openings.ouvrirImage("../../Data/coupeGap10/tumeur/" + str(i) + ".png"))) for i in indice_raiso]

        Result = [[], [], []]  # numero, cnn, ovation
        for k in liste_coeff:
            Result.append([])

        Result[0] = indice_raiso
        Result[1] = [Frame_Performances.indice_dice(expert_i[i], cnn_i[i]) for i in range(len(indice_raiso))]
        Result[2] = [Frame_Performances.indice_dice(expert_i[i], ovation_i[i]) for i in range(len(indice_raiso))]

        compteur = 1
        for i in liste_coeff:
            ngouldoum = str(i[0]) + "_" + str(i[1]) + "_" + str(i[2]) + "_" + str(i[3])
            path = saving_path.format(ngouldoum)
            lissage_3D_i = [Pic_Treatments.dim1(Pic_Openings.ouvrirImage(path.format(ngouldoum) + str(k) + ".png")) for k in indice_raiso]
            Result[2+compteur] = [Frame_Performances.indice_dice(lissage_3D_i[i],expert_i[i]) for i in range(len(lissage_3D_i))]
            compteur+=1

        ref_i = [Pic_Treatments.dim1(Pic_Openings.ouvrirImage("../../Data/basic/lissage_base_red/cnn_cnn/tumeur/"+str(k)+".png")) for k in indice_raiso]
        Result.append([Frame_Performances.indice_dice(ref_i[i],expert_i[i])] for i in range(len(indice_raiso)))

        return Result

    def storing_lissage():
        1



    tracer = not True
    dicing = True


    if __name__ == '__main__':

        print(os.cpu_count())
        start = time.time()

        if tracer:
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(tracer_lissage_3D, pivots):
                1
        if dicing:
            Result = [[], [], []]  # numero, cnn, ovation
            for k in liste_coeff:
                Result.append([])
            Result.append([]) # liste de la ref
            p = Pool(os.cpu_count() - 2)
            for rep in p.map(lissage_3D_dincing, pivots):
                if len(Result)!= len(rep):
                    raise ValueError("internal Error: contact dev")
                for i in range(len(rep)):
                    Result[i] += rep[i]

            def to_plot(Result):

                plt.clf()
                plt.title("Dice Lissage 3D")
                plt.xlabel("Numero Segmentation")
                plt.plot(Result[0], Result[1], label="cnn", color="r")
                plt.plot(Result[0], Result[2], label="ovassion", color="y")
                plt.plot(Result[0], Result[-1], label="ref", color="y")

                for k in range(3,len(Result)):
                    i = liste_coeff[k-3]
                    label =str(i[0]) + "_" + str(i[1]) + "_" + str(i[2]) + "_" + str(i[3])
                    plt.plot(Result[0], Result[k], label=label,marker='.',ls="")

                plt.grid()
                plt.legend()
                plt.show()

                return 1
            to_plot(Result)
            to_plot(Result[20:-20])

        print("delay:",time.time()-start)