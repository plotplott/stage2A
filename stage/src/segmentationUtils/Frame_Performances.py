import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Pic_Openings as Pic_Openings
import time

rein = "/rein/"
tumeur = "/tumeur/"
DATA_PATH = "../../Data/"
BD_PATH = "../../Data/mySql.db"

def indice_dice(matrice1, matrice2, param1="soft", param2="more"):
    '''But : obtenir l'indice Dice sur 2 coupes distinctes
        DICE = 2* | intersection | / ( || + ||)
    '''
    A = 2 * get_card(do_intersect(matrice1, matrice2))
    B = (get_card(matrice1) + get_card(matrice2))

    if A != 0 and B != 0:
        return A / B
    if A == 0 and B == 0:
        return 1
    if B == 0:
        return 1
    else:
        return A / B


'''But: obtenir l'indice de jaccard sur 2 coupes distinctes
    iUi = | intersection | / |Union|
'''
def indice_jaccard(matrice1, matrice2, param1="soft", param2="more"):
    A = get_card(do_intersect(matrice1, matrice2))
    B = get_card(do_union(matrice1, matrice2))

    if A != 0 and B != 0:
        return A / B
    if A == 0 and B == 0:
        return 1
    if B == 0:
        return 1
    else:
        return A / B


def indice_double(matrice1,matrice2):
    '''
    But: calculer d'un coup les 2 indices dans le but d'economiser le calcule d'une intersection
    dans l'optique de reduction des coups
    :param matrice1:
    :param matrice2:
    :return: dice,jacc
    '''
    A = get_card(do_intersect(matrice1, matrice2))
    B = get_card(do_union(matrice1, matrice2))
    C = 2*A
    D = (get_card(matrice1) + get_card(matrice2))

    dice = 0
    jacc = 0

    if C != 0 and D != 0:
        dice = C/D
    if C == 0 and D == 0:
        dice = 1
    if  D == 0:
        dice = 1
    else:
        dice = C/D

    if A != 0 and B != 0:
        jacc =  A / B
    if A == 0 and B == 0:
        jacc = 1
    if B == 0:
        jacc = 1
    else:
        jacc = A / B
    return dice,jacc

'''But : on donne une matrice en entree 
elle est composee de 0 et de 1
elle est de taille n*m
on renvoie le nombre de 1
'''

def get_card(matrice):
    result = 0
    size = matrice.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if (matrice[i][j] != 0):
                result += 1
    return result


'''But : on a deux matrices
composee de 0 et de 1
de taille n*m
on renvoie une matrice de taille n*m compose de 0 et de 1
cette matrice correspond a l'intersection des 2

# on specifie comment on traite le cas d'un inconnu(ie: un pixel a 0.5)
    "hard" supposera qu'on ne sait pas donc on le "sort" de l'intersection
    "soft" supposera qu'on aura fais le bon choix 
'''


def do_intersect(matrice1, matrice2, param="hard"):
    size = matrice1.shape
    new_mat = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if matrice1[i][j] != 0 and matrice2[i][j] != 0:
                if matrice1[i][j] == 1 and matrice2[i][j] == 1:
                    new_mat[i][j] = 1
                else:  # on a qql chose de la forme 0.5/1
                    # param hard: on fait le mauvais choix
                    if param == "hard":
                        pass  # on laisse a 0
                    # param soft: on fait le bon choix
                    else:
                        new_mat[i][j] = 1
    return new_mat


'''but : on a deux matrices en entree composee de 0 et de 1 de taille n*m
on renvoie une matrice de taille n*m composee de 0 et de 1
correspondant a l'union des 2

# on specifie comment on traite le cas d'un inconnu(ie: un pixel a 0.5)
    "less" supposera qu'on ne sait pas donc on le "sort" de l'union
    "more" supposera qu'on aura fais le bon choix, donc on rajoute un element

'''


def do_union(matrice1, matrice2, param="less"):
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

    return new_mat


'''but :
 parcourir les images qui sont dans un dossier terminal ex: "./Data/coupeAnal/tumeur/95/"
 et calculer l'ensemble des indices a chaque fois'''


def get_indice_from_1Frame(directory):
    if not os.path.exists(directory):
        raise ValueError("You Idiot: Directory Not Found")
    file_list = os.listdir(directory)

    list_dice = []  # la liste des indices dice
    list_jac = []  # la liste des indice jaccard
    list_num = []  # la liste des numeros de coupe d'entrainement

    current_slice = file_list[0].split("from")[0]
    # print(current_slice,file_list)
    expertCurrentPath = DATA_PATH + "coupeExpert" + tumeur + str(current_slice) + ".png"
    imageExpertCurrent = Pic_Openings.ouvrirImage(expertCurrentPath)
    matExpertCurrent = Pic_Treatments.dim1(imageExpertCurrent)

    training_slice = 0
    for k in file_list:
        training_slice = k.split("from")[1].split(".")[0]
        list_num.append(training_slice)
        # print(directory+k)
        imageAnalCurrent = Pic_Openings.ouvrirImage(directory + k)
        matAnalCurrent = Pic_Treatments.dim1(imageAnalCurrent)

        list_dice.append(indice_dice(matExpertCurrent, matAnalCurrent))
        list_jac.append(indice_jaccard(matExpertCurrent, matAnalCurrent))

    return list_dice, list_jac, list_num


'''but : 
    store les indices dice/jacard correspondant a une coupe calculee d'apres tel ou tel image d'entrainement
    on precise :
        - num_deduction : le numero de la coupe deduite
        - num_apprentissage : la liste des numeros sur lequel on s'est entrainee
        - indice_dice : la liste des indice dice correspondant
        - indice_jacc : la liste des indices jacc correspondant
        - db : le chemin vers la db sqlite, par defaut c'est BD_PATH = "../Data/mySql.db"
'''


def store_indicator_inDb(num_deduction, list_num_app, list_dice, list_jacc, db=BD_PATH):
    # inscription dans la base de donnes
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()
    sqlite_query = ""
    for k in range(len(list_num_app)):
        sqlite_query = 'INSERT INTO indice (num_deduction, num_apprentissage,indice_dice,indice_jac)  VALUES  ('
        sqlite_query += str(num_deduction) + ',' + str(list_num_app[k]) + ',' + str(list_dice[k]) + ',' + str(
            list_jacc[k])
        sqlite_query = sqlite_query + ')'
        # print(sqlite_query)
        cursor.execute(sqlite_query)
    sqliteConnection.commit()
    cursor.close()
    return 1


'''but : 
parcourir les images qui sont dans un dossier non terminal
ex: "./Data/coupeAnal/tumeur/ 
et performe l'ensemble des calculs d'indice et le store dans la bd
'''


def get_indice_from_allFrame(directory, db=BD_PATH):
    if not os.path.exists(directory):
        raise ValueError("You Idiot: Directory Not Found")

    dir_list = os.listdir(directory)

    for dir in dir_list:
        # le repertoire contenant l'ensemble des n images calculees
        # l'idee est d'obtenir la liste dice_list_jac list_num et de l'inserer dans la base de donnes
        my_slice_dir = directory + dir + '/'
        # print(my_slice_dir)
        num_current_slice = dir
        L = [1, 10, 11, 12, 13, 14, 15, 16, 17, 95]
        # if 1<= int(num_current_slice) <= 35 and int(num_current_slice) not in L:
        if int(num_current_slice) > 35:
            start = time.time()
            list_dice, list_jac, list_num = get_indice_from_1Frame(my_slice_dir)
            # inscription dans la base de donnes
            store_indicator_inDb(num_deduction=num_current_slice, list_num_app=list_num, list_dice=list_dice,
                                 list_jacc=list_jac)
            end = time.time()
            delay = end - start
            print("Indicator Calculation and Storage for:", num_current_slice, "took ", delay, "sec")

    return 1


# get_indice_from_allFrame("../Data/coupeAnal/tumeur/")


'''but :
    on donne en argument : la coupe que l'on souhaite observer
    on obtient en retour la liste dice et jac depuis la bdd
'''


def get_indicator_from_bd(num=95, db=BD_PATH):
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()
    sqlite_query = """select num_apprentissage,indice_dice,indice_jac from indice where num_deduction like "{}";""".format(
        str(num))
    print(sqlite_query)
    cursor.execute(sqlite_query)
    record = cursor.fetchall()
    num_apprentissage = []
    list_dice = []
    list_jacc = []

    for k in record:
        num_apprentissage.append(k[0])
        list_dice.append(k[1])
        list_jacc.append(k[2])

    print(record)
    cursor.close()
    return num_apprentissage, list_dice, list_jacc


'''but : observer de maniere sympa ce que renvoie la fonction get_indice_from_1frame'''


def ploting(dice, jac, num):
    plt.clf()
    x = [i for i in range(len(dice))]
    plt.legend()
    plt.plot(x, dice, label="dice")
    plt.plot(x, jac, label="jacc")
    plt.axvline(x=num)
    plt.show()


'''But : first step, guess pivot ?? 
    to doing this, we are looking on indice between 
    Expert and gap 10, to see is there is some maximum/minimum value
'''


def guess_pivot(expert="../Data/coupeExpert/tumeur/", gap10="../Data/coupeGap10/tumeur/"):
    # ici l'idee est juste de plot l'indice DICE + Jaccard sur une même courbe
    if not os.path.exists(expert) or not os.path.exists(gap10):
        raise ValueError("You Idiot: Directory Not Found")

    list_dice = []  # la liste des indices dice
    list_jac = []  # la liste des indice jacard
    list_num = []  # la liste des coupes visualisees

    file_list_exp = os.listdir(expert)
    file_list_gap = os.listdir(gap10)

    for k in file_list_exp:
        for l in file_list_exp:
            if k.split('.')[0] == l.split('.')[0]:
                list_num.append(int(k.split('.')[0]))

                imageExpert = Pic_Openings.ouvrirImage(expert + k)
                imageGap10 = Pic_Openings.ouvrirImage(gap10 + k)

                matExpert = Pic_Treatments.dim1(imageExpert)
                matGap10 = Pic_Treatments.dim1(imageGap10)

                list_dice.append(indice_dice(matExpert, matGap10))
                list_jac.append(indice_jaccard(matExpert, matGap10))

    return list_num, list_dice, list_jac


def support_dicing(expert,comparaison_liste=[]):
    '''
    But: factoriser du code recurent dans les mains


    :param expert: reference du dice
    :param comparaison_liste: liste a comparer
    :return: liste du type [ [], [] , [] , [] .. ]
    '''

    return [ indice_dice(expert,comparaison_liste[i]) for i in range(len(comparaison_liste))  ]