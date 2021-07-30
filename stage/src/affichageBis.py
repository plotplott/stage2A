import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import statistics

import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import src.segmentationUtils.Frame_Performances as Frame_Performances

def affichage_poids_Dice_tout(chemin):
    '''
        Le but est d'afficher le graphe de l'indice de Dice correspondant au dossier passé en paramètre
        La fonction affiche aussi CNN pur et Ovassion pur
    '''

    x = [i for i in range(1,137)]
    poids = []
    moyenne = []
    '''
    # on créé la liste pour l'affichage de CNN pur
    liste_cnn = [1 for i in range(136)]
    path_cnn = "../Data/coupeCnn/tumeur/"
    file_list = os.listdir(path_cnn)

    for i in file_list:
        num_seg = i.split(".")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageCNN = Pic_Openings.ouvrirImage(path_cnn + i)
        imageCNN = Pic_Treatments.niveauGris(imageCNN)
        matCNN = Pic_Treatments.dim1(imageCNN)

        liste_cnn[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matCNN)
    print("J'ai fini CNN")

    # on créé la liste pour l'affichage d'ovassion pur
    liste_ovassion = [1 for i in range(136)]
    path_ovassion = "../Data/coupeGap10/tumeur/"
    file_list = os.listdir(path_ovassion)

    for i in file_list:
        num_seg = i.split(".")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageOvassion = Pic_Openings.ouvrirImage(path_ovassion + i)
        imageOvassion = Pic_Treatments.niveauGris(imageOvassion)
        matOvassion = Pic_Treatments.dim1(imageOvassion)

        liste_ovassion[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matOvassion)
    print("J'ai fini Ovassion")
    '''
    # on créé la liste pour l'affichage du chemin
    liste_chemin1 = [1 for i in range(136)]
    path = chemin + "0.4-0.3-0.3/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin1[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.4-0.3-0.3")
    moyenne.append(statistics.mean(liste_chemin1))
    print("Liste chemin 1 = " + str(liste_chemin1))
    print("J'ai fini chemin1")

    # on créé la liste pour l'affichage du chemin2
    liste_chemin2 = [1 for i in range(136)]
    path = chemin + "0.4-0.4-0.2/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin2[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.4-0.4-0.2")
    moyenne.append(statistics.mean(liste_chemin2))
    print("Liste chemin 2 = " + str(liste_chemin2))
    print("J'ai fini chemin2")

    # on créé la liste pour l'affichage du chemin3
    liste_chemin3 = [1 for i in range(136)]
    path = chemin + "0.5-0.3-0.2/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin3[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.5-0.3-0.2")
    moyenne.append(statistics.mean(liste_chemin3))
    print("Liste chemin 3 = " + str(liste_chemin3))
    print("J'ai fini chemin3")

    # on créé la liste pour l'affichage du chemin4
    liste_chemin4 = [1 for i in range(136)]
    path = chemin + "0.5-0.4-0.1/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin4[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.5-0.4-0.1")
    moyenne.append(statistics.mean(liste_chemin4))
    print("Liste chemin 4 = " + str(liste_chemin4))
    print("J'ai fini chemin4")

    # on créé la liste pour l'affichage du chemin5
    liste_chemin5 = [1 for i in range(136)]
    path = chemin + "0.5-0.5-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin5[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.5-0.5-0.0")
    moyenne.append(statistics.mean(liste_chemin5))
    print("Liste chemin 5 = " + str(liste_chemin5))
    print("J'ai fini chemin5")

    # on créé la liste pour l'affichage du chemin6
    liste_chemin6 = [1 for i in range(136)]
    path = chemin + "0.6-0.2-0.2/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin6[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.6-0.2-0.2")
    moyenne.append(statistics.mean(liste_chemin6))
    print("Liste chemin 6 = " + str(liste_chemin6))
    print("J'ai fini chemin6")

    # on créé la liste pour l'affichage du chemin7
    liste_chemin7 = [1 for i in range(136)]
    path = chemin + "0.6-0.3-0.1/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin7[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.6-0.3-0.1")
    moyenne.append(statistics.mean(liste_chemin7))
    print("Liste chemin 7 = " + str(liste_chemin7))
    print("J'ai fini chemin7")

    # on créé la liste pour l'affichage du chemin8
    liste_chemin8 = [1 for i in range(136)]
    path = chemin + "0.6-0.4-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin8[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.6-0.4-0.0")
    moyenne.append(statistics.mean(liste_chemin8))
    print("Liste chemin 8 = " + str(liste_chemin8))
    print("J'ai fini chemin8")

    # on créé la liste pour l'affichage du chemin9
    liste_chemin9 = [1 for i in range(136)]
    path = chemin + "0.7-0.2-0.1/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin9[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.7-0.2-0.1")
    moyenne.append(statistics.mean(liste_chemin9))
    print("Liste chemin 9 = " + str(liste_chemin9))
    print("J'ai fini chemin9")

    # on créé la liste pour l'affichage du chemin10
    liste_chemin10 = [1 for i in range(136)]
    path = chemin + "0.7-0.3-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin10[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.7-0.3-0.0")
    moyenne.append(statistics.mean(liste_chemin10))
    print("Liste chemin 10 = " + str(liste_chemin10))
    print("J'ai fini chemin10")

    # on créé la liste pour l'affichage du chemin11
    liste_chemin11 = [1 for i in range(136)]
    path = chemin + "0.8-0.1-0.1/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin11[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.8-0.1-0.1")
    moyenne.append(statistics.mean(liste_chemin11))
    print("Liste chemin 11 = " + str(liste_chemin11))
    print("J'ai fini chemin11")

    # on créé la liste pour l'affichage du chemin12
    liste_chemin12 = [1 for i in range(136)]
    path = chemin + "0.8-0.2-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin12[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.8-0.2-0.0")
    moyenne.append(statistics.mean(liste_chemin12))
    print("Liste chemin 12 = " + str(liste_chemin12))
    print("J'ai fini chemin12")

    # on créé la liste pour l'affichage du chemin13
    liste_chemin13 = [1 for i in range(136)]
    path = chemin + "0.9-0.1-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin13[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("0.9-0.1-0.0")
    moyenne.append(statistics.mean(liste_chemin13))
    print("Liste chemin 13 = " + str(liste_chemin13))
    print("J'ai fini chemin13")

    # on créé la liste pour l'affichage du chemin14
    liste_chemin14 = [1 for i in range(136)]
    path = chemin + "1.0-0.0-0.0/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin14[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    poids.append("1.0-0.0-0.0")
    moyenne.append(statistics.mean(liste_chemin14))
    print("Liste chemin 14 = " + str(liste_chemin14))
    print("J'ai fini chemin14")

    #plt.plot(x, liste_cnn, label='CNN pur')
    #plt.plot(x, liste_ovassion, label = 'Ovassion pur')
    plt.plot(x, liste_chemin1, label = '0.4-0.3-0.3')
    plt.plot(x, liste_chemin2, label='0.4-0.4-0.2')
    plt.plot(x, liste_chemin3, label='0.5-0.3-0.2')
    plt.plot(x, liste_chemin4, label='0.5-0.4-0.1')
    plt.plot(x, liste_chemin5, label='0.5-0.5-0.0')
    plt.plot(x, liste_chemin6, label='0.6-0.2-0.2')
    plt.plot(x, liste_chemin7, label='0.6-0.3-0.1')
    plt.plot(x, liste_chemin8, label='0.6-0.4-0.0')
    plt.plot(x, liste_chemin9, label='0.7-0.2-0.1')
    plt.plot(x, liste_chemin10, label='0.7-0.3-0.0')
    plt.plot(x, liste_chemin11, label='0.8-0.1-0.1')
    plt.plot(x, liste_chemin12, label='0.8-0.2-0.0')
    plt.plot(x, liste_chemin13, label='0.9-0.1-0.0')
    plt.plot(x, liste_chemin14, label='1.0-0.0-0.0')
    plt.xlim(19, 131)
    plt.legend()
    plt.xlabel('Numéro du scan')
    plt.ylabel('Indice de Dice')
    plt.title("Représentation de l'indice de Dice pour différents poids \ndans la méthode de convolution")
    plt.show()

    plt.plot(poids, moyenne)
    plt.xlabel('Poids pour la méthode de convolution')
    plt.ylabel('Moyenne des indices de Dice')
    plt.title("Représentation des moyennes des indices de Dice pour différents poids \ndans la méthode de convolution")
    plt.show()

def affichage_poids_Dice(chemin):
    '''
        Le but est d'afficher le graphe de l'indice de Dice correspondant au dossier passé en paramètre
        La fonction affiche aussi CNN pur et Ovassion pur
    '''

    x = [i for i in range(1,137)]
    '''
    # on créé la liste pour l'affichage de CNN pur
    liste_cnn = [1 for i in range(136)]
    path_cnn = "../Data/coupeCnn/tumeur/"
    file_list = os.listdir(path_cnn)

    for i in file_list:
        num_seg = i.split(".")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageCNN = Pic_Openings.ouvrirImage(path_cnn + i)
        imageCNN = Pic_Treatments.niveauGris(imageCNN)
        matCNN = Pic_Treatments.dim1(imageCNN)

        liste_cnn[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matCNN)
    print("J'ai fini CNN")

    # on créé la liste pour l'affichage d'ovassion pur
    liste_ovassion = [1 for i in range(136)]
    path_ovassion = "../Data/coupeGap10/tumeur/"
    file_list = os.listdir(path_ovassion)

    for i in file_list:
        num_seg = i.split(".")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageOvassion = Pic_Openings.ouvrirImage(path_ovassion + i)
        imageOvassion = Pic_Treatments.niveauGris(imageOvassion)
        matOvassion = Pic_Treatments.dim1(imageOvassion)

        liste_ovassion[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matOvassion)
    print("J'ai fini Ovassion")
    '''
    # on créé la liste pour l'affichage du chemin
    liste_chemin = [1 for i in range(136)]
    file_list = os.listdir(chemin)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/tumeur/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(chemin + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)
    print("J'ai fini chemin")

    # on créé la liste pour l'affichage du chemin
    liste_chemin2 = [1 for i in range(136)]
    path = "../Data/lissage_after/rein/0.5-0.3-0.2/"
    file_list = os.listdir(path)

    for i in file_list:
        num_seg = i.split("from")[0]

        path_expert = "../Data/coupeExpert/rein/" + str(num_seg) + ".png"
        imageExpert = Pic_Openings.ouvrirImage(path_expert)
        imageExpert = Pic_Treatments.niveauGris(imageExpert)
        matExpert = Pic_Treatments.dim1(imageExpert)

        imageChemin = Pic_Openings.ouvrirImage(path + i)
        imageChemin = Pic_Treatments.niveauGris(imageChemin)
        matChemin = Pic_Treatments.dim1(imageChemin)

        liste_chemin2[int(num_seg) - 1] = Frame_Performances.indice_dice(matExpert, matChemin)

    #plt.plot(x, liste_cnn, label='CNN pur')
    #plt.plot(x, liste_ovassion, label = 'Ovassion pur')
    plt.plot(x, liste_chemin, label='Méthode avec convolution pour la tumeur')
    plt.plot(x, liste_chemin2, label='Méthode avec convolution pour le rein')
    plt.xlim(13, 131)
    plt.legend()
    plt.xlabel('Numéro du scan')
    plt.ylabel('Indice de Dice')
    plt.title("Représentation de l'indice de Dice avec la méthode de convolution \npour les poids 0.5-0.3-0.2 avec la tumeur et le rein")
    plt.show()

affichage_poids_Dice("../Data/lissage_after/0.5-0.3-0.2/")