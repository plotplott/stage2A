import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Pic_Openings as Pic_Openings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import time
import src.segmentationUtils.Segmentation as Segmentation
import os
import src.segmentationUtils.Frame_Performances as Frame_Performances
from multiprocessing import Pool
#import cv2
from scipy import ndimage
from skimage import measure


def cnnTab_to_anaTab_(expert, cnn1, cnn2, methodType='none', im_red=None,im_droite=None,im_gauche=None,vector_3D = [0.4, 0.3, 0.2, 0.1]):
    """
        But : on a 3 numpy.ndarray:
        1: segmentation de la coupeAnal sur une autre coupe
        2: segmentation du cnn sur une autre coupe
        3: segmentation du cnn sur notre coupe actuelle
        On renvoie une segmentation améliorée de notre coupe actuelle selon a:b::c:d

        L'argument Type permet de specifier la maniere dont on va traiter le cas ou l'equation est insoluble :
            none : on ecrit 0.5 pour dire 'je sais pas'
            classic : on fait confiance a la valeur en c
            SK : applique la méthode de Sheldon-Klein
            lissage_base_red : methode lissage_base_red
            lissage_3D : methode lissage en prenant la 3° dimension
                dans ce cas là, il faut preciser les 2 images support via le parametre im_red

            par default on laisse a none
    :param expert: image source expert b
    :param cnn1: image source cnn a
    :param cnn2: image cible cnn c
    :param methodType: "none" "classic" "lissage_base_red" "lissage_3D" ou "sheldon"
    :param im_red: list image

    :param im_droite: matrice pour lissage 3d à droite
    :param im_gauche: matrice pour lissage 3d à gauche
    :return: image
    """

    size = expert.shape
    row, col = size[0], size[1]

    my_segmentation = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            a = cnn1[i][j]
            b = expert[i][j]
            c = cnn2[i][j]
            if not isUnknown(a, b, c):
                my_segmentation[i][j] = compute_d_known(a, b, c)
            elif methodType == 'none':
                my_segmentation[i][j] = 0.5
            elif methodType == 'classic':
                my_segmentation[i][j] = c
            elif methodType == 'SK':
                my_segmentation[i][j] = a
            elif methodType == 'lissage_base_red':
                # print("a,b,c:", a, b, c, isUnknown(a, b, c))
                # print("lissage_base_red")
                # ici on veut utiliser la fonction compute lissage_base_red
                # pour ce faire, il nous faut la matrice 3x3 associe
                value = 0.5
                if 0 < i < row - 1:
                    if 0 < j < col - 1:
                        mat3x3 = get_mat3x3(im_red, i, j)
                        value = compute_d_lissage(mat3x3)
                my_segmentation[i][j] = value
            elif methodType == 'lissage_3D':
                # ne peut etre realise si et seulement si la frame est dans [| 2, 135 |]
                if im_droite.shape != im_gauche.shape or im_droite.shape != expert.shape:
                    raise ValueError("Every grid need to have same size")
                if 0 < i < row - 1:
                    if 0 < j < col - 1:
                        mat3x3_centre = get_mat3x3(cnn2, i, j)
                        mat3x3_droite = get_mat3x3(im_droite,i,j)
                        mat3x3_gauche = get_mat3x3(im_gauche,i,j)
                        value = compute_d_lissage_3D(mat3x3_gauche=mat3x3_gauche, mat3x3_centre=mat3x3_centre, mat3x3_droite=mat3x3_droite, vector=vector_3D)
    return my_segmentation


def do_lissage(mat1, mat2=None, mat3=None, dimension="1D",only_incertain = False):
    '''

    :param mat1: la matrice au centr
    :param mat2: la matrice au bord bas (en realite y a pas de difference dans le calcul mais on precise pour voila quoi)
    :param mat3: la matrice au bord haut
    :param dimension: permet de savoir si on prefere faire sur "1D" (lissage sur 1 frame) ou "3D" (lissage sur 3 frames concecutives
    :param only_incertain: True or False, precise si nous realisons le lsisage uniquement sur les valeures entre 0 et 1
    :return: une matrice prete a etre sauvegarde
    '''

    if not dimension == "1D" and (type(mat2) == type(None) or type(mat3) == type(None)):
        raise ValueError("Tu dois specifier les 2 matrices autour pour realiser le lissage sur 2 dimensions")

    size = mat1.shape
    row, col = size[0], size[1]

    new_matrice = np.zeros((row, col))

    # ici on s'interesse pas aux pixels de bordure car on ne peux extraire un masque de convolution 3x3 (x3)
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if mat1[i][j] == 0.5:
                # dans ce cas la on doit operer un lissage
                # deja on peut choper la matrice 3x3 issu de mat1 et du pixel i,j
                mat3x3_2 = get_mat3x3(mat1, i, j)
                if dimension == "1D":
                    new_value = compute_d_lissage(mat3x3=mat3x3_2)
                    new_matrice[i][j] = new_value
                elif dimension == "3D":
                    mat3x3_1 = get_mat3x3(mat2, i, j)
                    mat3x3_3 = get_mat3x3(mat3, i, j)
                    new_value = compute_d_lissage_3D(mat3x3_gauche=mat3x3_1, mat3x3_centre=mat3x3_2, mat3x3_droite=mat3x3_3)
                    new_matrice[i][j] = new_value
                else:
                    raise ValueError("Le parametre dimension doit etre '1D' ou '3D'")
            else:
                new_matrice[i][j] = mat1[i][j]
    return new_matrice


def compute_d_known(a, b, c):
    """
    But : obtenir la valeur de x de l'equation a:b::c:x
        rmq cette fonction est a utiliser SEULEMENT lorsque l'inconnue existe
    """
    if isUnknown(a, b, c):
        raise ValueError('You must use this function only if the equation is solvable')
    if (a == 0):
        # a = 0
        if (b == 0):
            # a = 0, b = 0
            if (c == 0):
                # a = 0, b = 0, c = 0
                return 0
            else:
                # a = 0, b = 0, c = 1
                return 1

        else:
            # a = 0, b = 1
            if (c == 0):
                # a = 0, b = 1, c = 0
                return 1
            else:
                # a = 0, b = 1, c = 1
                ''' par defaut on prend la valeur de c '''
                return 0.5

    else:
        # a = 1
        if (b == 0):
            # a = 1, b = 0
            if (c == 0):
                # a = 1, b = 0, c = 0
                ''' par defaut on prend la valeur de c '''
                return 0.5
            else:
                # a = 1, b = 0, c = 1
                return 0
        else:
            # a = 1, b = 1
            if (c == 0):
                # a = 1, b = 1, c = 0
                return 0
            else:
                # a = 1, b = 1, c = 1
                return 1
    return 1


def isUnknown(a, b, c):
    '''But : permet de savoir si l'equation est insoluble ou non'''
    if a == 1 and b == 0 and c == 0:
        return True
    elif a == 0 and b == 1 and c == 1:
        return True
    return False


def compute_d_lissage(mat3x3, weight_vector=[0.5, 0.3, 0.2]):
    """
    But: a partir d'une matrice 3x3 dont le centre est le pixel indecis,
    renvoie la valeur en tenant compte des voisins
    Rappel de la methode de lissage_base_red utilisée ici, on a besoin de :
        - un vector de poids [Wc,Wb,Wd] tel que Wc+Wd+Wb = 1 (base normé)
        - une matrice 3x3 telle que:
            0,0 0,2 2,0 2,2 sont des pixels de la diagonales a valeur dans {0,1}
            0,1 1,0 1,2 2,1 sont des pixels de bords a valeur dans {0,1}
            1,1 est le pixel du centre a valeur dans {0,0.5,1}
        - on calcul le poids Wc* C + 1/4 * Wb * somme(Wb) + 1/4 * Wd * somme(Wd)
        - si le poids < 0.5 on renvoie 0
                      = 0.5 renvoie 0.5
                      > 0.5 renvoie 1
    """
    if np.__name__ != type(mat3x3).__module__:
        # on a pas un numpy array
        raise TypeError("Lissage Function need a numpy array")
    # we need to get Weigh Value:
    poids = weight_vector[0] * mat3x3[1, 1]
    poids = poids + 1 / 4 * weight_vector[1] * (mat3x3[0, 1] + mat3x3[1, 0] + mat3x3[1, 2] + mat3x3[2, 1])
    poids = poids + 1 / 4 * weight_vector[2] * (mat3x3[0, 0] + mat3x3[0, 2] + mat3x3[2, 0] + mat3x3[2, 2])

    if not 0 <= poids <= 1:
        raise ValueError("Weight Calculus isn't good")
    incertitude = 10 ** (-5)
    if (poids + incertitude) < 0.5:
        return 0
    elif 0.5 - incertitude <= poids <= 0.5 + incertitude:
        print("poteau")
        return 0.5
    else:
        return 1
    return 1


def compute_d_lissage_3D(mat3x3_gauche, mat3x3_centre, mat3x3_droite, vector=[0.4, 0.3, 0.2, 0.1]):
    """But, a partir de 3 matrices, on cherche la valeur du centre
        mat3x3_1 et mat3x3_3 sont les matrices exterieures
        mat3x3_2 est la matrice interieure
    """
    if np.__name__ != type(mat3x3_gauche).__module__ or np.__name__ != type(mat3x3_centre).__module__ or np.__name__ != type(
            mat3x3_droite).__module__:
        # on a pas un numpy array
        raise TypeError("Lissage Function need a numpy array")

    if not 1-10**(-5)<= sum(vector) <= 1+10**(-5):
        raise ValueError("weigh_vector needs to be normalize at 1")

    poids = vector[0] * mat3x3_centre[1][1]  # le centre

    two = vector[1] * (1/6) * ( mat3x3_centre[0][1] + mat3x3_centre[1][0]+mat3x3_centre[1][2]+mat3x3_centre[2][1]+mat3x3_gauche[1][1]+mat3x3_droite[1][1] )
    poids += two

    three_sup =mat3x3_centre[0][0]+mat3x3_centre[0][2]+mat3x3_centre[2][0]+mat3x3_centre[2][2]
    three_sup += mat3x3_droite[0][1]+mat3x3_droite[1][0]+mat3x3_droite[1][2]+mat3x3_droite[2][1]
    three_sup += mat3x3_gauche[0][1]+mat3x3_gauche[1][0]+mat3x3_gauche[1][2]+mat3x3_gauche[2][1]
    three = vector[2] * (1/12) * (three_sup)
    poids += three

    poids = poids + (1 / 8)*vector[-1] * (
            mat3x3_gauche[0][0] + mat3x3_gauche[0][2] + mat3x3_gauche[2][0] + mat3x3_gauche[2][2] + mat3x3_droite[0][0] + mat3x3_droite[0][2] +
            mat3x3_droite[2][0] + mat3x3_droite[2][2])

    if not 0 <= poids <= 1:
        print("poids",poids)
        raise ValueError("Weight Calculus isn't good")
    incertitude = 10 ** (-5)
    if (poids + incertitude) < 0.5:
        return 0
    elif 0.5 - incertitude <= poids <= 0.5 + incertitude:
        print("poteau")
        return 0.5
    else:
        return 1


def get_mat3x3(cnn2, i, j):
    """But: reconstuire la matrice 3x3 qu'on souhaite"""
    size = cnn2.shape

    row, col = size[0], size[1]

    if not 0 < i < row - 1 or not 0 < j < col - 1:
        raise ValueError('center must not be in border')

    mat3x3 = np.ones((3, 3))
    for k in range(-1, 2, 1):
        for l in range(-1, 2, 1):
            mat3x3[k + 1][l + 1] = cnn2[i + k][j + l]
    return mat3x3


def red_calculation(method_type="none", sous_repertoire="red"):
    """
        But: a:b::c:x => calculer x
             on prend a/c dans cnn / cnn_ovation
        entree: method type
                sous_repertoire dans ../Data/basic" + sous repertoire + "cnn_cnn/tumeur/.."
    """
    CNN_OVATION_PATH = "../Data/coupeGap10/rein/"
    CNN_PATH = "../Data/coupeCnn/rein/"
    EXPERT_PATH = "../Data/coupeExpert/rein/"

    CNN_TO_CNN_PATH = "../Data/basic/" + sous_repertoire + "/cnn_cnn/rein/"
    CNN_TO_OVATION_PATH = "../Data/basic/" + sous_repertoire + "/cnn_ovation/rein/"
    OVATION_TO_CNN_PATH = "../Data/basic/" + sous_repertoire + "/ovation_cnn/rein/"
    OVATION_TO_OVATION_PATH = "../Data/basic/" + sous_repertoire + "/ovation_ovation/rein/"

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
    start = time.time()
    for pivot in pivots:
        # entrainment a, b
        b1 = Pic_Openings.ouvrirImage(nomFic=EXPERT_PATH + str(pivot) + ".png")
        b1 = Pic_Treatments.niveauGris(b1)
        b1 = Pic_Treatments.dim1(b1)
        a1 = Pic_Openings.ouvrirImage(nomFic=CNN_PATH + str(pivot) + ".png")
        a1 = Pic_Treatments.niveauGris(a1)
        a1 = Pic_Treatments.dim1(a1)
        a2 = Pic_Openings.ouvrirImage(nomFic=CNN_OVATION_PATH + str(pivot) + ".png")
        a2 = Pic_Treatments.niveauGris(a2)
        a2 = Pic_Treatments.dim1(a2)

        # entrainment c,c
        for i in range(-5, 6, 1):
            if pivot + i >= min and pivot + i <= max:
                c1 = Pic_Openings.ouvrirImage(CNN_PATH + str(pivot + i) + ".png")
                c1 = Pic_Treatments.niveauGris(c1)
                c1 = Pic_Treatments.dim1(c1)
                c2 = Pic_Openings.ouvrirImage(CNN_OVATION_PATH + str(pivot + i) + ".png")
                c2 = Pic_Treatments.niveauGris(c2)
                c2 = Pic_Treatments.dim1(c2)

                # ici le but est de calculer les differents d1,d2,d3,d4 puis de les sauvegarder
                mat1 = cnnTab_to_anaTab_(expert=b1, cnn1=a1, cnn2=c1, methodType=method_type)
                mat2 = cnnTab_to_anaTab_(expert=b1, cnn1=a1, cnn2=c2, methodType=method_type)
                mat3 = cnnTab_to_anaTab_(expert=b1, cnn1=a2, cnn2=c1, methodType=method_type)
                mat4 = cnnTab_to_anaTab_(expert=b1, cnn1=a2, cnn2=c2, methodType=method_type)

                d1 = Pic_Treatments.dim3(mat1)
                Pic_Openings.sauvegardeImage(CNN_TO_CNN_PATH + str(pivot + i) + "from" + str(pivot) + ".png", d1)
                d2 = Pic_Treatments.dim3(mat2)
                Pic_Openings.sauvegardeImage(CNN_TO_OVATION_PATH + str(pivot + i) + "from" + str(pivot) + ".png", d2)
                d3 = Pic_Treatments.dim3(mat3)
                Pic_Openings.sauvegardeImage(OVATION_TO_CNN_PATH + str(pivot + i) + "from" + str(pivot) + ".png", d3)
                d4 = Pic_Treatments.dim3(mat4)
                Pic_Openings.sauvegardeImage(OVATION_TO_OVATION_PATH + str(pivot + i) + "from" + str(pivot) + ".png",
                                             d4)

        print(Segmentation.Bcolors.WARNING + str(pivot) + Segmentation.Bcolors.RESET)

    end = time.time()

    print("delay time", end - start)


def do_oracle(pivot1_a, pivot1_b, pivot2_a, pivot2_b, cible_a, cible_expert, param1="defaut"):
    '''
    Cette fonction permet de "fusionner" 2 images qui seraient obtenues en rélaisant le raisonnement par analogie
    sur les 2 pivots les plus proches d'une segmentation cible
    Lorsque les 2 segmentations calculées sont d'accord entre elles il n'y a pas de soucis
    Si elles ne le sont pas, on regarde l'expert
    Si il y a des inconnus, on prend celle qui sait ou l'expert
    :param pivot1_a: image {0,1} cnn du premier pivot
    :param pivot1_b: image {0,1} expert du premier pivot
    :param pivot2_a: image {0,1} cnn du second pivot
    :param pivot2_b: image {0,1} expert du second pivot
    :param cible_a:  image {0,1} du cnn cible
    :param cible_expert:  image {0,1} expert de la cible
    :param param1: parametre indiquant comment resoudre le cas ou il y a 1 incertain
        "defaut" signifie qu'on selectionne celui qui sait
        "expert" sifgnifie qu'on selectionne l'expert
    :return:  image {0,1} correspondante a la fusion des deux segmentations obtenue

    But dans tous ça ? tracer le dice(fusion, expert) et le comparer aux autres methodes
    '''

    image1 = cnnTab_to_anaTab_(expert=pivot1_b, cnn1=pivot1_a, cnn2=cible_a)
    image2 = cnnTab_to_anaTab_(expert=pivot2_b, cnn1=pivot2_a, cnn2=cible_a)

    size = pivot1_a.shape
    row, col = size[0], size[1]
    # image1 et image2 sont des images n*m {0,0.5,1}
    # l'idee va etre de les fusionner

    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if image1[i][j] in (0, 1) and image2[i][j] in (0, 1):  # cas ou les 2 sont certaines
                if image1[i][j] == image2[i][j]:  # cas ou ils sont d'accord
                    result[i][j] = image1[i][j]
                else:  # cas ou ils ne sont pas d'accord
                    result[i][j] = cible_expert[i][j]
            elif image1[i][j] not in (0, 1) and image2[i][j] in (
            0, 1):  # cas ou l'image 1 est incertaine mais l'image2 incertaine
                if param1 == "defaut":
                    result[i][j] = image2[i][j]
                else:
                    result[i][j] = cible_expert[i][j]
            elif image1[i][j] in (0, 1) and image2[i][j] not in (
            0, 1):  # cas ou l'image 2 est incertaine mais iamge1 est certaine
                if param1 == "defaut":
                    result[i][j] = image1[i][j]
                else:
                    result[i][j] = cible_expert[i][j]
            else:  # cas ou les deux sont incertains
                result[i][j] = cible_expert[i][j]

    return result

def fill(data, start_coords, fill_value):
    """
    Flood fill algorithm

    Parameters
    ----------
    data : (M, N) ndarray of uint8 type
        Image with flood to be filled. Modified inplace.
    start_coords : tuple
        Length-2 tuple of ints defining (row, col) start coordinates.
    fill_value : int
        Value the flooded area will take after the fill.

    Returns
    -------
    None, ``data`` is modified inplace.
        """

    xsize, ysize = data.shape
    orig_value = data[start_coords[0], start_coords[1]]

    stack = set(((start_coords[0], start_coords[1]),))
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:
        x, y = stack.pop()

        if data[x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))

def get_contour(im):
    '''
    obtenir la liste des contours de forme
    :param im: numpy.ndarrays(row,col)
    :return: list(numpy.ndarrays(1,2)) contenant les contours
    '''
    image = im.copy()
    row, col = image.shape[0], image.shape[1]
    for i in range(row):
        image[0][i] = 1
        image[row - 1][i] = 1
        image[i][0] = 1
        image[i][row - 1] = 1
    contours = measure.find_contours(image, 0.99)
    # le premier contour est celui qui sert a tracer,
    # on le zapp
    return contours[1:]

def trace_contour(row,col,contour):
    '''
    tracer le contour sur une matrice
    :param row: nb ligne
    :param col: nb colonne
    :param contour: numpy.ndarrays(1,2)
    :return: numpy.ndarrays(1,2) avec contour
    '''
    result = np.zeros((row,col))
    for i in contour:
        x = round(i[0])
        y = round(i[1])
        result[int(x)][int(y)] = 1
    return result

def fill_contour(mat):
    row = mat.shape[0]
    result = mat.copy()
    for i in range(row):
        args = np.argwhere(mat[i, :] > 0)
        if args.shape[0] != 0:
            list = args.tolist()
            fill_point = (i + 1, list[0][0] + 1)
            if mat[fill_point[0]][fill_point[1]] == 0:
                fill(result, fill_point, 1)
                return result
    return result

def do_fermeture_convexe(mat, param = "horizontale"):
    '''

    :param mat: matrice n*m * {0,1}
    :param param:
        - "horizontale" par defaut/
        - "verticale" /
        - "diagnoale_montante"
        - "diagonale_descendante"
        - "rectangulaire"
    :return:
    '''
    if np.__name__ != type(mat).__module__:
        # on a pas un numpy array
        raise TypeError("do_fermeture_convexe function need a numpy array")
    row,col = mat.shape[0],mat.shape[1]
    if not row == col:
        raise ValueError("do_fermeture_convexe cas ou la matrice n'est pas un carre non implementee encore")

    new_mat = np.zeros((row,col))

    if param == "horizontale":
        for i in range(row):
            ligne_i = mat[i,:]
            if ligne_i.__contains__(1) or ligne_i.__contains__(0.5):
                min,max = -1,-1
                trouve = False
                for j in range(col):
                    if ligne_i[j] != 0:
                        if not trouve:
                            trouve = True
                            min,max = j,j
                        else:
                            max = j
                for j in range(min,max+1):
                    new_mat[i,j] = 1

    elif param == "verticale":
        for j in range(col):
            col_j = mat[:,j]
            if col_j.__contains__(1) or col_j.__contains__(0.5):
                min,max = -1,-1
                found = False
                for i in range(row):
                    if col_j[i] != 0:
                        if not found:
                            min,max = i,i
                            found = True
                        else:
                            max = i
                for i in range(row):
                    if i >= min and i <= max:
                        new_mat[i][j] = 1

    elif param == "diago_descendante":
        # il y a row + col - 1 diagonales dans le bail
        # ce chiffre est TOUJOURS impaire
        # on sera toujours dans le cas ou row = col

        for k in range(1,col):
            # au mieux la diago a row-1 element
            # au pire elle en a un
            diago_i = []
            taille = row - k
            row_min = 0
            current_row = row_min
            row_max = taille - 1
            col_min = k
            current_col = col_min
            col_max = col -1

            for l in range(taille):
                diago_i.append(mat[current_row][current_col])
                current_row+=1
                current_col+=1
            if len(diago_i) != taille:
                raise Exception("Bug in do_fermeture_convexe function")

            diago_i = np.array(diago_i)
            if diago_i.__contains__(1) or diago_i.__contains__(0.5):
                min,max = 0,0
                found = False
                for k in range(taille):
                    if diago_i[k] != 0:
                        if not found:
                            found = True
                            min,max = k,k

                        else:
                            max = k

                col_min = row - taille
                current_col = col_min
                current_row = 0

                for k in range(taille):
                    if k >= min and k <= max:
                        new_mat[current_row][current_col] = 1
                    current_row += 1
                    current_col += 1

        for k in range(1,row):
            # on traite les diagonales en dessous de la diagonale principale
            diago_i = []
            taille_diago = row - k


            row_min = row - taille_diago
            col_min = 0
            found = False
            min, max = -1, -1
            for i in range(taille_diago):
                if mat[row_min][col_min] != 0 and not found:
                    found = True
                    min,max = i,i
                if found and mat[row_min][col_min] != 0:
                    max = i
                diago_i.append(mat[row_min][col_min])
                row_min += 1
                col_min += 1

            if len(diago_i) != taille_diago:
                raise Exception("Erreur dans la fonction do_fermeture_convexe, contact Romain")

            diago_i = np.array(diago_i)
            if diago_i.__contains__(1) or diago_i.__contains__(0.5):
                row_min = row - taille_diago
                col_min = 0
                for i in range(taille_diago):
                    if i >= min and i <= max:
                        new_mat[row_min][col_min] = 1
                    row_min += 1
                    col_min += 1
        for k in range(1,2):
            # diago centrale
            taille_diago = row
            current_row = 0
            current_col = 0
            diago_i = []
            found = False
            min,max = -1,-1
            for i in range(taille_diago):
                if not found and mat[current_row][current_col] != 0:
                    found = True
                    min,max = i,i
                elif found and mat[current_row][current_col] != 0:
                    max = i
                current_row += 1
                current_col += 1

            current_row = 0
            current_col = 0
            if found:
                for i in range(taille_diago):
                    if i >= min and i <= max:
                        new_mat[current_row][current_col] = 1
                    current_row += 1
                    current_col += 1

    elif param == "diago_montante":
        # il y a row + col - 1 diagonales montantes
        for k in range(0,row):
            row_min = row -1 -k
            col_min = 0
            taille = col - k
            found = False
            min,max = -1,-1
            for i in range(taille):
                if mat[row_min][col_min] != 0:
                    if not found:
                        found = True
                        min,max = i,i
                    if found:
                        max = i
                row_min -= 1
                col_min += 1
            if found:
                row_min = row -1 - k
                col_min = 0
                for i in range(taille):
                    if i >= min and i <= max:
                        new_mat[row_min][col_min] = 1
                    row_min -= 1
                    col_min +=1

        for k in range(1,row):
            taille = row - 1 -k
            row_min = row-1
            col_min = k
            found = False
            min,max = -1,-1
            for i in range(taille):
                if mat[row_min][col_min] != 0:
                    if not found:
                        min,max = i,i
                        found = True
                    else:
                        max = i
                row_min -= 1
                col_min += 1
            if found:
                row_min = row - 1
                col_min = k
                for i in range(taille):
                    if i >= min and i <= max:
                        new_mat[row_min][col_min] = 1
                    row_min -= 1
                    col_min += 1

    elif param == "rectangulaire":
        line_min,line_max,col_min,col_max = Pic_Treatments.get_rectangle_coords_minimal(mat)
        for i in range(line_min,line_max+1):
            for j in range(col_min,col_max+1):
                new_mat[i][j]=1




    else:
        raise ValueError("do_fermeture_convexe function needs a param valid")

    return new_mat

