import numpy as np
import src.segmentationUtils.Pic_Openings as Pic_Openings
import matplotlib.pyplot as plt

rein = "/rein/"
tumeur = "/tumeur/"
DATA_PATH = "../../Data/"

class Bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR

def niveauGris(image):
    # passe d'une image en couleur à une image en niveau de gris
    l,c,k = image.shape
    for i in range (l):
        for j in range(c):
            if (image[i][j][0] == 1 and image[i][j][1] == 0 and image[i][j][2] == 0):
                raise ValueError(
                    Bcolors.WARNING + "ERREUR : Vous tentez de supprimer des pixels rouge d'une image!".format(
                        image) + Bcolors.RESET)
            if (image[i][j][1] > 0):
                image[i][j][0] = 1
                image[i][j][1] = 1
                image[i][j][2] = 1
                
    return(image)
    
def dim1(image):
    # passe d'une image en couleur en une matrice à une composante
    l,c,k = image.shape
    matrice = np.zeros((l, c))
    for i in range(l):
        for j in range(c):
            if (image[i][j][0] == 1 and image[i][j][1] == 0):
                matrice[i][j] = 0.5
            else:
                matrice[i][j] = image[i][j][0]
    return(matrice)

def dim3(matrice):
    # passe d'une matrice avec les valeurs 0, 0.5 et 1 à une image où les pixels à 0.5 sont rouge
    l,c = matrice.shape
    image = np.zeros((l,c,3))
    for i in range(l):
        for j in range(c):
            if (matrice[i][j] == 0.5):
                image[i][j][0] = 1
                image[i][j][1] = 0
                image[i][j][2] = 0
            else:
                for k in range(3):
                    if matrice[i][j] ==0:
                        image[i][j][k] = matrice[i][j]
                    elif matrice[i][j]==1:
                        image[i][j][k] = matrice[i][j]
                    else:
                        print(matrice[i][j],"putain")
    return(image)

def traitement_aller(image):
    image = niveauGris(image)
    matrice = dim1(image)
    return matrice

def traitement_retour(nomFic, matrice):
    image = dim3(matrice)
    Pic_Openings.sauvegardeImage(nomFic, image)

def ouverture_expert_tumeur(numImage):
    path = DATA_PATH + "coupeExpert" + tumeur + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_cnn_tumeur(numImage):
    path = DATA_PATH + "coupeCnn" + tumeur + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_ga10_tumeur(numImage):
    path = DATA_PATH + "coupeGap10" + tumeur + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_expert_rein(numImage):
    path = DATA_PATH + "coupeExpert" + rein + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_cnn_rein(numImage):
    path = DATA_PATH + "coupeCnn" + rein + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_gap10_rein(numImage):
    path = DATA_PATH + "coupeGap10" + rein + str(numImage) + ".png"
    image = Pic_Openings.ouvrirImage(path)
    return image

def ouverture_analogie(numSource, numCible, option = "tumeur"):
    #option prend tumeur par défaur mais peut aussi prendre rein
    if (option == "tumeur"):
        imageExpertSource = ouverture_expert_tumeur(numSource)
        imageCnnSource = ouverture_cnn_tumeur(numSource)
        imageCnnCible = ouverture_cnn_tumeur(numCible)
    elif (option == "rein"):
        imageExpertSource = ouverture_expert_rein(numSource)
        imageCnnSource = ouverture_cnn_rein(numSource)
        imageCnnCible = ouverture_cnn_rein(numCible)
    else :
        raise ValueError('Option incorrect : use tumeur or rein')
    matExpertSource = traitement_aller(imageExpertSource)
    matCnnSource = traitement_aller(imageCnnSource)
    matCnnCible = traitement_aller(imageCnnCible)
    return matExpertSource, matCnnSource, matCnnCible


def get_image(path):
    '''
    But: a partir d'un path on recupere l'image en matrice numpy n*m*{0,0.5,1}
    permet de reduire le nombre de ligne de code
    :param path: le path de l'image
    :return: une matrice n*m numpy
    '''
    im = Pic_Openings.ouvrirImage(path)
    #im = niveauGris(im)
    return dim1(im)


#=======================================================================================================================
#
#           FONCTION D'AFFICHAGE
#
#=======================================================================================================================


def do_fermeture_croisement(image):
    '''
    realiser la fermeture convexe d'une image 2d
    :param image:
    :return: image de meme dimension (np.array) dont la fermeture convexe a ete realise
    '''



    return image


def get_rectangle_coords_minimal(image):
    '''
    but: a partir d'une image, obtenir une liste de 4 coordonees correspondant aux 4 coins du rectangles
    minimal englobant la totalite de la forme presente dans l'image
    c'est equivalent a une fermeture convexe par rectangle
    :param image: image n*m*{0,0.5,1}
    :return: c,d,a,b,
        ou a,b dans [|0,image.shape[1]-1|] sont respectivement la premiere et derniere colonne a prendre en compte
        et c,d dans [|0,image.shape[0]-1|] sont respectivement la premiere et dernire ligne a prendre en compte
    '''
    if np.__name__ != type(image).__module__:
        # on a pas un numpy array
        raise TypeError("get_rectangle_coords_minimal Function need a numpy array")
    row,col = image.shape[0],image.shape[1]
    MIN = []
    MAX = []
    line_min,line_max = 0,0
    line_found = False
    for i in range(row):
        line = image[i,:]
        # obtenir la premiere/ derniere ligne
        if line.__contains__(1) or line.__contains__(0.5):
            if not line_found:
                line_min = i
                line_found = True
                line_max = i
            else:
                line_max = i
        # obtenir la premiere / dernire colonne
        col_min, col_max = 0,0
        col_found = False
        for j in range(col):
            if line[j] == 1 or line[j] == 0.5:
                if not col_found:
                    col_min = j
                    col_found = True
                    col_max = j
                else:
                    col_max = j
        if col_found:
            MIN.append(col_min)
            MAX.append(col_max)

    if len(MAX) == 0:
        column_max = 0
    else:
        column_max = max(MAX)


    if len(MIN) == 0:
        column_min = col
    else:
        column_min = min(MIN)

    # ajustement hauteur max/ largeur max
    if line_max < row - 1:
        line_max += 1

    if column_max < col -1:
        column_max += 1

    return line_min,line_max,column_min,column_max


def get_compressed_image_from_rectangle_coords(image,line_min=0,line_max=0,col_min=0,col_max=0):
    '''
    Aim: on a une image, et les 4 coordonees necessaire a la compresion rectangle minimale
    issue de la fonction get_rectangle_coords_minimal, on realise effectivement la compression
    on obtient donc une nouvelle matrice
    :param image: numpy.array
    :param line_min: coords dans [|0,image.shape[0] -1 |]
    :param line_max: coords dans [|0,image.shape[0] -1 |]
    :param col_min: coords dans [|0,image.shape[1] -1 |]
    :param col_max: coords dans [|0,image.shape[1] -1 |]
    :return: numpy.array de taille (line_max-line_min,col_max - col_min)

    ps: si on ne precise pas les parametres line_min, line_max,col_min,col_max
        on le calcul automatiquement
    '''
    if np.__name__ != type(image).__module__:
        # on a pas un numpy array
        raise TypeError("get_compressed_image_from_rectangle_coords function need a numpy array")
    row,col = image.shape[0],image.shape[1]

    if line_min not in range(0,row) or line_max not in range(0,row) or col_min not in range(0,col) or col_max not in range(0,col):
        if line_min > line_max or col_min > col_max:
            raise ValueError("Error in et_compressed_image_from_rectangle_coords function ! \n Erreur sur les indices "
                             "de coordones")
    if line_min == 0 and line_max == 0 and col_min == 0 and col_max == 0:
        line_min,line_max,col_min,col_max = get_rectangle_coords_minimal(image= image)

    hauteur = line_max - line_min
    largeur = col_max - col_min
    result = np.zeros((hauteur,largeur))
    for i in range(hauteur):
        for j in range(largeur):
            result[i][j] = image[i+line_min][j+col_min]
    return result


def fusion_coords_rectangle_minimal(images):
    '''
    :param images: liste d'image
    :return: line_min,line_max,col_min,col_max
    '''
    if type(images) != type([]):
        raise TypeError("fusion_rectangle_minimal function needs a list of image (numpy array)")
    if len(images) == 0:
        raise ValueError("fusion_rectangle_minimal function needs at least one image to return something")
    first = True
    row,col = 0,0
    for k in images:
        if np.__name__ != type(k).__module__:
            # on a pas un numpy array
            raise TypeError("fusion_rectangle_minimal function needs a list of numpy array")
        if first:
            row,col = k.shape[0],k.shape[1]
            first = False
        else:
            new_row,new_col = k.shape[0],k.shape[1]
            if new_row != row or new_col != col:
                raise ValueError("fusion_rectangle_minimal function needs a list of numpy array having all the same size !")

    L = [get_rectangle_coords_minimal(i) for i in images]

    return min([k[0] for k in L]),max([k[1]for k in L]),min([k[2] for k in L]),max([k[3] for k in L])


def fusion_double_image(image1,image2,have_red = False,couleur1=[i/255 for i in (223, 109, 20)],couleur2=[i/255 for i in (0, 127, 255)],couleur3=[i/255 for i in (255, 255, 0)],couleur4=[i/255 for i in (255, 0, 255)]):
    '''
    but: realiser generer une image coloree issue de la fusion de 2 segmentations pour faire ressortir les differences

    :param image1: image n*m a valeur dans {0,1} si have_red est false ou true: dans tous les cas y a pas de 0.5
    :param image2: image n*m a valeur dans {0,1} si have_red est false {0,0.5,1} sinon
    :param have_red: True or False pour specifier si y'a des 0.5
    :param couleur1: liste valeur rgb a 3 elements normalise dans [|0,1|] pour specifie la couleur1
    :param couleur2: liste valeur rgb a 3 elements normalise dans [|0,1|] pour specifie la couleur2

    :return: image n*m*3 prete a etre enregistre ou affichee
    '''
    if np.__name__ != type(image1).__module__ or np.__name__ != type(image2).__module__:
        # on a pas un numpy array
        raise TypeError("fusion_rectangle_minimal function needs a list of numpy array")
    row,col = image1.shape[0],image1.shape[1]
    if row != image2.shape[0] or col != image2.shape[1]:
        raise ValueError("fusion_double_mage function needs image of same dimension")

    if not have_red and (image1.__contains__(0.5) or image2.__contains__(0.5)):
        raise ValueError("fusion_double_mage function needs image of 0 and 1 if have_red is False")

    if have_red and image1.__contains__(0.5):
        raise ValueError("fusion_double_mage function needs image1 having 0 and 1 only even if have_red is True")

    result = np.zeros((row,col,3))
    if not have_red:
        for i in range(row):
            for j in range(col):
                if image1[i][j] == 1:
                    if image2[i][j] == 1:
                        for k in range(3):
                            result[i][j][k] = 1
                    else:
                        # im1 a 1 et im2 a 0, on passe en couleur 1
                        for k in range(3):
                            result[i][j][k] = couleur1[k]
                else:  # im 1 a 0
                    if image2[i][j] == 1:
                        # im 1 a 0 et im2 a 1, on passe en couleur 2
                        for k in range(3):
                            result[i][j][k] = couleur2[k]
                    else:
                        # im 1 et im2 a 0
                        for k in range(3):
                            result[i][j][k] = 0
    if have_red:
        for i in range(row):
            for j in range(col):
                if image2[i][j] == 1:
                    for k in range(3):
                        if image1[i][j] == 1:
                            result[i][j][k] = 1
                        else:
                            #image 1 a 0 et image 2 a 1, on passe en couleur 2
                            result[i][j][k] = couleur2[k]
                elif image2[i][j] == 0.5:
                    for k in range(3):
                        if image1[i][j]== 1:
                            #im1 = 1 et im2 = 0.5 : couleur 3
                            result[i][j][k] = couleur3[k]
                        else:
                            #im 1 = 0 et im2 = 0.5: couleur 4
                            result[i][j][k] = couleur4[k]
                else:
                    for k in range(3):
                        if image1[i][j]== 1:
                            #cas ou im1 = 1 et im2 = 0
                            result[i][j][k] = couleur1[k]
                        else:
                            #im1 et im2 = 0
                            result[i][j][k] = 0
    return result


def plot_a_b_c_d(imA,imB,imC,imD,imCenter = None):
    '''
    But: obtenir le graphique 3x3 representant les differences A,B,C,D du raisonnement par analogie

        orange/bleu
        et jaune/violet

        ie: A   A/B     B
            A/C         B/D
            C   C/D     D


        convention couleur, plusieurs cas de figure

                        A/B     A/C      C/D     B/D
        blanc blanc    blanc    blanc   blanc   blanc
        blanc noir
        noir blanc
        noir noir      noir     noir    noir    noir

        # uniquement B:D et C:D car le rouge sera uniquement chez D
        blanc rouge
        noir rouge


    :param imA: numpy array n*m*{0,1}
    :param imB: numpy array n*m*{0,1}
    :param imC: numpy array n*m*{0,1}
    :param imD: numpy array n*m*{0,0.5,1}
    :return: None
    '''

    line_min, line_max, col_min, col_max = fusion_coords_rectangle_minimal(images=[imA, imB, imC, imD])
    print(line_min,line_max,col_min,col_max)
    A = get_compressed_image_from_rectangle_coords(imA,line_min=line_min,line_max=line_max,col_min=col_min,col_max=col_max)
    B = get_compressed_image_from_rectangle_coords(imB,line_min=line_min,line_max=line_max,col_min=col_min,col_max=col_max)
    C = get_compressed_image_from_rectangle_coords(imC,line_min=line_min,line_max=line_max,col_min=col_min,col_max=col_max)
    D = get_compressed_image_from_rectangle_coords(imD,line_min=line_min,line_max=line_max,col_min=col_min,col_max=col_max)


    #faut preparer l'enesemble des 8 images en respectant les conventions

        # premiere etape: generer a:b et a:c qui se font de maniere analogues
    A_B = fusion_double_image(image1=A,image2=B)
    A_C = fusion_double_image(image1=A,image2=C)
        # deuxieme etape: generer c:d et b:d qui se font egualement de maniere analogue
    C_D = fusion_double_image(image1=C,image2=D,have_red=True)
    B_D = fusion_double_image(image1=B,image2=D,have_red=True)


    fig = plt.figure(figsize=(8,8))
    columns = 3
    rows = 3
    for i in range(1,columns*rows+1):
        fig.add_subplot(rows,columns,i)
        if i == 1:
            img = dim3(A)
        elif i == 2:
            img = A_B
        elif i == 3:
            img = dim3(B)
        elif i == 4:
            img = A_C
        elif i == 5:
            if imCenter != None:
                img = imCenter
            else:
                img = np.zeros(A.shape)
        elif i == 6:
            img = B_D
        elif i == 7:
            img = dim3(C)
        elif i == 8:
            img = C_D
        else:
            img = dim3(D)
        plt.imshow(img)
    plt.show()
    return 1