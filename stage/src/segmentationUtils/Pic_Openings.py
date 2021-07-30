import matplotlib.image as mpimg
import os

def ouvrirImage(nomFic):
    # ouvrir une image à partir du chemin déjà prédéfini
    return (mpimg.imread(nomFic))
    
def sauvegardeImage(nomFic, image):
    # sauvegarder une image vers le chemin d'accès déjà prédéfini
    mpimg.imsave(nomFic, image)

def if_not_exist(PATH):
    '''
    But, verifier l'existance d'un path et le creer s'il n'existe pas
    :param PATH: chemin ex "../Data/basic/..
    :return: rien
    '''
    if not os.path.exists(PATH):
        os.makedirs(PATH)
