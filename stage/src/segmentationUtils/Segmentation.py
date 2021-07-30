import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import os


class Bcolors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


class Segmentation:
    """
    Cette classe permet la gestion des images de segmentation.
    Grace a cette classe, l'utilisateur ne manipulera que des liens d'image et des np.arrays
    """

    def __init__(self, link=None, repertoire=None, nomFichier=None,num = 1):
        # Attention ! si on donne le liens, il faut separer repertoire et nomFichier comme il faut
        # Attention ! si on donne repertoire + nomFichier, il faut reconstituer le liens comme il se faut

        self.numero = num
        self.matrice = None
        self.liens = None
        self.rep = None
        self.nomFichier = nomFichier

        if link != None:
            self.setLiens(link)

            if link.__contains__("/"):
                # il faut spliter
                self.rep = link.replace(link.split("/")[-1], "")
                self.nomFichier = link.split("/")[-1]
            else:
                self.rep = "./"
                self.nomFichier = link


        if repertoire != None:
            self.setRepertoire(repertoire)
        if nomFichier != None:
            self.setNomFichier(nomFichier)

    # --------------------- SETTERS --------------------------

    def setNumero(self,num):
        self.numero = num

    def setLiens(self, liens):
        # Attention ! si on donne le liens, il faut separer repertoire et nomFichier comme il faut
        if not os.path.exists(liens.replace(liens.split("/")[-1], "")):
            raise ValueError(Bcolors.WARNING + "Erreur! Le fichier {} n'a pas pu etre ouvert.\nMise a jour impossible".format(
                liens) + Bcolors.RESET)
        self.liens = liens

    def setRepertoire(self, repertoire):
        if not os.path.isdir(repertoire):
            raise ValueError(Bcolors.WARNING + "Erreur! Le repertoire {} n' existe pas.\nMise a jour impossible".format(
                repertoire) + Bcolors.RESET)
        if repertoire[-1] != "/":
            repertoire = repertoire + "/"
        self.rep = repertoire
        self.liens = self.rep + self.nomFichier

    def setNomFichier(self, fileName):
        self.nomFichier = fileName
        self.liens = self.rep + self.nomFichier

    def loadMatrice(self):
        # importer l'image en noir et blanc (et rouge) np.array m.n.{0,0.5,1} depuis le liens
        assert os.path.exists(self.liens), "Le lien vers le repertoire n'existe pas"
        image = Pic_Openings.ouvrirImage(self.liens)
        image = Pic_Treatments.niveauGris(image)
        matrice = Pic_Treatments.dim1(image)
        self.matrice = matrice

    def setMatrice(self,mat):
        # attribuer un array a l'attribut matrice, cet array a ete calcule au prealable
        self.matrice = mat


    # --------------------- GETTERS --------------------------

    def getNumero(self):
        return self.numero

    def getLiens(self):
        return self.liens

    def getRepertoire(self):
        return self.rep

    def getNomFichier(self):
        return self.nomFichier

    def getMatrice(self):
        return self.matrice

    # --------------------- AUTRES METHODES --------------------------

    def saveIt(self):
        # enregistrer l'image sur le liens en noir et blanc et rouge
        assert os.path.exists(self.rep), "Le lien vers le repertoire n'existe pas"
        # faire un rafraichissement
        print(type(self.matrice),self.matrice.shape)
        image = Pic_Treatments.dim3(self.matrice)
        Pic_Openings.sauvegardeImage(self.liens, image)
        pass

    def getSize(self):
        # renvoie la taille de l'image
        assert len(self.matrice) > 0, "Aucune matrice n'a ete ajoutee"
        return self.matrice.shape
