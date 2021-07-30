import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Segmentation as Segmentation


# print in color: on a acces a la classe bicolors suite a l'importation de la classe Segmentation

class Segmentation_Calculator:
    """
    Cette Classe permet de generer la segmentation x issue du raisonnement par analogie a:b::c:x

    a,b sont des objets de type segmentation
    c peut etre un objet de type segmentation ou une liste d'objets de type segmentation, il faudra faire les traitements en consequence
    """
# besoin d'ajouter une fct add qui ajoute une segmentation à la liste des segmentations cibles ou qui créée une liste de 2 segmentations ?

    def __init__(self, expertSource, cnnSource, cnnCible):
        assert isinstance(expertSource,Segmentation.Segmentation),Segmentation.Bcolors.WARNING+"expertSource doit etre une instance de la classe Segmentation"+Segmentation.Bcolors.RESET
        self.setExpertSource(expertSource)
        assert isinstance(cnnSource,Segmentation.Segmentation),Segmentation.Bcolors.WARNING+"cnnSource doit etre une instance de la classe Segmentation"+Segmentation.Bcolors.RESET
        self.setCnnSource(cnnSource)

        if isinstance(cnnCible,Segmentation.Segmentation):
            self.setCnnCible(cnnCible)
        else:
            msg = Segmentation.Bcolors.WARNING+"cnnCible dans la classe Segmentation_Calculator doit etre une liste d'instance de Segmentation"+Segmentation.Bcolors.RESET
            if type(cnnCible) != type([]):
                raise TypeError(msg)
            for k in cnnCible:
                if not isinstance(k,Segmentation.Segmentation):
                    raise TypeError(msg)
            self.setCnnCible(cnnCible)


    # --------------------- SETTERS --------------------------

    def setExpertSource(self, expertSource):
        assert isinstance(expertSource, Segmentation.Segmentation), "Votre image expert source n'est pas une instance de la classe Segmentation"
        self.expertSource = expertSource
        pass

    def setCnnSource(self, cnnSource):
        assert isinstance(cnnSource,Segmentation.Segmentation), "Votre image cnn source n'est pas une instance de la classe Segmentation"
        self.cnnSource = cnnSource
        pass

    def setCnnCible(self, cnnCible):
        if isinstance(cnnCible,Segmentation.Segmentation):
            self.cnnCible = cnnCible
        else:
            msg = Segmentation.Bcolors.WARNING+"cnnCible dans la classe Segmentation_Calculator doit etre une liste d'instance de Segmentation"+Segmentation.Bcolors.RESET
            if type(cnnCible) != type([]):
                raise TypeError(msg)
            for k in cnnCible:
                if not isinstance(k,Segmentation.Segmentation):
                    raise TypeError(msg)
            self.cnnCible = cnnCible
        pass

    # --------------------- GETTERS --------------------------

    def getExpertSource(self):
        return self.expertSource

    def getCnnSource(self):
        return self.cnnSource

    def getCnnCible(self):
        return self.cnnCible

    # --------------------- AUTRES METHODES --------------------------

    def calcul(self, method_type="None"):
        """ici il faut faire attention a 2 choses
            le type de C: si c'est une liste, on renvoie une liste d'images
            le method_type corrrespond au method type present dans Analogy_computing.cnnTab_to_anaTab
        """
        1

