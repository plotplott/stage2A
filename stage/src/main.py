import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import os
import time
'''
Convention pour nommer les nouveaux fichiers dans le repertoire "./Data/coupeAnal/tumeur/" :
    
    Pour chaque coupe allant de 1 à 136
    - on cree un repertoire portant comme nom le numero de la coupe
        Dans ce dossier :
            - on calcule l'ensemble des 135 images obtenus à partir du raisonnement par analogie (on ne peut pas le faire avec elle même)
            - on enregistre l'image obtenue au nom de "[n° image segmente (identique au n° du dossier)]from[n° image d'apprentissage].png"
            
        ex : /95/95 from 1.png ..
                95from87.png ..
                95 from 136.png
    
'''

Analogy_Computing.red_calculation(method_type="classic",sous_repertoire="c")

N= 136
coupe_app = 75
coupe_deduite = 95
rein = "/rein/"
tumeur = "/tumeur/"
DATA_PATH = "../Data/"

'''But : generer une image issue du raisonement par analogie 
    parametre d'entree : 
        - le numero d'apprentissage
        - le numero a deduire
        
    on ne renvoie rien mais ca cree une image la ou il faut 
'''
def generate_pic(num_apprentissage = 75,num_deduction = 95):

    rein = "/rein/"
    tumeur = "/tumeur/"
    DATA_PATH = "../Data/"
    saving_dir = str(num_deduction)+"/"

    #on verifie que le repertoire existe
    repertoire = DATA_PATH + "coupeAnal" +tumeur+saving_dir
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)

    # creation des path
    expertOtherPath = DATA_PATH + "coupeExpert" + tumeur + str(num_apprentissage) + ".png"
    cnnOtherPath = DATA_PATH + "coupeCnn" + tumeur + str(num_apprentissage) + ".png"
    cnnCurrentPath = DATA_PATH + "coupeCnn" + tumeur + str(num_deduction) + ".png"
    analogyPath = DATA_PATH + "coupeAnal" + tumeur + saving_dir + str(num_deduction)+"from"+str(num_apprentissage) + ".png"
    #print(expertOtherPath, cnnOtherPath, cnnCurrentPath,analogyPath)

    # chargement en memoire des images
    imageExpertOther = Pic_Openings.ouvrirImage(expertOtherPath)
    imageCnnOther = Pic_Openings.ouvrirImage(cnnOtherPath)
    imageCnnCurrent = Pic_Openings.ouvrirImage(cnnCurrentPath)

    # conversion a une dimension pour les calculs
    matExpertOther = Pic_Treatments.dim1(imageExpertOther)
    matCnnOther = Pic_Treatments.dim1(imageCnnOther)
    matCnnCurrent = Pic_Treatments.dim1(imageCnnCurrent)

    # creer la nouvelle segmentation
    matAnalogy = Analogy_Computing.cnnTab_to_anaTab_(matExpertOther, matCnnOther, matCnnCurrent)
    imageAnalogy = Pic_Treatments.dim3(matAnalogy)
    Pic_Openings.sauvegardeImage(analogyPath, imageAnalogy)

    return 1


'''But: generer l'ensemble de toutes les possibilites possibles'''
def generate_all_pic(number_of_pic=136):
    for k in range(1,number_of_pic+1):
        start = time.time()
        for l in range(1,number_of_pic+1):

            if k != l:
                if k > 72:

                    generate_pic(l,k)
        end = time.time()
        delay = end-start
        print("num:",k,"processed in: ",delay,"sec")

    return 1

#generate_all_pic()


#indices = get_indice_from_1frame("../Data/coupeAnal/tumeur/95/")
#ploting(indices[0],indices[1],95)

