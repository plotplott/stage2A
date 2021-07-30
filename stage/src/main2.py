import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import os
import matplotlib.pyplot as plt
import time

'''
Frame_Performances.get_indicator_from_bd(num=95)

start = time.time()
num,dice,jacc = Frame_Performances.guess_pivot()
end = time.time()

print("calculus took : ",end-start)

plt.clf()
plt.legend()
plt.title()
plt.plot(num,dice,'o',label="dice")
plt.plot(num,jacc,'o',label="jacc")
plt.show()
'''



'''but : qu'est ce que je veux ? 
    -> comparer CNN / CNN + Gap10 / CNN + Analogie
    idee on selectionne 1 images sur 10
    entre ces pivots on regarde : 
        on regarde CNN + analogie autour 
        on regarde CNN + Gap 10 
    puis on compare les deux 

    Qu'est ce qu'il nous faut ?
        - les indices des images sources, d'apres la reu : [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
        - les images sorties par gap 10 entre ces images sources
        - les images sorties par analogie entre ces images sources
'''

DATA_PATH = "../Data/"
tumeur = "/tumeur/"
path_to_store = "../Data/stupid/"

def do_comparaison_gap10_analog_calcul(list_pivot=[]):
    path_to_store = "../Data/stupid/"
    if not os.path.exists(path=path_to_store):
        raise ValueError("You Idiot: Directory Not Found")

    list_pivot = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
    borne = 5
    min = 1
    max = 136

    start = time.time()

    for k in range(len(list_pivot)):
        for i in range(1,borne+1):
            if list_pivot[k]-i >= min:
                # creation des path
                num_apprentissage = list_pivot[k]
                num_deduction= list_pivot[k]-i
                expertOtherPath = DATA_PATH + "coupeExpert" + tumeur + str(num_apprentissage) + ".png"
                cnnOtherPath = DATA_PATH + "coupeCnn" + tumeur + str(num_apprentissage) + ".png"
                cnnCurrentPath = DATA_PATH + "coupeCnn" + tumeur + str(num_deduction) + ".png"
                analogyPath =path_to_store + str(num_deduction) + "from" + str(num_apprentissage) + ".png"

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

            if list_pivot[k]+i <= max:
                # creation des path
                num_apprentissage = list_pivot[k]
                num_deduction = list_pivot[k] + i
                expertOtherPath = DATA_PATH + "coupeExpert" + tumeur + str(num_apprentissage) + ".png"
                cnnOtherPath = DATA_PATH + "coupeCnn" + tumeur + str(num_apprentissage) + ".png"
                cnnCurrentPath = DATA_PATH + "coupeCnn" + tumeur + str(num_deduction) + ".png"
                analogyPath = path_to_store + str(num_deduction) + "from" + str(num_apprentissage) + ".png"

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

    end = time.time()
    print("frame calculus took:",end-start,"sec")
    return 1


def do_comparaison_gap10_analog_print(list_pivot=[]):
    path_to_store = "../Data/stupid/"
    if not os.path.exists(path_to_store):
        raise ValueError("You Idiot: Directory Not Found")

    dir_list = os.listdir(path_to_store)

    # 6from7.png num_app = 7, current_num = 6

    current_nums = []
    list_dice_anal = []
    list_jacc_anal = []
    list_dice_gap10 = []
    list_jacc_gap10 = []

    start = time.time()
    for dir in dir_list:
        print("dir:",dir)
        num_app = dir.split("from")[1].split(".")[0]
        current_slice = dir.split("from")[0]

        current_nums.append(current_slice)

        #expert = reference
        expertCurrentPath = DATA_PATH + "coupeExpert" + tumeur + str(current_slice) + ".png"
        imageExpertCurrent = Pic_Openings.ouvrirImage(expertCurrentPath)
        matExpertCurrent = Pic_Treatments.dim1(imageExpertCurrent)

        #gap10 = concurrent
        gap10CurrentPath = DATA_PATH + "coupeGap10" + tumeur + str(current_slice) + ".png"
        imageGap10Current = Pic_Openings.ouvrirImage(expertCurrentPath)
        matGap10Current = Pic_Treatments.dim1(imageExpertCurrent)

        #analog = challenger
        imageAnalCurrent = Pic_Openings.ouvrirImage(path_to_store + dir)
        matAnalCurrent = Pic_Treatments.dim1(imageExpertCurrent)

        dice_expert_gap10= Frame_Performances.indice_dice(matExpertCurrent, matGap10Current)
        dice_expert_anal = Frame_Performances.indice_dice(matExpertCurrent, matAnalCurrent)
        jac_expert_gap10= Frame_Performances.indice_jaccard(matExpertCurrent, matGap10Current)
        jac_expert_anal= Frame_Performances.indice_jaccard(matExpertCurrent, matGap10Current)

        list_dice_anal.append(dice_expert_anal)
        list_jacc_anal.append(jac_expert_anal)

        list_dice_gap10.append(dice_expert_gap10)
        list_jacc_gap10.append(jac_expert_gap10)
    end = time.time()
    print("indice calculus took",end-start,"sec")


    plt.clf()

    plt.title="Dice Indicator"
    for k in [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]:
        plt.axvline(x=k)
    plt.plot(current_nums,list_dice_anal,'o',label="analogie")
    plt.plot(current_nums,list_dice_gap10,'o',label="gap10")
    plt.xlabel=("Dice")
    plt.xlim(75,110)
    plt.legend()
    plt.show()

    plt.clf()
    plt.title = "Jacc Indicator"
    for k in [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]:
        plt.axvline(x=k)
    plt.plot(current_nums, list_jacc_anal, 'o', label="analogie")
    plt.plot(current_nums, list_jacc_gap10, 'o', label="gap10")
    plt.xlim(75,110)
    plt.xlabel="Jacc"
    plt.legend()

    plt.show()

    return 1


#do_comparaison_gap10_analog_calcul()
do_comparaison_gap10_analog_print()