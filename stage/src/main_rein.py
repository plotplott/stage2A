import src.segmentationUtils.Pic_Openings as Pic_Openings
import src.segmentationUtils.Pic_Treatments as Pic_Treatments
import src.segmentationUtils.Analogy_Computing as Analogy_Computing
import src.segmentationUtils.Frame_Performances as Frame_Performances
import os
import matplotlib.pyplot as plt
import time

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
rein = "/rein/"
path_to_store = "../Data/rein/"

def do_comparaison_gap10_analog_calcul(list_source=[]):
    path_to_store = "../Data/rein/"
    if not os.path.exists(path=path_to_store):
        raise ValueError("You Idiot: Directory Not Found")

    list_source = [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]
    borne = 5
    min = 1
    max = 136

    start = time.time()

    for k in range(len(list_source)):
        for i in range(1,borne+1):
            if list_source[k]-i >= min:
                # creation des path
                num_apprentissage = list_source[k]
                num_deduction= list_source[k]-i
                expertSource = DATA_PATH + "coupeExpert" + rein + str(num_apprentissage) + ".png"
                cnnSource = DATA_PATH + "coupeCnn" + rein + str(num_apprentissage) + ".png"
                cnnCible = DATA_PATH + "coupeCnn" + rein + str(num_deduction) + ".png"
                analogyPath =path_to_store + str(num_deduction) + "from" + str(num_apprentissage) + ".png"

                # chargement en memoire des images
                imageExpertSource = Pic_Openings.ouvrirImage(expertSource)
                imageCnnSource = Pic_Openings.ouvrirImage(cnnSource)
                imageCnnCible = Pic_Openings.ouvrirImage(cnnCible)

                # conversion a une dimension pour les calculs
                matExpertSource = Pic_Treatments.dim1(imageExpertSource)
                matCnnSource = Pic_Treatments.dim1(imageCnnSource)
                matCnnCible = Pic_Treatments.dim1(imageCnnCible)

                # creer la nouvelle segmentation
                matAnalogy = Analogy_Computing.cnnTab_to_anaTab_(matExpertSource, matCnnSource, matCnnCible)
                imageAnalogy = Pic_Treatments.dim3(matAnalogy)
                Pic_Openings.sauvegardeImage(analogyPath, imageAnalogy)

            if list_source[k]+i <= max:
                # creation des path
                num_apprentissage = list_source[k]
                num_deduction = list_source[k] + i
                expertSource = DATA_PATH + "coupeExpert" + rein + str(num_apprentissage) + ".png"
                cnnSource = DATA_PATH + "coupeCnn" + rein + str(num_apprentissage) + ".png"
                cnnCible = DATA_PATH + "coupeCnn" + rein + str(num_deduction) + ".png"
                analogyPath = path_to_store + str(num_deduction) + "from" + str(num_apprentissage) + ".png"

                # chargement en memoire des images
                imageExpertSource = Pic_Openings.ouvrirImage(expertSource)
                imageCnnSource = Pic_Openings.ouvrirImage(cnnSource)
                imageCnnCible = Pic_Openings.ouvrirImage(cnnCible)

                # conversion a une dimension pour les calculs
                matExpertSource = Pic_Treatments.dim1(imageExpertSource)
                matCnnSource = Pic_Treatments.dim1(imageCnnSource)
                matCnnCible = Pic_Treatments.dim1(imageCnnCible)

                # creer la nouvelle segmentation
                matAnalogy = Analogy_Computing.cnnTab_to_anaTab_(matExpertSource, matCnnSource, matCnnCible)
                imageAnalogy = Pic_Treatments.dim3(matAnalogy)
                Pic_Openings.sauvegardeImage(analogyPath, imageAnalogy)

    end = time.time()
    print("frame calculus took:",end-start,"sec")
    return 1


def do_comparaison_gap10_analog_print(list_source=[]):
    path_to_store = "../Data/rein/"
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
        expertSource = DATA_PATH + "coupeExpert" + rein + str(current_slice) + ".png"
        imageExpertSource = Pic_Openings.ouvrirImage(expertSource)
        matExpertSource = Pic_Treatments.dim1(imageExpertSource)

        #gap10 = concurrent
        gap10Source = DATA_PATH + "coupeGap10" + rein + str(current_slice) + ".png"
        imageGap10Source = Pic_Openings.ouvrirImage(expertSource)
        matGap10Source = Pic_Treatments.dim1(imageExpertSource)

        #analog = challenger
        imageAnalSource = Pic_Openings.ouvrirImage(path_to_store + dir)
        matAnalSource = Pic_Treatments.dim1(imageExpertSource)

        dice_expert_gap10= Frame_Performances.indice_dice(matExpertSource, matGap10Source)
        dice_expert_anal = Frame_Performances.indice_dice(matExpertSource, matAnalSource)
        jac_expert_gap10= Frame_Performances.indice_jaccard(matExpertSource, matGap10Source)
        jac_expert_anal= Frame_Performances.indice_jaccard(matExpertSource, matAnalSource)

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
    #plt.plot(current_nums,list_dice_gap10,'o',label="gap10")
    plt.xlabel=("Dice")
    plt.xlim(75,110)
    plt.legend()
    plt.show()

    plt.clf()
    plt.title = "Jacc Indicator"
    for k in [2, 13, 24, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134]:
        plt.axvline(x=k)
    plt.plot(current_nums, list_jacc_anal, 'x', label="analogie")
    #plt.plot(current_nums, list_jacc_gap10, 'o', label="gap10")
    plt.xlim(75,110)
    plt.xlabel="Jacc"
    plt.legend()

    plt.show()

    return 1


#do_comparaison_gap10_analog_calcul()
do_comparaison_gap10_analog_print()
