import src.segmentationUtils.Frame_Performances as Frame_Performances
import matplotlib.pyplot as plt

NUM = 95

num_app,list_dice,list_jacc = Frame_Performances.get_indicator_from_bd(num=NUM)


plt.clf()
plt.title("Indice Dice et Jacc en fonction du numero d'apprentissage sur l'image n° {}".format(str(NUM)))
plt.axvline(x=NUM)
plt.xlabel("Indice Dice")
plt.plot(num_app,list_dice,'o',label='dice')
plt.plot(num_app,list_jacc,'o',label='jacc')
plt.legend()
plt.show()