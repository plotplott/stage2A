La racine du projet initiale est le dossier "./stage2A/stage/".
Les fonctions utiles sont écrites (en partant de la racine) dans "./src/segmentationUtils":
    - Pic_Openings.py contient deux fonctions:
      ° Une fonction pour ouvrir une image à partir d'un fichier ".png" et donne une matrice numpy array n*m*3 (3 composantes rouge vert bleu)
      ° Une fonction pour sauvegarder une matrice numpy array n*m*3 en un fichier ".png".
    - Pic_Treatments.py contient plusieurs types de fonctions:
      ° Des fonctions qui permettent la transition entre une matrice n*m*3 a une matrice n*m*1 composée de 0/0.5/1.
      
      Méthode pour ouvrir une image issue de la base de données(images avec du vert ou brun dedans):
          => Pic_openings.ouvrirImage
          => Pic_Treatments.niveauGris
          => Pic_Treatments.dim1
       Méthode pour ouvrir une image issue d'un traitement au préalable (images en noir/blanc/rouge):
          => Pic_openings.ouvrirImage
          => Pic_Treatments.dim1
       Méthode pour sauvegarder une matrice n*m*{0,0.5,1} en un fichier ".png"
          => Pic_Treatments.dim3
          => Pic_Openings.sauvegardeImage
    
   ° Des fonctions supports qui permettent de realiser de manipuler des images n*m*{0,0.5,1} pour realsier une méthode explorée durant le stage.
   En gros, elles ne sont pas utilisable directement, sauf exceptions.
   Exceptions:
        - Pic_Treatments.plot_a_b_c_d qui permet une représentation des images a,b,c,d (selon le raisonnement par analogie) en matrice 3x3:
            1.1 a               1.2 fusion a:b      1.3b
            2.1 fusion a:c                         2.3 fusion b:d
            3.1 c               3.2 fusion c:d      3.3 d
   
   - Frame_Performances.py contient des fonctions permettant nottament de calculer les indices de Dice et Jaccard:
        ° Frame_Performances.indice_dice est la fonction qu'on utilisera le plus pour obtenir l'indice de Dice entre 2 images sous formes numpy array n*m*{0,0.5,1}
   
