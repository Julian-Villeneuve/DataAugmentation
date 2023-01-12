# DataAugmentation
Projet Traitement du Signal et Applications en Imagerie, sujet 6 : L'augmentation de données pour la classification d'images satellitaires.

L'objectif de ce projet est l'étude des techniques d’augmentation de données pour la classification
des séries temporelles d’images satellitaires. En apprentissage automatique, l'augmentation de
données est une technique utilisée pour augmenter artificiellement la quantité de données
étiquetées. Elle consiste à ajouter des copies légèrement modifiées de données déjà existantes
ou à générer des données synthétiques. Alors que les techniques d’augmentation de données
pour les images ont été largement étudiées dans la littérature, elles l’ont moins été pour les séries
temporelles d'images satellitaires.

Pour ce projet nous nous sommes servis du projet PASTIS (https://github.com/VSainteuf/pastis-benchmark)
et Utae-PAPS (https://github.com/VSainteuf/utae-paps)

Deux fichiers principaux sont présents:
- TdS_Results qui sert à effectuer des tests sur les réseaux
- TdS_train qui sert à entrainer les réseaux avec ou sans augmentations

dans le dossier results se trouvent deux modèles déjà entrainés, un dans augs qui a été entrainé avec augmentations de données, et un dans no_augs qui a été entrainé sans augmentations de données.
