# Projet SVM et Réseaux de neurones : prédiction qualité et type de vin


![oenologie-quelles-differences-entre-un-vin-blanc-et-un-vin-rouge-istock-com-piranka-209-1537370769](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/117682143/af589b39-1ba7-4e9b-95fb-86bda8b86ee1)

# Introducion

Dans le cadre de notre projet de Machine Learning et de Deep Learning, nous nous sommes lancés dans l'analyse approfondie d'une base de données portant sur la qualité et le type de vins. Cette base, récupérée sur le site de référence Kaggle et créé par un auteur du nom de Raj Parmar, inclut des informations détaillées sur un vaste éventail de vins, rouges comme blancs.
Notre objectif est double : prédire la qualité du vin, évaluée sur une échelle de 0 à 10, et classifier le vin en tant que rouge ou blanc. Cette tâche s'appuie sur l'exploitation de divers paramètres chimiques et physiques des vins, tels que l'acidité, le sucre résiduel, les chlorures, le sulfite, la densité, le pH, les sulfates, et la teneur en alcool. 
La première phase de notre projet a consisté en une compréhension approfondie de la base de données, suivie d'un nettoyage rigoureux et d'une préparation des données, incluant la gestion des valeurs manquantes et la création d'indicateurs pertinents. Cette étape essentielle a permis d'établir des fondations solides pour nos analyses et modélisations ultérieures.
Dans un deuxième temps, nous avons exploré les distributions et les corrélations des variables explicatives avec la qualité et le type de vin. Cette analyse exploratoire est cruciale pour formuler des hypothèses sur les facteurs ayant une influence sur ces derniers. Finalement, nous avons développé et testé une série de modèles de prédiction visant à évaluer la qualité et à déterminer le type de vin. 

# Description du dataset
Nous avons exploité une base de données disponible sur le site Kaggle, spécialisée dans les caractéristiques et la qualité des vins rouges et blancs. Cette base de données, riche en informations, nous permet d'analyser en profondeur les différents aspects influençant la qualité du vin. La base de données initiale comprend des enregistrements sur un grand nombre de vins, couvrant diverses variables chimiques et physiques.
## 1. Description des variables



| Caractéristique       | Description |
|-----------------------|-------------|
| Type                  | Indique si le vin est rouge ou blanc. |
| Fixed Acidity         | L'acidité fixe du vin. |
| Volatile Acidity      | L'acidité volatile, influençant l'arôme et le goût du vin. |
| Citric Acid           | Le niveau d'acide citrique, un facteur important dans la saveur du vin. |
| Residual Sugar        | La quantité de sucre résiduel après la fin de la fermentation. |
| Chlorides             | La concentration en chlorures, affectant le goût salé du vin. |
| Free Sulfur Dioxide   | Quantité de dioxyde de soufre libre, jouant un rôle dans la prévention de l'oxydation et la croissance de microbes. |
| Total Sulfur Dioxide  | Quantité totale de dioxyde de soufre, un indicateur important pour la conservation du vin. |
| Density               | La densité du vin, liée à son taux d'alcool et de sucre. |
| pH                    | Mesure de l'acidité ou de la basicité du vin. |
| Sulphates             | Niveau de sulfates, influençant la fermentation et le goût du vin. |
| Alcohol               | Le pourcentage d'alcool dans le vin. |
| Quality               | La note de qualité attribuée au vin, sur une échelle de 0 à 10. |
