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



# Analyse exploratoire 
## Analyse univariée

Nous passons à présent à l'analyse exploratoire, nous allons éxaminer de plus près les caractéristiques distinctives de notre collection de vins à travers une analyse univariée, révélant les tendances et les nuances de leur composition et de leur qualité.


### variables dépendantes




![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/c5827219-fb71-4ef8-988c-6ff1a03fd62d)
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/533cd6d5-08a9-48db-89bc-f08ef4b98d22)


Le graphique ci dessus nous montre la répartition de deux types de vins. On observe une répartition assez inégale, en effet, le vin blanc représente une part beaucoup plus importante, avec 75.39%, tandis que le vin rouge est moins fréquent à 24.61%.



![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/a380df43-6cf3-4f8c-ac81-dcd54569aaa0)

Pour ce qui est de la variable qualité, nous avons préféré un histogramme en barre pour analyser la répartition. Les histogrammes en barres offrent une comparaison et une interprétation plus claires des fréquences lorsque l’on analyse des variables catégorielles. 
Ainsi, nous observons une distribution asymétrique où la majorité des vins se concentrent autour des notes de qualité moyenne correspondant à une loi normale, ce qui est logique  puisque la plupart des vins reçoivent des notes moyennes et moins reçoivent des notes extrêmes.

De ce fait, nous devrons procéder à un rééquilibrage des classes afin d’avoir des modèles pertinents  

### variables explicatives

|       | Fixed Acidity | Volatile Acidity | Citric Acid | Residual Sugar | Chlorides | Free Sulfur Dioxide | Total Sulfur Dioxide | Density  | pH    | Sulphates | Alcohol | Quality |
|-------|---------------|------------------|-------------|----------------|-----------|---------------------|----------------------|----------|-------|-----------|---------|---------|
| Count | 6487.000000   | 6489.000000      | 6494.000000 | 6495.000000    | 6495.000000 | 6497.000000         | 6497.000000          | 6497.000000 | 6488.000000 | 6493.000000 | 6497.000000 | 6497.000000 |
| Mean  | 7.216579      | 0.339691         | 0.318722    | 5.444326       | 0.056042  | 30.525319           | 115.744574           | 0.994697  | 3.218395  | 0.531215  | 10.491801 | 5.818378 |
| Std   | 1.296750      | 0.164649         | 0.145265    | 4.758125       | 0.035036  | 17.749400           | 56.521855            | 0.002999  | 0.160748  | 0.148814  | 1.192712  | 0.873255 |
| Min   | 3.800000      | 0.080000         | 0.000000    | 0.600000       | 0.009000  | 1.000000            | 6.000000             | 0.987110  | 2.720000  | 0.220000  | 8.000000  | 3.000000 |
| 25%   | 6.400000      | 0.230000         | 0.250000    | 1.800000       | 0.038000  | 17.000000           | 77.000000            | 0.992340  | 3.110000  | 0.430000  | 9.500000  | 5.000000 |
| 50%   | 7.000000      | 0.290000         | 0.310000    | 3.000000       | 0.047000  | 29.000000           | 118.000000           | 0.994890  | 3.210000  | 0.510000  | 10.300000 | 6.000000 |
| 75%   | 7.700000      | 0.400000         | 0.390000    | 8.100000       | 0.065000  | 41.000000           | 156.000000           | 0.996990  | 3.320000  | 0.600000  | 11.300000 | 6.000000 |
| Max   | 15.900000     | 1.580000         | 1.660000    | 65.800000      | 0.611000  | 289.000000          | 440.000000           | 1.038980  | 4.010000  | 2.000000  | 14.900000 | 9.000000 |










