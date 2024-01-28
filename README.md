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

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/533cd6d5-08a9-48db-89bc-f08ef4b98d22)


Le graphique ci dessus nous montre la répartition de deux types de vins. On observe une répartition assez inégale, en effet, le vin blanc représente une part beaucoup plus importante, avec 75.39%, tandis que le vin rouge est moins fréquent à 24.61%.


![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/10322cb0-422c-4ab6-ab05-944b64c8f84c)


Pour ce qui est de la variable qualité, nous avons préféré un histogramme en barre pour analyser la répartition. Les histogrammes en barres offrent une comparaison et une interprétation plus claires des fréquences lorsque l’on analyse des variables catégorielles. 
Ainsi, nous observons une distribution asymétrique où la majorité des vins se concentrent autour des notes de qualité moyenne correspondant à une loi normale, ce qui est logique  puisque la plupart des vins reçoivent des notes moyennes et moins reçoivent des notes extrêmes.

De ce fait, nous devrons procéder à un rééquilibrage des classes afin d’avoir des modèles pertinents  

### variables explicatives
<div align="center">
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


Dans notre étude de plus de 6400 vins, nous avons remarqué quelques tendances intéressantes concernant leurs caractéristiques. Certains aspects des vins sont assez similaires d'une bouteille à l'autre, tandis que d'autres varient beaucoup.
Premièrement, il y a des éléments comme la densité et le pH où la plupart des vins sont assez semblables. La densité des vins ne change pas beaucoup, ce qui signifie que la "lourdeur" ou la "légèreté" du vin en termes de poids est presque la même pour tous. Le pH est également assez constant, indiquant que l'équilibre acide-basique ne varie pas trop d'un vin à l'autre.
En revanche, le sucre résiduel, qui est le sucre restant après la fermentation, et le dioxyde de soufre, utilisé pour conserver le vin, montrent beaucoup plus de différences entre les vins. Certains vins sont beaucoup plus sucrés que d'autres, et la quantité de dioxyde de soufre varie également beaucoup. Cela nous donne une idée de la diversité des goûts et des méthodes de fabrication des vins.
D'autres caractéristiques comme l'acidité, le niveau de certains acides (comme l'acide citrique), les chlorures (qui influencent le goût salé), les sulfates (utilisés aussi pour la conservation) et l'alcool ont des valeurs plus équilibrées. 
Pour conclure, notre étude montre qu'il y a beaucoup de similitudes dans certains aspects des vins, mais aussi une grande variété dans d'autres. Cela reflète la complexité du vin et la façon dont différents ingrédients et méthodes de fabrication peuvent influencer le goût final. 

## Analyse bivariée
### Variable qualité
Nous n’allons pas réaliser cette analyse sur l’ensemble de nos variables, en effet, étant donné que nous avons 2 variables targets différentes, cela serait trop long. Ainsi, nous avons sélectionné 4 variables explicatives qui semblent pertinentes par rapport à nos variables qualité et type. Il s’agit des variables Alcool, Acidité Volatile, Sucre Résiduel et Chlorides. Ces variables ont été choisies car elles jouent un rôle crucial dans la détermination des caractéristiques sensorielles et de la conservation du vin.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/fec01362-5124-4e47-8a8f-087a061ea038)

Le graphique montre que les vins de meilleure qualité tendent à avoir une teneur en alcool plus élevée, avec une augmentation progressive de l'alcool allant des vins de qualité inférieure aux vins de qualité supérieure. Il y a cependant une exception à cette tendance avec les vins notés 6, qui ont une teneur en alcool légèrement plus basse que ceux notés 5. Les barres d'erreur indiquent une variabilité similaire dans la teneur en alcool à travers les différentes qualités de vin, à l'exception des vins de qualité 9, qui montrent une plus grande variabilité.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/a7a03a00-13a0-4ba0-9f3f-8d5adf0aa716)

Le graphique indique que les vins de qualité supérieure ont généralement une acidité volatile plus basse. On observe également une certaine stagnation à partir des vins ayant une qualité supérieure à 5. Ainsi, une acidité volatile élevée serait gage de mauvais vin, alors qu’une acidité volatile plutôt faible ne nous permettrait pas de statuer entre un moyen ou un bon vin.
Cependant, les vins avec les notes de qualité les plus élevées et les plus basses montrent une plus grande variabilité dans l'acidité volatile que ceux de qualité moyenne. Cela suggère que l'acidité volatile est un indicateur clé de la qualité, où moins d'acidité volatile correspond à une meilleure qualité perçue.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/b3fb1370-60d4-4412-8a2a-070a79fbcdb2)

Il n'y a pas de tendance claire reliant la qualité du vin au sucre résiduel; les vins de qualité moyenne et supérieure ont des niveaux de sucre résiduel similaires. On note une grande variabilité dans le sucre résiduel pour les vins de toutes les qualités, en particulier pour les vins notés 9, indiquant que la douceur peut varier considérablement au sein d'une même catégorie de qualité. Ainsi, on peut conclure que le sucre résiduel n'est pas un indicateur direct de la qualité du vin.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/5a5cfccc-ccf0-4f4e-8df6-bbbc2960d3c1)

On observe que les vins de qualité inférieure ont des concentrations plus élevées en chlorides, tandis que les vins de qualité supérieure ont tendance à en avoir moins. La concentration en chlorides diminue globalement à mesure que la qualité augmente. La variabilité des concentrations en chlorides semble diminuer également avec l'augmentation de la qualité, particulièrement visible pour les vins de qualité 9. Cela nous permet de dire que les vins mieux notés ont une composition plus cohérente en termes de chlorides.


### Variable type

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/9b6e7b16-7045-4c84-bca8-23b210e15c80)

Le graphique montre que les vins blancs ont des niveaux de dioxyde de soufre total nettement plus élevés que les vins rouges, avec une variabilité moindre dans les concentrations pour les vins rouges.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ee57dc69-da2d-4dd1-87a4-0c0554e11f63)

Les vins rouges présentent des concentrations plus élevées en chlorides par rapport aux vins blancs. En moyenne, sur notre échantillon, on note presque deux fois plus de chlorides pour un vin rouge par rapport à un vin blanc.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/efb7e23c-9f6f-4146-b8ed-69cff8e7dde5)

Pour ce qui est de la teneur en sucre, on remarque un taux plus de 2 fois supérieur pour les vin blanc par rapport aux vins rouges.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/98ad2ec7-1df7-4b99-ac83-f6d3c1bc86fc)

Enfin, pour le taux d’alcool contenu dans les vins, on ne remarque aucune différence, signifiant que le taux d’alcool ne dépend pas du type de vin. Ainsi, notre modèle ne pourra pas se baser sur cette variable pour reconnaître le type de vin.



![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/79b3eb17-c8ef-484c-9f9c-e5f26c979951)

Le graphique à barres présenté détaille la distribution des notes de qualité pour les deux catégories de vins. Il ressort que la note '5' est la plus commune pour les rouges, tandis que les blancs sont le plus souvent notés '6', révélant une qualité perçue légèrement meilleure pour ces derniers. Les données visuelles suggèrent que les vins blancs tendent à être évalués plus favorablement que les rouges comme en témoignent les barres plus élevées pour les notes '6', '7', et '8' par rapport aux vins rouges.

## Test de Khi-II

Une étape clé est d'examiner la relation entre la qualité du vin et son type (rouge ou blanc) afin de vérifier qu’elles ne sont pas dépendantes.
Pour cela, un test statistique de Chi2 a été utilisé, qui est un outil standard pour évaluer si deux variables catégorielles sont indépendantes l'une de l'autre ou non.
Le test de Chi2 appliqué aux données a révélé des résultats significatifs. Avec une valeur de Chi2 de 117.03 et une p-valeur extrêmement faible (approximativement 6.86e-23), le test indique clairement que la qualité du vin et le type de vin ne sont pas indépendants.
Ainsi nous nous sommes demandé s’il ne fallait pas retirer les variables type et qualité dans les modèles visant à expliquer l'autre. Cependant, l'intégration de la variable "type" dans la modélisation de la qualité, et de la "qualité" pour prédire le type, est une approche qui nous semble malgré tout pertinente puisque cela permet d'exploiter pleinement les données disponibles et in fine d’augmenter la précision de la prédiction de notre modèle, ce qui est l’objectif de ce dossier.



## Distribution des features

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/7bfa72b3-a5e2-4daa-b582-e2c29c23d4a6)

L'analyse des caractéristiques des vins à travers des graphiques en violon révèle une hétérogénéité intéressante propre à la vinification. L'acidité fixe s'inscrit dans une distribution quasi normale, gravitant majoritairement autour de 7 à 8 g/dm³. Ainsi, si la plupart des vins suivent un standard, certains s'écartent de la norme avec des acidités nettement plus élevées.
En ce qui concerne l'acidité volatile, la tendance est clairement orientée vers des valeurs faibles, bien que quelques vins se démarquent par des niveaux supérieurs.
La concentration en acide citrique est en général modérée, se concentrant entre 0,25 et 0,5 g/dm³, avec quelques exceptions notables qui pourraient influencer le goût et la conservation du vin.
Le sucre résiduel, quant à lui, présente une distribution fortement asymétrique vers la droite, ce qui indique une prédominance de vins avec moins de 10 g/dm³. Cependant, la présence d'une queue longue à droite du graphique souligne l'existence de vins probablement doux, avec des teneurs en sucre bien plus élevées.
Les niveaux de chlorures sont concentrés et généralement bas.
La distribution du dioxyde de soufre libre et total est légèrement inclinée vers la droite, ce qui révèle que si la plupart des vins contiennent des concentrations basses à modérées, quelques-uns présentent des niveaux plus élevés.
La densité et le pH affichent des distributions normales.
Les sulfates, montrent aussi une inclinaison vers des valeurs basses, avec là encore quelques vins qui s'en distinguent par des concentrations plus importantes.
La teneur en alcool expose une distribution multimodale, reflétant la diversité des types de vins, des plus légers aux plus corsés, et des spécialités comme les vins fortifiés.
Enfin, la qualité, bien que subjective, semble se concentrer dans une gamme moyenne (5 à 7), avec moins de vins atteignant des scores élevés.

## Analyse multivariée


A présent, nous concentrons notre analyse sur les corrélations entre nos différentes variables afin d’éliminer les variables inutiles à notre modélisation  en raison de l’une multicolinéarité potentielle. Pour ce faire nous analysons la matrice des corrélation.


![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e360b141-ea74-46ae-8b30-d48ba316b6a6)

Étant donné qu’une matrice des corrélations classiques n’est pas très visible, nous en réalisons une ou seulement les corrélation supérieure à 0,65 sont affichées. 
Nous obtenons donc 2 groupes de variables corrélées entre-elles, à savoir sulfure dioxyde libre et sulfure dioxyde total avec une corrélation de 0,74 ainsi que les variables densité et alcool qui sont corrélées à -0,7.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/aabe1c2e-6016-4cab-be90-8f66896ae1db)

Pour choisir quelles variables nous allons garder, nous nous intéressons aux diagrammes en barres de l’analyse bivariée. 
Premièrement pour les variables liées au sulfure, on ne remarque pas de différences significatives entre les 2, par rapport à la variable qualité comme nous pouvons le voir sur le graphique ci-dessous

<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ed4d08d2-6124-42b9-92b9-e8c5aea79d87" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/f76ba96e-6078-4cfe-95cf-bb3af8c99938" width="420"/>

Cependant, pour la variable type on note que la variable sulfure dioxyde total marque une plus grande différence par rapport au type de vin la rendant plus intéressante pour le déterminer par la suite dans nos modèles. Ainsi, c’est la variable que nous retiendrons.

<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e60e2ee7-7f5b-45a0-a0d2-853f8b604988" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/4f8d92e7-2c6a-4d26-b19a-cb317f0bc51d" width="420"/>



Ensuite, nous procédons à la même analyse pour nos variables densité et alcool. Par rapport au type de vin.

<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/1d33828c-19d7-496c-9a8a-4eebb6b7c71c" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/1b7f6664-54eb-4182-b853-7387536c51bc" width="420"/>


on observe qu'il n'y a aucune différence pour la densité ainsi que pour la teneur en alcool. Nous nous intéressons alors à la variable qualité. 

<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/6a94d0df-bf00-4162-b2c6-df7392b47a84" width="49%"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/75f3a62f-2e4e-4eee-8028-3ee2d47e3da2" width="49%"/>


La densité ne semble pas non plus impacter la qualité d’un vin puisque l’on observe aucune différence de valeur par rapport aux notes de qualité. Par ailleurs, pour ce qui est de la teneur en alcool, on remarque assez facilement une augmentation progressive de celle-ci à mesure que la qualité augmente pour les notes supérieures à 5. Ainsi il semble plus pertinent de conserver la variable alcool pour nos modèles puisqu’elle permet de différencier davantage les vins que la variable densité qui s'avère moins intéressante en raison de sa faible variabilité en fonction du type et de la qualité. 


# Préparation de la BDD

Suite à l'examen détaillé des variables individuelles et à l'étude des liens qu'elles entretiennent avec nos deux variables cibles, l'étape suivante consiste à affiner notre base de données. Cette phase de préparation implique un nettoyage, en effet,  les données manquantes seront écartées et les points atypiques, susceptibles de fausser nos modèles prédictifs, seront corrigées. Ainsi, notre base sera prête pour la construction de  différents modèles prédictifs.

## Recodage des variables 

Lors de la préparation des données pour un modèle de machine learning, il est crucial de convertir toutes les variables catégorielles en un format que le modèle peut comprendre. En général, les modèles de machine learning travaillent avec des données numériques, donc les étapes d'encodage transforment les informations catégorielles en nombres.

Dans notre démarche, nous traitons deux types de variables catégorielles : une variable ordinale (`quality`) et une variable nominale (`type`).

Pour la variable ordinale `quality`, qui représente une note de qualité du vin, nous attribuons à chaque catégorie un code numérique unique. Cela est fait dans le respect de l'ordre inhérent à la variable, où chaque niveau de qualité supérieur reçoit un numéro plus élevé que le niveau précédent. 

Ensuite, pour la variable nominale `type`, nous utilisons deux approches d'encodage différentes. Tout d'abord, nous appliquons un encodage binaire, où chaque type de vin est représenté par un 0 ou un 1. Cette méthode est efficace lorsque la variable catégorielle a seulement deux catégories.

Parallèlement à l'encodage binaire, nous appliquons également un encodage one-hot sur la même variable. Cela crée une nouvelle colonne pour chaque catégorie de la variable `type`, où la présence de chaque type de vin est indiquée par un 1 dans sa colonne respective et des 0 dans toutes les autres. Cette méthode est particulièrement utile lorsque la variable catégorielle comporte plus de deux catégories et où il n'y a pas d'ordre inhérent à considérer.

Ces étapes d'encodage sont essentielles car elles permettent de transformer des données textuelles en signaux numériques qui préservent les informations catégorielles d'origine tout en les rendant interprétables par les algorithmes de machine learning. L'encodage one-hot est souvent préféré car il ne suppose aucune relation d'ordre entre les catégories, ce qui pourrait induire le modèle en erreur. Cependant, pour les variables ordinales, l'encodage ordinal est préférable car il conserve la hiérarchie des catégories. Choisir le bon type d'encodage pour chaque variable est donc une étape critique pour assurer la précision et l'efficacité des modèles prédictifs.

## traitement des valeurs manquantes 

A présent, il est important de traiter les valeurs manquantes. Pour ce faire on regarde le nombre d’outliers par variable et on obtient les résultats suivants.


| Feature              | Value |
|----------------------|-------|
| Type                 | 0     |
| Fixed Acidity        | 10    |
| Volatile Acidity     | 8     |
| Citric Acid          | 3     |
| Residual Sugar       | 2     |
| Chlorides            | 2     |
| Total Sulfur Dioxide | 0     |
| pH                   | 9     |
| Sulphates            | 4     |
| Alcohol              | 0     |
| Quality              | 0     |


Le nombre de valeurs manquantes est très faible donc il sera mieux de simplement les supprimer. Une imputation par la moyenne et la médiane aurait pu être réaliser mais cela n'est pas indispensable en raison de leur faible nombre dans notre dataset comme nous pouvons le constater avec le tableau suivant qui nous donne le pourcentage d’outliers par variables.

| Feature              | Value      |
|----------------------|------------|
| Type                 | 0.000000   |
| Fixed Acidity        | 0.153917   |
| Volatile Acidity     | 0.123134   |
| Citric Acid          | 0.046175   |
| Residual Sugar       | 0.030783   |
| Chlorides            | 0.030783   |
| Total Sulfur Dioxide | 0.000000   |
| pH                   | 0.138525   |
| Sulphates            | 0.061567   |
| Alcohol              | 0.000000   |
| Quality              | 0.000000   |


## points atypiques 

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/9766de5a-6831-433e-99f2-d55f1c7a8fba)

L'analyse des boxplots pour l'ensemble des variables numériques de notre jeu de données révèle la présence de valeurs aberrantes pour chaque variable. Cependant, l'hétérogénéité des échelles entre les différentes variables complique la visualisation précise et la comparaison directe de ces outliers. Pour remédier à cela et faciliter une interprétation plus claire, nous procéderons à la réalisation de boxplots par variables ayant une échelle comparable.

<p float="left">
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ec271eca-b71d-407d-90b0-2cea9fe36e3a" width="32%"/>
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/c953dd70-f816-42cc-ac04-dade37295dff" width="32%"/>
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/a2439082-9f27-44d0-9ece-c71225fa05e4" width="32%"/>
</p>


Les boxplots des différentes variables du vin montrent une dispersion variable des données, on peut noter une large variabilité pour les variables 'Residual sugar' et 'total sulfur dioxide' qui présentent une quantité notable d'outliers. Les autres variables, telles que 'fixed acidity', 'alcohol', 'volatile acidity', 'citric acid', 'chlorides', 'pH' et 'sulphates', affichent également des outliers, mais dans une moindre mesure.

Ainsi nous procédons à leurs corrections de la manière suivante. Pour ce faire, nous utilisons la fonction zscore de la bibliothèque scipy.stats pour identifier les valeurs atypiques. Cette fonction calcule le score z de chaque point de données, nous permettant de connaître leur éloignement de la moyenne en termes d'écart-types. Les valeurs avec un score z supérieur à 3 ou inférieur à -3 sont considérées atypiques. Nous nous basons sur la règle statistique que la majorité des données dans une distribution normale se trouvent dans cette plage. Nous pouvons donc retirer ces valeurs de notre dataframe et passer à la partie rééquilibrage.


## Split
Cette étape consiste en la préparation des données pour le machine learning en séparant les caractéristiques et la cible, en divisant les données en ensembles d'entraînement et de test. Nous avons donc un ensemble test composé de 20% des valeurs et un ensemble train avec le reste de celles-ci. De plus, nous avons vérifié la cohérence de la division, nous avons donc 4782 observations pour l'entraînement et 1196 pour le test.

## Rééquilibrage

Maintenant, il est essentiel de procéder au rééquilibrage de la variable cible 'type', qui est de nature binaire et est distribuée de manière inégale avec environ 20% de vins rouges contre 80% de vins blancs. Cette disproportion peut entraîner un biais significatif dans les performances de notre modèle, en le prédisposant à mieux reconnaître les vins blancs au détriment des rouges.

Pour remédier à cela, des techniques de rééquilibrage telles que le suréchantillonnage des vins rouges ou le sous-échantillonnage des vins blancs peuvent être employées. Le suréchantillonnage, par exemple, impliquerait de générer artificiellement des données supplémentaires pour les vins rouges, ou de dupliquer les échantillons existants, afin de compenser leur présence plus faible. En outre, le sous-échantillonnage réduirait le nombre d'échantillons de vins blancs pour équilibrer la répartition. Dans notre cas, nous avons appliqué un sur-échantillonnage à l’aide de la fonction RandomOverSampler.

En équilibrant la présence des deux classes de 'type' dans notre jeu de données d'entraînement, nous pouvons améliorer significativement la capacité du modèle à apprendre de manière équitable les caractéristiques de chaque type de vin. Cela permettra une amélioration de la précision globale du modèle et assurera que les prédictions soient aussi fiables pour les vins rouges que pour les vins blancs. 


## Standardisation

Vient à présent l’étape de standardisation de nos variables. Elle consiste en une technique de traitement des données pour rendre les variables plus comparables et donc plus adaptées à l'analyse. La standardisation transforme les données de manière à ce que leur moyenne soit égale à 0 et leur écart-type égal à 1. Cela est particulièrement utile lorsque les caractéristiques ont des échelles différentes comme c’est notre cas ici. Pour cela, nous avons utilisé la fonction StandardScaler de la bibliothèque scikit-learn.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/27a1be4d-8f42-4554-a4e1-2a5f2c49f910)

# Modélisation

Maintenant que notre base est prête, nous pouvons passer à la réalisation des modèles. Nous allons tester différents modèles pour chaque analyse puis nous comparerons leur qualité à l’aide de différents indicateurs.

## Analyse multiclasse 
Nous commençons par l’analyse multiclasse qui à pour but de prédire la note de qualité d’un vin.

### Les modèles utilisés 

Pour prédire la qualité du vin, nous avons utilisé plusieurs modèles, à savoir les approches One-vs-One (OvO), One-vs-Rest (OvR) et les réseaux de neurones.
Le modèle OvO compare chaque paire de catégories pour mieux les distinguer, un peu comme si on comparait chaque type de vin deux par deux pour voir ce qui les différencie. Le modèle OvR, quant à lui, regarde chaque catégorie de vin séparément et la compare à toutes les autres en même temps. C'est une méthode plus simple et rapide, surtout quand il y a beaucoup de catégories de vins différentes à analyser.
Nous avons également utilisé des réseaux de neurones pour évaluer la qualité du vin. Le réseau de neurones passe en revue toutes les informations sur chaque vin (l’ensemble de nos variables explicatives) et apprend à partir de ces données pour prédire sa qualité. 

### Comparaison des modèles 
Maintenant, nous pouvons passer à la comparaison des résultats obtenus pour ces différents modèles dans l’optique de sélectionner le meilleur. Nous commencerons par comparer les modèles ovo et ovr puis nous intégrerons le modèle de réseau de neurones à l'analyse.

Nous avons réalisé une première validation croisée, en utilisant 5 subdivisions (folds), pour examiner nos 2 modèles différents. Le graphique présenté ci-dessous illustre le score de précision, qui correspond au ratio des classifications correctes sur le nombre total de classification pour chacune des 5 subdivisions.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/82df27bd-849f-4db8-8ff1-f8173466c9d3)

Le graphique illustre la performance de deux modèles sur cinq essais distincts d'une validation croisée. Le modèle représenté OVO (ligne bleu) démontre une précision supérieure et plus stable que le modèle OVR, signalant une fiabilité accrue et une constance dans la prédiction sur les différents folds. Néanmoins, il est important de noter que la précision, bien qu'utile, n'est pas l'unique indicateur de la performance d'un modèle. Des mesures telles que le Recall, l'aire sous la courbe ROC (AUC) et le score F1 sont essentielles pour une évaluation approfondie et précise des modèles.


Nous comparons maintenant ces métriques avec le modèle réseau de neurones et obtenons les métriques suivantes :

| Méthode             | Précision | Rappel  | F1-score | Accuracy |
|---------------------|-----------|---------|----------|----------|
| OVO (One-vs-One)    | 0.5515    | 0.4423  | 0.4657   | 0.4423   |
| OVR (One-vs-Rest)   | 0.5188    | 0.3562  | 0.3698   | 0.3562   |
| Réseau de Neurones  | 0.2443    | 0.3208  | 0.2282   | 0.3821   |


Nous concluons donc que le modèle OVO semble offrir les meilleures performances, avec une précision relativement élevée et une meilleure capacité à classer correctement les échantillons par rapport aux modèles OVR et Réseau de Neurones. Le modèle OVR, bien qu'ayant une précision et un rappel inférieurs à OVO, surpasse légèrement le Réseau de Neurones en termes d'accuracy. Cependant, le Réseau de Neurones se trouve à la traîne avec des scores nettement inférieurs dans toutes les métriques, notamment en précision et en rappel, suggérant qu'il pourrait avoir du mal à gérer correctement les classifications dans ce cas spécifique.

### Grid Search sur meilleurs modèles

Nous n'avons pas procédé à l'optimisation par grid search pour le meilleur modèle multiclasse, comme initialement prévu, car le processus s'est avéré être excessivement long et fastidieux à exécuter.


## Analyse de classification binaire 

Nous nous interessons maintenant à l’analyse binaire qui à pour but de prédire le type de vin.

### Les modèles utilisés 

Pour commencer cette partie, nous résumons succinctement chaque méthode utilisée : 

Régression Logistique (lgr) : Utilisée pour des classifications binaires, elle modélise la probabilité d'un événement en fonction des variables d'entrée.
Classification à Vecteurs de Support Linéaire (lsvc) : Trouve un hyperplan qui sépare de manière optimale les classes, elle s’avère particulièrement efficace pour des classifications précises.
Classificateur à Descente de Gradient Stochastique (sgdc) : Optimise des modèles linéaires de manière efficace, particulièrement adapté aux grands ensembles de données.
Classification à Vecteurs de Support (svc) : Utilise une marge maximale pour distinguer les classes, efficace même avec des frontières de décision complexes.
Fonction de Base Radiale (rbf) et Noyau Polynomial (poly) : Deux fonctions de noyau pour SVM, traitant respectivement les relations non linéaires et complexes entre les caractéristiques.
Nous utiliserons également un réseau de neurones pour cette analyse.

### Comparaison des modèles 

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/dde84a0c-f0ed-4146-b4af-5ee4be727334)

Il est important de souligner que la précision fournit une évaluation de la performance globale d'un modèle sur un ensemble de données, par ailleurs, la stabilité indique la constance de cette performance sur différents ensembles de données ou situations. Un modèle optimal allie donc une haute précision à une grande stabilité, assurant ainsi des prédictions fiables et régulières.

| Modèle N° | Accuracy                | Std                        |
|-----------|-------------------------|----------------------------|
| 0         | 0.9892885993348649      | 0.001433107725945217       |
| 1         | 0.9891563242025899      | 0.0015924545038446404      |
| 2         | 0.9880985602730481      | 0.002765273489977155       |
| 3         | 0.989949800144777       | 0.001271630019946524       |
| 4         | 0.9923299656941428      | 0.001962150690119038       |
| 5         | 0.9927268785167002      | 0.0015505065873888573      |

En conclusion, les modèles rbf et poly se distinguent par leur performance, avec une légère avance pour le modèle poly, notamment dans les derniers folds. Bien que les autres modèles (lgr, lsvc, sgdc, svc) affichent une certaine variabilité, ils maintiennent une stabilité relative qui ne permet pas de surpasser les modèles rbf et poly. Concernant la stabilité, le modèle poly présente une légère supériorité par rapport au modèle rbf, le positionnant ainsi comme le meilleur sur l'ensemble des métriques analysées.
Nous regardons à présent les métriques pour le réseaux de neurones et observons les résultats suivants ; 

| Métrique  | Valeur                 |
|-----------|------------------------|
| Accuracy  | 0.9941471571906354     |
| Precision | 0.9936775553213909     |
| Recall    | 0.9989406779661016     |
| F1 Score  | 0.9963021658742736     |

Le modèle a montré une excellente performance sur l'ensemble de validation, avec une exactitude remarquable de 99.41%, indiquant que presque toutes les prédictions étaient correctes. La précision était également très élevée à 99.37%, signifiant que la majorité des cas classés comme positifs étaient réellement positifs. Le rappel, à 99.89%, montre que le modèle a identifié avec succès presque tous les cas positifs réels. Le score F1 élevé de 99.63% révèle un équilibre parfait entre précision et rappel. La matrice de confusion confirme ces résultats avec un très faible nombre de faux positifs et un seul faux négatif, soulignant la fiabilité exceptionnelle du modèle dans ses prédictions faisant de lui celui que l’on retiendra comme notre meilleur modèle. au vu de son accuracy qui dépasse nos meilleurs modèles SVM.


faire une phrase pour expliquer pourquoi on gridsearch pas sur le rdn



### Grid Search sur le meilleur modèle SVM


L'approche GridSearch, souvent utilisée en apprentissage automatique, est une méthode pour sélectionner les meilleurs paramètres pour un modèle donné. Cette technique utilise la création d'une “grille" de paramètres possibles pour le modèle. Chaque combinaison de paramètres dans cette grille est ensuite évaluée et comparée.
Dans les modèles SVM, C et gamma sont des paramètres permettant l’ajustement du modèle. Le paramètre C contrôle la régularisation et influence la complexité du modèle. Une valeur plus faible de C mène à une simplification du modèle, limitant le surajustement mais risquant le sous-ajustement, tandis qu'une valeur élevée permet une meilleure adaptation aux données d'entraînement mais avec un risque accru de surajustement. 
D'autre part, gamma détermine l'impact de chaque point de données sur la formation du modèle. Un gamma élevé augmente l'influence de chaque point, menant potentiellement à un modèle trop spécifique aux données d'entraînement (surajustement), alors qu'un gamma faible diminue cette influence, favorisant ainsi la généralisation du modèle.


| Paramètre | Valeur |
|-----------|--------|
| C         | 10     |
| gamma     | 0.1    |


Cela indique, comme dit précédemment qu'une régularisation modérée combinée à une influence relativement faible de chaque point de données produit les meilleurs résultats pour notre modèle. Regardons à présent l’évolution des métriques.


| Metric    | Class 0 | Class 1 |
|-----------|---------|---------|
| Precision | 0.99    | 0.99    |
| Recall    | 0.97    | 1.00    |
| F1-score  | 0.98    | 0.99    |
| Support   | 252     | 944     |


Le modèle a démontré une excellente performance sur l'ensemble de test, atteignant une exactitude globale de 99.16%. Dans le détail, pour la classe 0, il a montré une précision de 99%, un rappel de 97%, et un score F1 de 98%, indiquant une grande précision et une bonne capacité à identifier correctement cette classe. Pour la classe 1, le modèle a été encore plus performant, avec une précision et un rappel de 99% et 100% respectivement, aboutissant à un score F1 de 99%. Ces résultats soulignent non seulement la capacité du modèle à distinguer avec précision les deux classes, mais aussi son équilibre remarquable entre précision et rappel, comme reflété dans les scores F1 élevés et les moyennes globales proches de 99%.



### Importance des variables 

Pour finir cette analyse, il nous a semblé pertinent d’étudier l’importance des variables, pour ce faire nous avons utilisé une méthode de permutation.

La méthode de permutation permet d’évaluer l'importance des caractéristiques dans un modèle SVM. Pour ce faire, on mélange aléatoirement les valeurs d'une caractéristique dans l'ensemble de test et observe l'effet sur la performance du modèle. Une baisse notable de la performance après la permutation indique que la caractéristique est importante. En répétant cette opération plusieurs fois et en moyennant les résultats, on obtient une estimation fiable de l'importance de chaque caractéristique. Le graphique à boîtes illustre cette importance, permettant d'identifier rapidement les caractéristiques essentielles au modèle.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/7df308b3-c7a7-4c65-83b6-39b5c4a5a94a)

En observant le graphique généré, nous pouvons interpréter l'importance des différentes caractéristiques dans la détermination du type de vin. Les caractéristiques situées vers le haut, avec des valeurs d'importance plus élevées sur l'axe des x, sont celles qui influencent le plus le modèle. Par exemple, les "chlorides" et le "total sulfur dioxide" semblent être des indicateurs très influents avec des boîtes s'étendant plus loin sur l'axe des x. Cela suggère qu'une permutation de ces valeurs a un impact notable sur la capacité du modèle à différencier les vins rouges des blancs.
En revanche, les caractéristiques en bas du graphique, telles que "quality_5" et "quality_6", ont des boîtes qui sont très proches de l'origine sur l'axe des x, indiquant que la permutation de ces caractéristiques a peu ou pas d'effet sur la performance du modèle. Nous concluons donc que ces caractéristiques sont potentiellement moins importantes pour la prédiction du type de vin.

# Conclusion



</div>












