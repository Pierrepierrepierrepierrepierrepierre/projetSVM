<div align="center">
  
# Classification de la qualité et du type de vin

</div>

<p align="center">
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/117682143/6631788f-243c-4ca9-bbb0-cf1c898ca7c2" alt="White and Red Wine Glasses" width="400"/>
</p>


# I. Introduction

Dans le cadre de notre projet de Machine Learning, nous nous sommes lancés dans l'analyse approfondie d'une base de données portant sur la qualité et le type de vins. Cette base, récupérée sur le site de référence Kaggle et créé par un auteur du nom de Raj Parmar, inclut des informations détaillées sur un vaste éventail de vins, rouges comme blancs.
Notre objectif est double : prédire la qualité du vin, évaluée sur une échelle de 0 à 10, et classifier le vin en tant que rouge ou blanc. Cette tâche s'appuie sur l'exploitation de divers paramètres chimiques et physiques des vins, tels que l'acidité, le sucre résiduel, les chlorures, le sulfite, la densité, le pH, les sulfates, et la teneur en alcool. 
&nbsp; 

La première phase de notre projet a consisté en une compréhension approfondie de la base de données, suivie d'un nettoyage rigoureux et d'une préparation des données. Cette étape essentielle a permis d'établir des fondations solides pour nos analyses et modélisations ultérieures.
Dans un deuxième temps, nous avons développé et testé une série de modèles de prédiction visant à évaluer la qualité et à déterminer le type de vin. 
&nbsp;

# II. Description du jeu de données

Nous avons exploité une base de données disponible sur le site Kaggle, spécialisée dans les caractéristiques et la qualité des vins rouges et blancs. Cette base de données, riche en informations, nous permet d'analyser en profondeur les différents aspects influençant la qualité du vin. La base de données initiale comprend des enregistrements sur un grand nombre de vins, couvrant diverses variables chimiques et physiques.

## a) Description des variables

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



# III. Analyse exploratoire 
## a) Analyse univariée

Nous passons à présent à l'analyse exploratoire, nous allons éxaminer de plus près les caractéristiques distinctives de notre collection de vins à travers une analyse univariée, révélant les tendances et les nuances de leur composition et de leur qualité.


### 1- Targets

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/533cd6d5-08a9-48db-89bc-f08ef4b98d22)

</div>

Le graphique ci dessus nous montre la répartition de deux types de vins. On observe une répartition assez inégale, en effet, le vin blanc représente une part beaucoup plus importante, avec 75.39%, tandis que le vin rouge est moins fréquent à 24.61%.

<div align="center">


<img width="452" alt="Capture d’écran 2024-01-29 135751" src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/aa31ac83-fcd7-4edb-9a11-499025b88127">

 
</div>

Pour ce qui est de la variable qualité, nous avons préféré un histogramme en barre pour analyser la répartition. Les histogrammes en barres offrent une comparaison et une interprétation plus claires des fréquences lorsque l’on analyse des variables catégorielles. 
Ainsi, nous observons une distribution asymétrique où la majorité des vins se concentrent autour des notes de qualité moyenne correspondant à une loi normale, ce qui est logique  puisque la plupart des vins reçoivent des notes moyennes et moins reçoivent des notes extrêmes.

De ce fait, nous devrons procéder à un rééquilibrage des classes afin d’avoir des modèles pertinents.

### 2- Features

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

## b) Analyse bivariée


### Target 'Quality'


Nous n’allons pas réaliser cette analyse sur l’ensemble de nos variables, en effet, étant donné que nous avons 2 variables targets différentes, cela serait trop long. Ainsi, nous avons sélectionné 4 variables explicatives qui semblent pertinentes par rapport à nos variables qualité et type. Il s’agit des variables Alcool, Acidité Volatile, Sucre Résiduel et Chlorides. Ces variables ont été choisies car elles jouent un rôle crucial dans la détermination des caractéristiques sensorielles et de la conservation du vin.

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/fec01362-5124-4e47-8a8f-087a061ea038)

</div>

Le graphique montre que les vins de meilleure qualité tendent à avoir une teneur en alcool plus élevée, avec une augmentation progressive de l'alcool allant des vins de qualité inférieure aux vins de qualité supérieure. Il y a cependant une exception à cette tendance avec les vins notés 6, qui ont une teneur en alcool légèrement plus basse que ceux notés 5. Les barres d'erreur indiquent une variabilité similaire dans la teneur en alcool à travers les différentes qualités de vin, à l'exception des vins de qualité 9, qui montrent une plus grande variabilité.

<div align="center">

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/a7a03a00-13a0-4ba0-9f3f-8d5adf0aa716)

</div>

Le graphique indique que les vins de qualité supérieure ont généralement une acidité volatile plus basse. On observe également une certaine stagnation à partir des vins ayant une qualité supérieure à 5. Ainsi, une acidité volatile élevée serait gage de mauvais vin, alors qu’une acidité volatile plutôt faible ne nous permettrait pas de statuer entre un moyen ou un bon vin.
Cependant, les vins avec les notes de qualité les plus élevées et les plus basses montrent une plus grande variabilité dans l'acidité volatile que ceux de qualité moyenne. Cela suggère que l'acidité volatile est un indicateur clé de la qualité, où moins d'acidité volatile correspond à une meilleure qualité perçue.

<div align="center">

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/b3fb1370-60d4-4412-8a2a-070a79fbcdb2)

</div>

Il n'y a pas de tendance claire reliant la qualité du vin au sucre résiduel; les vins de qualité moyenne et supérieure ont des niveaux de sucre résiduel similaires. On note une grande variabilité dans le sucre résiduel pour les vins de toutes les qualités, en particulier pour les vins notés 9, indiquant que la douceur peut varier considérablement au sein d'une même catégorie de qualité. Ainsi, on peut conclure que le sucre résiduel n'est pas un indicateur direct de la qualité du vin.

<div align="center">

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/5a5cfccc-ccf0-4f4e-8df6-bbbc2960d3c1)

</div>

On observe que les vins de qualité inférieure ont des concentrations plus élevées en chlorides, tandis que les vins de qualité supérieure ont tendance à en avoir moins. La concentration en chlorides diminue globalement à mesure que la qualité augmente. La variabilité des concentrations en chlorides semble diminuer également avec l'augmentation de la qualité, particulièrement visible pour les vins de qualité 9. Cela nous permet de dire que les vins mieux notés ont une composition plus cohérente en termes de chlorides.


### Target 'Type'


<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/9b6e7b16-7045-4c84-bca8-23b210e15c80)

</div>

Le graphique montre que les vins blancs ont des niveaux de dioxyde de soufre total nettement plus élevés que les vins rouges, avec une variabilité moindre dans les concentrations pour les vins rouges.


<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ee57dc69-da2d-4dd1-87a4-0c0554e11f63)

</div>

Les vins rouges présentent des concentrations plus élevées en chlorides par rapport aux vins blancs. En moyenne, sur notre échantillon, on note presque deux fois plus de chlorides pour un vin rouge par rapport à un vin blanc.


<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/efb7e23c-9f6f-4146-b8ed-69cff8e7dde5)

</div>

Pour ce qui est de la teneur en sucre, on remarque un taux plus de 2 fois supérieur pour les vin blanc par rapport aux vins rouges.

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/98ad2ec7-1df7-4b99-ac83-f6d3c1bc86fc)

</div>

Enfin, pour le taux d’alcool contenu dans les vins, on ne remarque aucune différence, signifiant que le taux d’alcool ne dépend pas du type de vin. Ainsi, notre modèle ne pourra pas se baser sur cette variable pour reconnaître le type de vin.


<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/79b3eb17-c8ef-484c-9f9c-e5f26c979951)

</div>

Le graphique à barres présenté détaille la distribution des notes de qualité pour les deux catégories de vins. Il ressort que la note '5' est la plus commune pour les rouges, tandis que les blancs sont le plus souvent notés '6', révélant une qualité perçue légèrement meilleure pour ces derniers. Les données visuelles suggèrent que les vins blancs tendent à être évalués plus favorablement que les rouges comme en témoignent les barres plus élevées pour les notes '6', '7', et '8' par rapport aux vins rouges.

## c) Test de Khi-II

Un aspect crucial consiste à examiner la relation entre la qualité du vin et son type (rouge ou blanc) afin de vérifier qu'elles ne sont pas interdépendantes. À cette fin, nous avons employé un test statistique de Chi2, un outil standard pour évaluer l'indépendance entre deux variables catégorielles. Les résultats du test de Chi2 appliqué à nos données ont été significatifs, entraînant le rejet de l'hypothèse nulle qui stipulait l'absence de relation entre les deux variables catégorielles. Avec une p-valeur extrêmement faible (environ 6.86e-23), il est clair que la qualité du vin et le type de vin ne sont pas indépendants.

Cette constatation a suscité la réflexion sur la possibilité d'éliminer les variables "type" et "qualité" des modèles visant à expliquer l'une par l'autre. Cependant, nous avons décidé de maintenir ces variables dans nos modèles. Intégrer la variable "type" dans la modélisation de la qualité, et vice versa, nous semble toujours pertinente, car cela exploite pleinement les données disponibles et vise ultimement à accroître la précision de la prédiction du modèle, qui demeure l'objectif principal de notre étude.




## d) Distribution des features

<div align="center">

![outputviolin](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/d37ad49c-195f-45f4-acea-6a577c0e09ad)

</div>
Les violin plots présentés montrent que les features de l'ensemble de données possèdent des échelles de valeurs variées, des distributions de fréquence différentes, et pour certaines, des valeurs extrêmes. La standardisation de ces features est justifiée car elle permet de mettre chaque feature à la même échelle afin qu'il soit comparable dans nos modèles de machine learning. En ajustant les données pour qu'elles aient une moyenne de zéro et une variance unitaire, on garantit que chaque feature contribue de manière équitable au modèle, sans être influencée par l'ampleur de sa gamme de valeurs. 

## e) Analyse multivariée


A présent, nous concentrons notre analyse sur les corrélations entre nos différentes variables afin d’éliminer les variables inutiles à notre modélisation  en raison de l’une multicolinéarité potentielle. Pour ce faire nous analysons la matrice des corrélation.

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e360b141-ea74-46ae-8b30-d48ba316b6a6)

</div>

Étant donné qu’une matrice des corrélations classiques n’est pas très visible, nous en réalisons une ou seulement les corrélation supérieure à 0,65 sont affichées. 
Nous obtenons donc 2 groupes de variables corrélées entre-elles, à savoir sulfure dioxyde libre et sulfure dioxyde total avec une corrélation de 0,74 ainsi que les variables densité et alcool qui sont corrélées à -0,7.

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/aabe1c2e-6016-4cab-be90-8f66896ae1db)

</div>

Pour choisir quelles variables nous allons garder, nous nous intéressons aux diagrammes en barres de l’analyse bivariée. 
Premièrement pour les variables liées au sulfure, on ne remarque pas de différences significatives entre les 2, par rapport à la variable qualité comme nous pouvons le voir sur le graphique ci-dessous

<div align="center">
  
<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ed4d08d2-6124-42b9-92b9-e8c5aea79d87" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/f76ba96e-6078-4cfe-95cf-bb3af8c99938" width="420"/>

</div>

Cependant, pour la variable type on note que la variable sulfure dioxyde total marque une plus grande différence par rapport au type de vin la rendant plus intéressante pour le déterminer par la suite dans nos modèles. Ainsi, c’est la variable que nous retiendrons.

<div align="center">
  
<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e60e2ee7-7f5b-45a0-a0d2-853f8b604988" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/4f8d92e7-2c6a-4d26-b19a-cb317f0bc51d" width="420"/>

</div>



Ensuite, nous procédons à la même analyse pour nos variables densité et alcool. Par rapport au type de vin.

<div align="center">
  
<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/1d33828c-19d7-496c-9a8a-4eebb6b7c71c" width="420"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/1b7f6664-54eb-4182-b853-7387536c51bc" width="420"/>

</div>

On observe qu'il n'y a aucune différence pour la densité ainsi que pour la teneur en alcool. Nous nous intéressons alors à la variable qualité. 

<div align="center">
  
<img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/6a94d0df-bf00-4162-b2c6-df7392b47a84" width="49%"/> <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/75f3a62f-2e4e-4eee-8028-3ee2d47e3da2" width="49%"/>

</div>


La densité ne semble pas non plus impacter la qualité d’un vin puisque l’on observe aucune différence de valeur par rapport aux notes de qualité. Par ailleurs, pour ce qui est de la teneur en alcool, on remarque assez facilement une augmentation progressive de celle-ci à mesure que la qualité augmente pour les notes supérieures à 5. Ainsi il semble plus pertinent de conserver la variable alcool pour nos modèles puisqu’elle permet de différencier davantage les vins que la variable densité qui s'avère moins intéressante en raison de sa faible variabilité en fonction du type et de la qualité. 


# IV. Préparation de la BDD

Suite à l'examen détaillé des variables individuelles et à l'étude des liens qu'elles entretiennent avec nos deux variables cibles, l'étape suivante consiste à affiner notre base de données. Cette phase de préparation implique un nettoyage, en effet,  les données manquantes seront écartées et les points atypiques, susceptibles de fausser nos modèles prédictifs, seront corrigées. Ainsi, notre base sera prête pour la construction de  différents modèles prédictifs.

## a) Recodage des variables 

Lors de la préparation des données pour un modèle de machine learning, il est essentiel de convertir toutes les variables catégorielles dans un format compréhensible par le modèle. En général, les modèles de machine learning requièrent des données numériques, et donc les étapes d'encodage sont utilisées pour transformer les informations catégorielles en valeurs numériques.

Dans notre approche, nous traitons deux types de variables catégorielles : une variable ordinale (`quality`) et une variable nominale (`type`).

Pour les modèles où la variable cible est 'quality', représentant la note de qualité du vin, nous attribuons à chaque catégorie un code numérique unique respectant l'ordre inhérent à la variable. Ainsi, chaque niveau de qualité supérieur reçoit un numéro plus élevé que le niveau précédent. En ce qui concerne la variable 'type', qui est une feature, nous utilisons un encodage one-hot. Cela crée une nouvelle colonne pour chaque catégorie de la variable `type`, générant ainsi deux nouvelles variables : 'type_white' et 'type_red'.

Ensuite, pour les modèles où la variable cible est `type`, nous appliquons un encodage binaire à l'aide du LabelEncoder. Chaque type de vin est représenté par 0 ou 1, où la valeur 1 correspond au vin blanc et la valeur 0 correspond au vin rouge. De plus, pour la variable 'quality', qui est une feature de notre modèle, nous utilisons un encodage one-hot, créant ainsi une nouvelle variable pour chaque niveau de qualité.

## b) traitement des valeurs manquantes 

A présent, il est important de traiter les valeurs manquantes. Pour ce faire on regarde le nombre d’outliers par variable et on obtient les résultats suivants.

<div align="center">


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

</div>


Le nombre de valeurs manquantes est très faible donc il sera mieux de simplement les supprimer. Une imputation par la moyenne et la médiane aurait pu être réaliser mais cela n'est pas indispensable en raison de leur faible nombre dans notre dataset comme nous pouvons le constater avec le tableau suivant qui nous donne le pourcentage d’outliers par variables.


## c) points atypiques 


L'analyse des boxplots pour l'ensemble des variables numériques de notre jeu de données révèle la présence de valeurs aberrantes pour chaque variable. Cependant, l'hétérogénéité des échelles entre les différentes variables complique la visualisation précise et la comparaison directe de ces outliers. Pour remédier à cela et faciliter une interprétation plus claire, nous procéderons à la réalisation de boxplots par variables ayant une échelle comparable.

<p float="left">
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ec271eca-b71d-407d-90b0-2cea9fe36e3a" width="32%"/>
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/c953dd70-f816-42cc-ac04-dade37295dff" width="32%"/>
  <img src="https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/a2439082-9f27-44d0-9ece-c71225fa05e4" width="32%"/>
</p>


Les boxplots des différentes variables du vin montrent une dispersion variable des données, on peut noter une large variabilité pour les variables 'Residual sugar' et 'total sulfur dioxide' qui présentent une quantité notable d'outliers. Les autres variables, telles que 'fixed acidity', 'alcohol', 'volatile acidity', 'citric acid', 'chlorides', 'pH' et 'sulphates', affichent également des outliers, mais dans une moindre mesure.

Ainsi nous procédons à leurs corrections de la manière suivante. Pour ce faire, nous utilisons la fonction zscore de la bibliothèque scipy.stats pour identifier les valeurs atypiques. Cette fonction calcule le score z de chaque point de données, nous permettant de connaître leur éloignement de la moyenne en termes d'écart-types. Les valeurs avec un score z supérieur à 3 ou inférieur à -3 sont considérées atypiques. Nous nous basons sur la règle statistique que la majorité des données dans une distribution normale se trouvent dans cette plage. Nous pouvons donc retirer ces valeurs de notre dataframe et passer à la partie rééquilibrage.


## d) Split
Cette étape consiste en la préparation des données pour le machine learning en séparant les caractéristiques et la cible, en divisant les données en ensembles d'entraînement et de test. Nous avons donc un ensemble test composé de 20% des valeurs et un ensemble train avec le reste de celles-ci. De plus, nous avons vérifié la cohérence de la division, nous avons donc 4782 observations pour l'entraînement et 1196 pour le test.

## e) Rééquilibrage

Maintenant, il devient crucial d'effectuer le rééquilibrage de nos variables cibles. Prenons l'exemple de la variable cible `type`, qui est binaire et présente une distribution inégale avec environ 20% de vins rouges et 80% de vins blancs. Cette disparité peut introduire un biais significatif dans les performances du modèle, le prédisposant à mieux reconnaître les vins blancs au détriment des vins rouges. Une situation similaire se présente avec la variable `quality`, où certaines notations sont sous-représentées et d'autres sont sur-représentées.

Pour remédier à cette situation, des techniques de rééquilibrage telles que le suréchantillonnage des vins rouges ou le sous-échantillonnage des vins blancs peuvent être mises en œuvre. Dans notre cas, nous avons opté pour le suréchantillonnage en utilisant la fonction RandomOverSampler. Le suréchantillonnage implique la génération artificielle de données supplémentaires pour les vins rouges en dupliquant les échantillons existants, afin de compenser leur présence plus faible.

En résumé, nous avons rééquilibré les classes dans notre ensemble d'entraînement, visant à améliorer significativement la performance de nos modèles. Cette démarche contribuera à améliorer la précision globale du modèle et garantira que les prédictions soient également fiables pour chaque classe.


## f) Standardisation

Vient à présent l’étape de standardisation de nos variables. Elle consiste en une technique de traitement des données pour rendre les variables plus comparables et donc plus adaptées à l'analyse. La standardisation transforme les données de manière à ce que leur moyenne soit égale à 0 et leur écart-type égal à 1. Cela est particulièrement utile lorsque les caractéristiques ont des échelles différentes comme c’est notre cas ici. Pour cela, nous avons utilisé la fonction StandardScaler de la bibliothèque scikit-learn.


# V. Modélisation

Maintenant que notre base est prête, nous pouvons passer à la réalisation des modèles. Nous allons tester différents modèles pour chaque analyse puis nous comparerons leur qualité à l’aide de différents indicateurs.

## a) Analyse multiclasse 

Nous commençons par l’analyse multiclasse qui à pour but de prédire la note de qualité d’un vin.

### 1- Les modèles utilisés 

Pour prédire la qualité du vin, nous avons employé divers modèles, notamment des modèles SVM et un réseau de neurones.

Concernant les modèles SVM, nous avons appliqué les approches One-vs-One (OvO) et One-vs-Rest (OvR). Le modèle OvO compare chaque paire de catégories afin de mieux les distinguer, similaire à une comparaison exhaustive de chaque type de vin deux par deux pour identifier leurs différences. D'un autre côté, le modèle OvR examine chaque catégorie de vin individuellement et la compare simultanément à toutes les autres. Cette méthode est plus simple et rapide, particulièrement lorsqu'il y a un grand nombre de catégories de vins différentes à analyser.
Nous avons employé des réseaux de neurones afin d'évaluer la qualité du vin. La configuration du réseau de neurones a été spécifiée de manière à ce que le modèle tienne compte du fait que la variable cible à prédire est catégorielle. Ainsi, nous avons utilisé la fonction d'activation 'softmax'.

### 2- Comparaison des modèles 
Maintenant, nous pouvons passer à la comparaison des résultats obtenus pour ces différents modèles dans l’optique de sélectionner le meilleur. Nous commencerons par comparer les modèles ovo et ovr entre eux, puis nous intégrerons le modèle de réseau de neurones à l'analyse.

Nous avons réalisé une première validation croisée, pour examiner nos 2 modèles différents. Le graphique présenté ci-dessous illustre le score de précision, qui correspond au ratio des classifications correctes sur le nombre total de classification pour chacune des 5 subdivisions.

<div align="center">
  
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/82df27bd-849f-4db8-8ff1-f8173466c9d3)

</div>


Le graphique illustre la performance de deux modèles sur cinq essais distincts d'une validation croisée. Le modèle représenté OVO (ligne bleu) démontre une précision supérieure et plus stable que le modèle OVR, signalant une fiabilité accrue et une constance dans la prédiction sur les différents folds. Néanmoins, il est important de noter que la précision, bien qu'utile, n'est pas l'unique indicateur de la performance d'un modèle. Des mesures telles que le Recall, l'aire sous la courbe ROC (AUC) et le score F1 sont essentielles pour une évaluation approfondie et précise des modèles.


Nous comparons maintenant ces métriques avec le modèle réseau de neurones et obtenons les métriques suivantes :

<div align="center">
  
| Méthode             | Précision | Rappel  | F1-score | Accuracy |
|---------------------|-----------|---------|----------|----------|
| OVO (One-vs-One)    | 0.5515    | 0.4423  | 0.4657   | 0.4423   |
| OVR (One-vs-Rest)   | 0.5188    | 0.3562  | 0.3698   | 0.3562   |
| Réseau de Neurones  | 0.2443    | 0.3208  | 0.2282   | 0.3821   |

</div>


Nous concluons donc que le modèle OVO semble offrir les meilleures performances, avec une précision relativement élevée et une meilleure capacité à classer correctement les échantillons par rapport aux modèles OVR et réseau de Neurones. Le modèle OVR, bien qu'ayant une précision et un rappel inférieurs à OVO, surpasse légèrement le Réseau de Neurones en termes d'accuracy. De plus, le réseau de Neurones se trouve à la traîne avec des scores nettement inférieurs dans toutes les métriques, notamment en précision et en rappel, suggérant qu'il pourrait avoir du mal à gérer correctement les classifications dans ce cas spécifique.

Par ailleurs, nous n'avons pas procédé à l'optimisation par grid search pour le meilleur modèle multiclasse, comme initialement prévu, car le processus s'est avéré être excessivement long à exécuter.


## b) Analyse de classification binaire 

Nous nous interessons maintenant à l’analyse binaire qui à pour but de prédire le type de vin.

### 1- Les modèles utilisés 

Pour commencer cette partie, nous résumons succinctement chaque méthode utilisée : 

- Régression Logistique (lgr) : Utilisée pour des classifications binaires, elle modélise la probabilité d'un événement en fonction des variables d'entrée.
- Classification à Vecteurs de Support Linéaire (lsvc) : Trouve un hyperplan qui sépare de manière optimale les classes, elle s’avère particulièrement efficace pour des classifications précises.
- Classificateur à Descente de Gradient Stochastique (sgdc) : Optimise des modèles linéaires de manière efficace, particulièrement adapté aux grands ensembles de données.
- Classification à Vecteurs de Support (svc) : Utilise une marge maximale pour distinguer les classes, efficace même avec des frontières de décision complexes.
- Fonction de Base Radiale (rbf) et Noyau Polynomial (poly) : Deux fonctions de noyau pour SVM, traitant respectivement les relations non linéaires et complexes entre les caractéristiques.

Nous inclurons également un réseau neuronal dans cette analyse. Sa spécification diffère de celle utilisée dans la modélisation multiclasse, avec l'utilisation d'une fonction d'activation 'sigmoid' adaptée à la classification binaire.

### 2- Comparaison des modèles 

<div align="center">

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/dde84a0c-f0ed-4146-b4af-5ee4be727334)

</div>

<div align="center">


| Modèle N° | Accuracy                | Std                        |
|-----------|-------------------------|----------------------------|
| 0         | 0.9892885993348649      | 0.001433107725945217       |
| 1         | 0.9891563242025899      | 0.0015924545038446404      |
| 2         | 0.9880985602730481      | 0.002765273489977155       |
| 3         | 0.989949800144777       | 0.001271630019946524       |
| 4         | 0.9923299656941428      | 0.001962150690119038       |
| 5         | 0.9927268785167002      | 0.0015505065873888573      |

</div>

En somme, les modèles rbf et poly se distinguent par leur performance, avec une légère avance pour le modèle poly, notamment dans les derniers folds. Bien que les autres modèles (lgr, lsvc, sgdc, svc) affichent une certaine variabilité, ils maintiennent une stabilité relative qui ne permet pas de surpasser les modèles rbf et poly. Concernant la stabilité, le modèle poly présente une légère supériorité par rapport au modèle rbf, le positionnant ainsi comme le meilleur sur l'ensemble des métriques analysées.

Nous regardons à présent les métriques pour le réseaux de neurones et observons les résultats suivants ; 

<div align="center">

| Métrique  | Valeur                 |
|-----------|------------------------|
| Accuracy  | 0.9941471571906354     |
| Precision | 0.9936775553213909     |
| Recall    | 0.9989406779661016     |
| F1 Score  | 0.9963021658742736     |

</div>


Le modèle a montré une excellente performance sur l'ensemble de validation, avec une accuracy remarquable de 99.41%, indiquant que presque toutes les prédictions étaient correctes. La précision était également très élevée à 99.37%, signifiant que la majorité des cas classés comme positifs étaient réellement positifs. Le rappel, à 99.89%, montre que le modèle a identifié avec succès presque tous les cas positifs réels. Le score F1 élevé de 99.63% révèle un équilibre parfait entre précision et rappel. La matrice de confusion confirme ces résultats avec un très faible nombre de faux positifs et un seul faux négatif.


Ensuite, nous avons tenté d'ajuster les paramètres de notre réseau de neurones. Malheureusement, nous avons rencontré plusieurs difficultés sans parvenir à trouver de solution. L'erreur qui est apparue concerne la fonction de perte que nous avons spécifiée comme étant 'binary_crossentropy'. Malgré nos efforts avec différentes approches, nous n'avons pas obtenu les résultats escomptés. En conséquence, nous allons effectuer une recherche par grille sur notre meilleur modèle SVM pour évaluer s'il permet d'améliorer nos prédictions et comment il classe notre variable cible.


### 3- Grid Search sur le meilleur modèle SVM


L'approche GridSearch, souvent utilisée en apprentissage automatique, est une méthode pour sélectionner les meilleurs paramètres pour un modèle donné. Cette technique utilise la création d'une “grille" de paramètres possibles pour le modèle. Chaque combinaison de paramètres dans cette grille est ensuite évaluée et comparée.
Dans les modèles SVM, C et gamma sont des paramètres permettant l’ajustement du modèle. Le paramètre C contrôle la régularisation et influence la complexité du modèle. Une valeur plus faible de C mène à une simplification du modèle, limitant le surajustement mais risquant le sous-ajustement, tandis qu'une valeur élevée permet une meilleure adaptation aux données d'entraînement mais avec un risque accru de surajustement. 
D'autre part, gamma détermine l'impact de chaque point de données sur la formation du modèle. Un gamma élevé augmente l'influence de chaque point, menant potentiellement à un modèle trop spécifique aux données d'entraînement (surajustement), alors qu'un gamma faible diminue cette influence, favorisant ainsi la généralisation du modèle.

<div align="center">

| Paramètre | Valeur |
|-----------|--------|
| C         | 10     |
| gamma     | 0.1    |

</div> 


Cela indique, comme dit précédemment qu'une régularisation modérée combinée à une influence relativement faible de chaque point de données produit les meilleurs résultats pour notre modèle. Regardons à présent l’évolution des métriques.

<div align="center">

| Metric    | Class 0 | Class 1 |
|-----------|---------|---------|
| Precision | 0.99    | 0.99    |
| Recall    | 0.97    | 1.00    |
| F1-score  | 0.98    | 0.99    |

</div>


Le modèle a démontré une excellente performance sur l'ensemble de test, atteignant une exactitude globale de 99.16%. Dans le détail, pour la classe 0, il a montré une précision de 99%, un rappel de 97%, et un score F1 de 98%, indiquant une grande précision et une bonne capacité à identifier correctement cette classe. Pour la classe 1, le modèle a été encore plus performant, avec une précision et un rappel de 99% et 100% respectivement, aboutissant à un score F1 de 99%. Ces résultats soulignent non seulement la capacité du modèle à distinguer avec précision les deux classes, mais aussi son équilibre remarquable entre précision et rappel, comme reflété dans les scores F1 élevés et les moyennes globales proches de 99%.



### 4- Importance des features

Pour finir cette analyse, il nous a semblé pertinent d’étudier l’importance des variables, pour ce faire nous avons utilisé une méthode de permutation.

La méthode de permutation permet d’évaluer l'importance des caractéristiques dans un modèle SVM. Pour ce faire, on mélange aléatoirement les valeurs d'une caractéristique dans l'ensemble de test et observe l'effet sur la performance du modèle. Une baisse notable de la performance après la permutation indique que la caractéristique est importante. En répétant cette opération plusieurs fois et en moyennant les résultats, on obtient une estimation fiable de l'importance de chaque features. Le graphique à boîtes illustre cette importance, permettant d'identifier rapidement les caractéristiques essentielles au modèle.

<div align="center">

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/7df308b3-c7a7-4c65-83b6-39b5c4a5a94a)

</div> 

En observant le graphique généré, nous pouvons interpréter l'importance des différentes caractéristiques dans la détermination du type de vin. Les caractéristiques situées vers le haut, avec des valeurs d'importance plus élevées sur l'axe des x, sont celles qui influencent le plus le modèle. Par exemple, les "chlorides" et le "total sulfur dioxide" semblent être des indicateurs très influents avec des boîtes s'étendant plus loin sur l'axe des x. Cela suggère qu'une permutation de ces valeurs a un impact notable sur la capacité du modèle à différencier les vins rouges des blancs.
En revanche, les caractéristiques en bas du graphique, telles que "quality_5" et "quality_6", ont des boîtes qui sont très proches de l'origine sur l'axe des x, indiquant que la permutation de ces caractéristiques a peu ou pas d'effet sur la performance du modèle. Nous concluons donc que ces caractéristiques sont potentiellement moins importantes pour la prédiction du type de vin.


# VI. Conclusion


En conclusion de notre projet visant à prédire la qualité et le type de vin, nous avons mis en place une série de modèles de machine learning pour atteindre nos objectifs.

Nous avons choisi de supprimer les variables fortement corrélées entre elles, car elles n'apportaient pas d'information significative lorsqu'elles étaient toutes deux impliquées dans nos modèles. Toutefois, même en présence d'une dépendance entre les variables qualité et type, nous avons décidé de les conserver toutes les deux afin d'optimiser la précision de nos modèles.

Dans notre démarche, nous avons standardisé les variables quantitatives et transformé les variables qualitatives comportant plus de deux modalités. Dans un premier temps, nous avons exploré plusieurs modèles de classification multiclasse, notamment deux modèles SVM (OVO et OVR) et un réseau de neurones. Dans ce contexte, le modèle OVO s'est révélé le plus performant et le mieux adapté pour prédire la qualité du vin.

Par ailleurs, nous avons testé différents algorithmes de classification binaire, tels que la régression logistique, plusieurs variantes de SVM (avec différents noyaux tels que linéaire, polynomial, rbf) ainsi qu'un réseau de neurones. Concernant nos modèles SVM, celui avec un noyau rbf s'est avéré être le plus performant. Le réseau de neurones, quant à lui, a affiché une performance supérieure à tous les modèles SVM, bien que nous n'ayons pas réussi à le tuner. Nous avons donc optimisé le modèle SVM avec un noyau rbf pour le grid search, obtenant des résultats presque similaires à ceux du modèle avec le noyau kernel sans grid search. En fin de compte, le modèle de réseau de neurones s'est avéré être le meilleur pour classifier le type de vin, même sans ajustement fin.



# VII. Discussion 

Au cours de cette étude, nous avons été confrontés à divers défis techniques et méthodologiques qui ont influencé notre approche et nos résultats finaux. 

Tout d'abord, aucune sélection de variables n'a été effectuée. Il est courant d'utiliser des techniques de sélection de variables pour réduire la dimensionnalité et améliorer l'interprétabilité des modèles. Cependant, dans notre cas, compte tenu du nombre relativement limité de variables, nous avons choisi de ne pas effectuer de sélection de variables.

Un autre défi a été le réglage des hyperparamètres pour nos réseaux de neurones. Malgré plusieurs tentatives, nous avons rencontré des difficultés avec le code d'optimisation, ce qui a généré des messages d'erreur indiquant que la fonction de perte n'était pas interprétable pour nos données. L'absence de cette étape aurait pu limiter la capacité de notre modèle à généraliser au-delà de l'échantillon d'entraînement. De plus, nous n'avons pas effectué de grid search sur nos modèles de classification multiclasse en raison du temps considérable que cela aurait demandé.







