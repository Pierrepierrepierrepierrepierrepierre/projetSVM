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

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e360b141-ea74-46ae-8b30-d48ba316b6a6)

A présent, nous concentrons notre analyse sur les corrélations entre nos différentes variables afin d’éliminer les variables inutiles à notre modélisation  en raison de l’une multicolinéarité potentielle. Pour ce faire nous analysons la matrice des corrélation.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/959f7e13-d62e-4cba-ac76-669bf127fe89)


Étant donné qu’une matrice des corrélations classiques n’est pas très visible, nous en réalisons une ou seulement les corrélation supérieure à 0,65 sont affichées. 
Nous obtenons donc 2 groupes de variables corrélées entre-elles, à savoir sulfure dioxyde libre et sulfure dioxyde total avec une corrélation de 0,74 ainsi que les variables densité et alcool qui sont corrélées à -0,7.


Pour choisir quelles variables nous allons garder, nous nous intéressons aux diagrammes en barres de l’analyse bivariée. 
Premièrement pour les variables liées au sulfure, on ne remarque pas de différences significatives entre les 2, par rapport à la variable qualité comme nous pouvons le voir sur le graphique ci-dessous

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/ed4d08d2-6124-42b9-92b9-e8c5aea79d87)
![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/f76ba96e-6078-4cfe-95cf-bb3af8c99938)


Cependant, pour la variable type on note que la variable sulfure dioxyde total marque une plus grande différence par rapport au type de vin la rendant plus intéressante pour le déterminer par la suite dans nos modèles. Ainsi, c’est la variable que nous retiendrons.

![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/e60e2ee7-7f5b-45a0-a0d2-853f8b604988) ![image](https://github.com/Pierrepierrepierrepierrepierrepierre/projetSVM/assets/124379009/4f8d92e7-2c6a-4d26-b19a-cb317f0bc51d)





Ensuite, nous procédons à la même analyse pour nos variables densité et alcool. Par rapport au type de vin, on observe qu'il n'y a aucune différence pour la densité ainsi que pour la teneur en alcool. Nous nous intéressons alors à la variable. La densité ne semble pas non plus impacter la qualité d’un vin puisque l’on observe aucune différence de valeur par rapport aux notes de qualité. Par ailleurs, pour ce qui est de la teneur en alcool, on remarque assez facilement une augmentation progressive de celle-ci à mesure que la qualité augmente pour les notes supérieures à 5. Ainsi il semble plus pertinent de conserver la variable alcool pour nos modèles puisqu’elle permet de différencier davantage les vins que la variable densité qui s'avère moins intéressante en raison de sa faible variabilité en fonction du type et de la qualité. 


