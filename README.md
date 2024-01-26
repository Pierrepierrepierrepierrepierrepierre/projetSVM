<h1 style="text-align:center; color:red; font-weight:bold;"># Prédiction et modélisation de la qualité et du type de vin </h1>


## Introduction : 

Dans le cadre de notre projet de Machine Learning et de Deep Learning, nous nous sommes lancés dans l'analyse approfondie d'une base de données portant sur la qualité et le type de vins. Cette base, récupérée sur le site de référence Kaggle et créé par un auteur du nom de Raj Parmar, inclut des informations détaillées sur un vaste éventail de vins, rouges comme blancs.
Notre objectif est double : prédire la qualité du vin, évaluée sur une échelle de 0 à 10, et classifier le vin en tant que rouge ou blanc. Cette tâche s'appuie sur l'exploitation de divers paramètres chimiques et physiques des vins, tels que l'acidité, le sucre résiduel, les chlorures, le sulfite, la densité, le pH, les sulfates, et la teneur en alcool. 
La première phase de notre projet a consisté en une compréhension approfondie de la base de données, suivie d'un nettoyage rigoureux et d'une préparation des données, incluant la gestion des valeurs manquantes et la création d'indicateurs pertinents. Cette étape essentielle a permis d'établir des fondations solides pour nos analyses et modélisations ultérieures.
Dans un deuxième temps, nous avons exploré les distributions et les corrélations des variables explicatives avec la qualité et le type de vin. Cette analyse exploratoire est cruciale pour formuler des hypothèses sur les facteurs ayant une influence sur ces derniers. Finalement, nous avons développé et testé une série de modèles de prédiction visant à évaluer la qualité et à déterminer le type de vin. 

Description du dataset
Nous avons exploité une base de données disponible sur le site Kaggle, spécialisée dans les caractéristiques et la qualité des vins rouges et blancs. Cette base de données, riche en informations, nous permet d'analyser en profondeur les différents aspects influençant la qualité du vin. La base de données initiale comprend des enregistrements sur un grand nombre de vins, couvrant diverses variables chimiques et physiques.
1. Description des variables
Type : Cette variable indique si le vin est rouge ou blanc.
Fixed Acidity : L'acidité fixe du vin.
Volatile Acidity : L'acidité volatile, qui influence l'arôme et le goût du vin.
Citric Acid : Le niveau d'acide citrique, un facteur important dans la saveur du vin.
Residual Sugar : La quantité de sucre résiduel après la fin de la fermentation.
Chlorides : La concentration en chlorures, affectant le goût salé du vin.
Free Sulfur Dioxide : Quantité de dioxyde de soufre libre, jouant un rôle dans la prévention de l'oxydation et la croissance de microbes.
Total Sulfur Dioxide : Quantité totale de dioxyde de soufre, un indicateur important pour la conservation du vin.
Density : La densité du vin, liée à son taux d'alcool et de sucre.
pH : Mesure de l'acidité ou de la basicité du vin.
Sulphates : Niveau de sulfates, influençant la fermentation et le goût du vin.
Alcohol : Le pourcentage d'alcool dans le vin.
Quality : La note de qualité attribuée au vin, sur une échelle de 0 à 10.

Analyse exploratoire 
Analyse univariée

variables dépendantes: 

Répartition des données : 

Variable type : 
Le graphique ci dessus nous montre la répartition de deux types de vins. On observe une répartition assez inégale, en effet, le vin blanc représente une part beaucoup plus importante, avec 75.39%, tandis que le vin rouge est moins fréquent à 24.61%.

variable qualité : 
Pour ce qui est de la variable qualité, nous avons préféré un histogramme en barre pour analyser la répartition. Les histogrammes en barres offrent une comparaison et une interprétation plus claires des fréquences lorsque l’on analyse des variables catégorielles. 

Ainsi, nous observons une distribution asymétrique où la majorité des vins se concentrent autour des notes de qualité moyenne correspondant à une loi normale, ce qui est logique  puisque la plupart des vins reçoivent des notes moyennes et moins reçoivent des notes extrêmes.

De ce fait, nous devrons procéder à un rééquilibrage des classes afin d’avoir des modèles pertinents  


variables explicatives : 

Statistiques descriptives de nos variables explicatives 

Dans notre étude de plus de 6400 vins, nous avons remarqué quelques tendances intéressantes concernant leurs caractéristiques. Certains aspects des vins sont assez similaires d'une bouteille à l'autre, tandis que d'autres varient beaucoup.
Premièrement, il y a des éléments comme la densité et le pH où la plupart des vins sont assez semblables. La densité des vins ne change pas beaucoup, ce qui signifie que la "lourdeur" ou la "légèreté" du vin en termes de poids est presque la même pour tous. Le pH est également assez constant, indiquant que l'équilibre acide-basique ne varie pas trop d'un vin à l'autre.
En revanche, le sucre résiduel, qui est le sucre restant après la fermentation, et le dioxyde de soufre, utilisé pour conserver le vin, montrent beaucoup plus de différences entre les vins. Certains vins sont beaucoup plus sucrés que d'autres, et la quantité de dioxyde de soufre varie également beaucoup. Cela nous donne une idée de la diversité des goûts et des méthodes de fabrication des vins.
D'autres caractéristiques comme l'acidité, le niveau de certains acides (comme l'acide citrique), les chlorures (qui influencent le goût salé), les sulfates (utilisés aussi pour la conservation) et l'alcool ont des valeurs plus équilibrées. 
Pour conclure, notre étude montre qu'il y a beaucoup de similitudes dans certains aspects des vins, mais aussi une grande variété dans d'autres. Cela reflète la complexité du vin et la façon dont différents ingrédients et méthodes de fabrication peuvent influencer le goût final. 

Analyse bivariée
Nous n’allons pas réaliser cette analyse sur l’ensemble de nos variables, en effet, étant donné que nous avons 2 variables targets différentes, cela serait trop long. Ainsi, nous avons sélectionné 4 variables explicatives qui semblent pertinentes par rapport à nos variables qualité et type. Il s’agit des variables Alcool, Acidité Volatile, Sucre Résiduel et Chlorides. Ces variables ont été choisies car elles jouent un rôle crucial dans la détermination des caractéristiques sensorielles et de la conservation du vin.


Variable quality : 

Alcool
Le graphique montre que les vins de meilleure qualité tendent à avoir une teneur en alcool plus élevée, avec une augmentation progressive de l'alcool allant des vins de qualité inférieure aux vins de qualité supérieure. Il y a cependant une exception à cette tendance avec les vins notés 6, qui ont une teneur en alcool légèrement plus basse que ceux notés 5. Les barres d'erreur indiquent une variabilité similaire dans la teneur en alcool à travers les différentes qualités de vin, à l'exception des vins de qualité 9, qui montrent une plus grande variabilité.

Acidité volatile 
Le graphique indique que les vins de qualité supérieure ont généralement une acidité volatile plus basse. On observe également une certaine stagnation à partir des vins ayant une qualité supérieure à 5. Ainsi, une acidité volatile élevée serait gage de mauvais vin, alors qu’une acidité volatile plutôt faible ne nous permettrait pas de statuer entre un moyen ou un bon vin.
Cependant, les vins avec les notes de qualité les plus élevées et les plus basses montrent une plus grande variabilité dans l'acidité volatile que ceux de qualité moyenne. Cela suggère que l'acidité volatile est un indicateur clé de la qualité, où moins d'acidité volatile correspond à une meilleure qualité perçue.
Sucre résiduel : 
Il n'y a pas de tendance claire reliant la qualité du vin au sucre résiduel; les vins de qualité moyenne et supérieure ont des niveaux de sucre résiduel similaires. On note une grande variabilité dans le sucre résiduel pour les vins de toutes les qualités, en particulier pour les vins notés 9, indiquant que la douceur peut varier considérablement au sein d'une même catégorie de qualité. Ainsi, on peut conclure que le sucre résiduel n'est pas un indicateur direct de la qualité du vin.
Chlorides
On observe que les vins de qualité inférieure ont des concentrations plus élevées en chlorides, tandis que les vins de qualité supérieure ont tendance à en avoir moins. La concentration en chlorides diminue globalement à mesure que la qualité augmente. La variabilité des concentrations en chlorides semble diminuer également avec l'augmentation de la qualité, particulièrement visible pour les vins de qualité 9. Cela nous permet de dire que les vins mieux notés ont une composition plus cohérente en termes de chlorides.

Variable type : 


total sulfure dioxyde 

Le graphique montre que les vins blancs ont des niveaux de dioxyde de soufre total nettement plus élevés que les vins rouges, avec une variabilité moindre dans les concentrations pour les vins rouges.

chlorides

Les vins rouges présentent des concentrations plus élevées en chlorides par rapport aux vins blancs. En moyenne, sur notre échantillon, on note presque deux fois plus de chlorides pour un vin rouge par rapport à un vin blanc.

residual sugar

Pour ce qui est de la teneur en sucre, on remarque un taux plus de 2 fois supérieur pour les vin blanc par rapport aux vins rouges.


alcool
Enfin, pour le taux d’alcool contenu dans les vins, on ne remarque aucune différence, signifiant que le taux d’alcool ne dépend pas du type de vin. Ainsi, notre modèle ne pourra pas se baser sur cette variable pour reconnaître le type de vin.

graphique répartitions type par qualité : 

Le graphique à barres présenté détaille la distribution des notes de qualité pour les deux catégories de vins. Il ressort que la note '5' est la plus commune pour les rouges, tandis que les blancs sont le plus souvent notés '6', révélant une qualité perçue légèrement meilleure pour ces derniers. Les données visuelles suggèrent que les vins blancs tendent à être évalués plus favorablement que les rouges comme en témoignent les barres plus élevées pour les notes '6', '7', et '8' par rapport aux vins rouges.



KHI 2

Une étape clé est d'examiner la relation entre la qualité du vin et son type (rouge ou blanc) afin de vérifier qu’elles ne sont pas dépendantes.
Pour cela, un test statistique de Chi2 a été utilisé, qui est un outil standard pour évaluer si deux variables catégorielles sont indépendantes l'une de l'autre ou non.
Le test de Chi2 appliqué aux données a révélé des résultats significatifs. Avec une valeur de Chi2 de 117.03 et une p-valeur extrêmement faible (approximativement 6.86e-23), le test indique clairement que la qualité du vin et le type de vin ne sont pas indépendants.
Ainsi nous nous sommes demandé s’il ne fallait pas retirer les variables type et qualité dans les modèles visant à expliquer l'autre. Cependant, l'intégration de la variable "type" dans la modélisation de la qualité, et de la "qualité" pour prédire le type, est une approche qui nous semble malgré tout pertinente puisque cela permet d'exploiter pleinement les données disponibles et in fine d’augmenter la précision de la prédiction de notre modèle, ce qui est l’objectif de ce dossier.

Distribution des features

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




Analyse multivariée
corrélation 
A présent, nous concentrons notre analyse sur les corrélations entre nos différentes variables afin d’éliminer les variables inutiles à notre modélisation  en raison de l’une multicolinéarité potentielle. Pour ce faire nous analysons la matrice des corrélation.
Étant donné qu’une matrice des corrélations classiques n’est pas très visible, nous en réalisons une ou seulement les corrélation supérieure à 0,65 sont affichées. 
Nous obtenons donc 2 groupes de variables corrélées entre-elles, à savoir sulfure dioxyde libre et sulfure dioxyde total avec une corrélation de 0,74 ainsi que les variables densité et alcool qui sont corrélées à -0,7.

Pour choisir quelles variables nous allons garder, nous nous intéressons aux diagrammes en barres de l’analyse bivariée. 
Premièrement pour les variables liées au sulfure, on ne remarque pas de différences significatives par rapport à la variable qualité. Cependant, pour la variable type on note que la variable sulfure dioxyde total marque une plus grande différence par rapport au type de vin la rendant plus intéressante pour le déterminer par la suite dans nos modèles. Ainsi, c’est la variable que nous retiendrons.
Ensuite, nous procédons à la même analyse pour nos variables densité et alcool. Par rapport au type de vin, on observe qu'il n'y a aucune différence pour la densité ainsi que pour la teneur en alcool. Nous nous intéressons alors à la variable. La densité ne semble pas non plus impacter la qualité d’un vin puisque l’on observe aucune différence de valeur par rapport aux notes de qualité. Par ailleurs, pour ce qui est de la teneur en alcool, on remarque assez facilement une augmentation progressive de celle-ci à mesure que la qualité augmente pour les notes supérieures à 5. Ainsi il semble plus pertinent de conserver la variable alcool pour nos modèles puisqu’elle permet de différencier davantage les vins que la variable densité qui s'avère moins intéressante en raison de sa faible variabilité en fonction du type et de la qualité. 

Préparation de la BDD
Suite à l'examen détaillé des variables individuelles et à l'étude des liens qu'elles entretiennent avec nos deux variables cibles, l'étape suivante consiste à affiner notre base de données. Cette phase de préparation implique un nettoyage, en effet,  les données manquantes seront écartées et les points atypiques, susceptibles de fausser nos modèles prédictifs, seront corrigées. Ainsi, notre base sera prête pour la construction de  différents modèles prédictifs.





