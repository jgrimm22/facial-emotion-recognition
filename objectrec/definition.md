# Was ist Facial Emotion Recognition?


```{admonition} Definition
Facial Emotion Recognition ist die Erkennung von Gesichtsausdrücken und die damit verbundenen Emotionen durch ein maschinelles Lernverfahren.
```

Die automatische Erkennung von Emotionen durch maschinelle Lernverfahren ist der Prozess menschliche Emotionen aus Signalen, wie Gesichtsausdrücken oder Sprache zu extrahieren und zu erkennen. 

Emotionen im Gesicht zu erkennen ist ein wichtiger Faktor in menschlicher Kommunikation, um die Absichten untereinander zu verstehen. Menschen schließen anhand von Gesichtsausdrücken und Stimmlagen auf die Gefühlszustände anderer Menschen. Das Interesse an automatischer Gesichtsemotionserkennung hat in den vergangenen Jahren mit der rasanten Entwicklung von KI-Techniken zugenommen.

Die Abkürzung FER wird sowohl für Facial Emotion Recognition als auch Facial Expression Recognition verwendet. Die beiden Begriffe werden in der Literatur häufig synonym verwendet. 


<!-- #region -->
## Aufbau FER


Die Facial Emotion Recognition setzt sich aus drei Komponenten zusammen, die egal welcher maschinelle Lernansatz verwendet wird, in irgendeiner Form benötigt werden. 

Diese Komponenten sind zum einen die **Gesichtserkennung** und die **Facial Component Detection**, wobei Gesichtskomponenten, (z.B. Nase und Augen) oder Facial Landmarks im Gesicht erkannt werden. Des Weiteren findet eine **Feature Extraction** statt, in der die wichtigen Mekamle eines Bildes extrahiert werden. Schießlich wird bei der **Klassifizierung** das Bild mit Hilfe der extrahierten Merkmale einer Emotion zugeordnet.   

![Komponenten](./images/fer_process.png)

Um ein gutes maschinelles Lernverfahren für die Facial Emotion Recognition zu entwickeln, ist es wichtig Emotionen und Mimik des Menschen zu verstehen und diese richtig zu beschreiben. Dies ist notwendig, um einer Emotion ein Label zuordnen zu können. Außerdem werden aussagekräftige Daten benötigt, um das FER-System trainieren zu können.

<!-- #endregion -->

## Agenda
Um eine Facial Emotion Recognition durchführen zu können müssen einige Themen beachtet werden. Im folgenden wird ein Überblick über die einzelnen Komponenten gegeben, die für die Realisierung eines FER-Models benötigt werden:
* Menschliche Kommunikation
* Feature Extraction
* Datensätze
* Modelle/FER-Ansätze
* Implementiertes Modell
* Kritik
* Fazit
