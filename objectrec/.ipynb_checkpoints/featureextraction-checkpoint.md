# Mimikcodierung und Feature Extraction

Gerade im Hinblick auf die Emotionserkennung ist die Codierung der Gesichtsausdrücke wichtig, da hier auch die Extraktion der relevanten Features ansetzt.

<!-- #region -->
## Facial Action Coding System  (FACS)

Ekman und Friesen entwickelten 1976 ein Codierungssystem der menschlichen Gesichtsbewegungen, welches als Facial Action Coding System (FACS) bezeichnet wird. Die unterschiedlichen Muskelaktionen im Gesicht werden zu 58 Action Units (AUs) zusammengefasst, an denen jeweils auch mehrere Muskeln beteiligt sein können. Dabei bezogen sie sich auf die Arbeit von Hjortsjö (1970), welcher die Änderungen auf der Gesichtsfläche (z.B. Faltenbildung, Mundstellungen) die durch Kontraktionen der 23 Gesichtsmuskeln bewirkt werden, genau beschrieb und durchnummerierte.

![Action Units](./images/AUs.jpg)

Die mimischen Gesichtsausdrücke der Grundemotionen und zusammengesetzten Emotionen lassen sich durch eine Kombination von AUs präzise und nachvollziehbar beschreiben. 
AUs werden dabei auf einer 5-stufigen Intensitätsskala bewertet.

Kodierung:
   * Stärke A: an der Wahrnehmungsgrenze oder angedeutet
   * Stärke B: gut sichtbar
   * Stärke C: deutlich sichtbar
   * Stärke D: ausgeprägt
   * Stärke E: im physiologischen Höchstmaß (individuell)
   
Die Ausprägungsstärke ist dabei nicht gleichbedeutend mit der Ausdrucksstärke, da die Wirkung eines Ausdrucks nicht allein von der Stärke der Muskelbewegung abhängt. 
 

**Kritik**

* beschäftigt sich nur mit deutlich sichtbaren Veränderungen in der Gesichtsbewegung
* ausschließlich für die Messung von Bewegungen im Gesicht entwickelt, sodass andere Gesichtsphänomene (z. B. Änderungen der Hautfärbung, Schwitzen, Tränen, etc.) nicht berücksichtigt werden


<!-- #endregion -->

## Facial Landmarks (FL)

Gesichtslandmarken sind definiert als die Erkennung und Lokalisierung bestimmter Punktmerkmale auf dem Gesicht. Häufig verwendete Orientierungspunkte sind die Augenwinkel, die Nasenspitze, die Nasenlochwinkel, die Mundwinkel, die Endpunkte der Augenbrauenbögen, Ohrläppchen, Kinn usw. 

Es ist bekannt, dass Landmarken wie Augenwinkel oder Nasenspitze nur wenig von der Mimik beeinflusst werden, daher sind sie zuverlässiger und werden in der Literatur zur Gesichtsverarbeitung als Referenzpunkte oder Referenzlandmarken bezeichnet.

![Beispiel für Facial landmarks](./images/FacialLandmarks.png)

Unterteilung der Orientierungspunkte:
* primäre Orientierungspunkte (Augen, Mundwinkel, Nasenspitze, Augenbrauen): Vor allem für die Gesichtserkennung wichtig
* sekundären Orientierungspunkte (Kinn, Wangenkonturen, Augenbrauen- und Lippenmittelpunkte): Vor allem für die Erkennung des Gesichtsausdrucks relevant 

![Beispiel für Facial landmarks in verschiednene Datensätzen](./images/facialLandmarksDatasets.jpg)


## Feature Extraction
Die oben erläuterten Ansätze sind nur ein kleiner Einblick in die Verfahren und Modelle, die verwendet werden, um Gesichtsausdrücke zu codieren. 

Weitere Beispiele sind:
* Principal Component Analysis (PCA)
* Local Binary Patterns (LBP)
* Geometrie basierte Methoden (Facial Landmarks)
* Gabor feature extraction