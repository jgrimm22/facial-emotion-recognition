# FER Ansätze

## Konventioneller Ansatz

1. Bildvorverarbeitung
2. Extrahieren von Merkmalen 
3. Klassifizierungen 
* Geringere Rechenleistung und Speicherbedarf als Deep-Learning-basierte Ansätze
* Weniger abhängig von Daten und Hardware
* Merkmalsextraktion und die Klassifizierung müssen manuell und separat entwickelt werden, was bedeutet, dass diese beiden Phasen nicht gleichzeitig optimiert werden können

Konventionelle FER-Verfahren kann in drei Hauptschritte unterteilt werden:

![Konventioneller FER-Ansatz](./images/conventional_model.png)
### 1. Image Processing
Das Ziel des Image Processing ist die Eliminierung irrelevanter Information aus dern Eingabebildern und verbesserern der Erkennungsfähigkeit relevanter Informationen. 
* Kann sich direkt auf die Extraktion von Merkmalen und die Leistung der expression Classification auswirken
* Bilder können aus verschieden Gründen durch andere Signale „verunreinigt“ sein (komplexe Hintergründe, Lichtintensität, Verdeckung etc.)

Image Preprocessing Prozesse sind: 
* Noise reduction: Average Filter (AF), Gaussian Filter (GF ), Median Filter (MF), Bilateral Filter (BF)
* Face detection: Gesichtserkennung hat sich zu einem eigenständigen Gebiet entwickelt, Vorstufe in FER mit dem Ziel, die Gesichtsregion zu lokalisiern und zu extrahieren 
* Normalisierung der Skala und der Graustufen: Normalisierung von Größe und Farbe der Eingabebilder, mit dem Ziel Berechnungskomplexität zu reduzieren unter der Prämisse, die wichtigsten Merkmale des Gesichts zu erhalten
* Histogramm-Entzerrung: Verbesserung der Bildwirkung 

### 2.	Feature Extraction:
Die Feature Extraction ist der Prozess zur Extraktion nützlicher Daten oder Informationen aus dem Bild, z.B. Werte, Vektoren und Symbole.

* “Nicht-Bild“-Darstelllungen/Beschreibungen sind Features des Bildes 
* Methoden: 
    * Gabor feature extraction
    * Local Binary Pattern (LBP)
    * Optical Flow Method
    * Haar-Like Feature Extraction
    * Feature Point Tracking
    * ...
* Kann sich direkt auf die Leistung der Algorithmen auswirken

### 3. Expression Classifier
Der Gesichtsausdruck wird basierend auf eine der Gesichtskategorien bestimmt. Die Kategorien werden mit Hilfe von Pattern Classifiers vortrainiert.

* Weitverbreite Classifier: 
    * kNN (k-Nearest Neighbours )
    * SVM (Support Vector Machine)
    * Adaboost (Adaptive Boosting)
    * Bayessches Netz
    * SRC (Sparse Representation-based Classifier)
    * PNN (Probabilistic Neural Network)





<!-- #region -->
## Deep-Learning-basierter Ansatz
* Deep-Learning-basierte Algorithmen für die Merkmalsextraktion, Klassifizierung und Erkennungsaufgaben
* Bei vielen Aufgaben des maschinellen Lernens hervorragende Leistungen gezeigt einschließlich Identifizierung, Klassifizierung und Zielerkennung
* Bei FER: 
    * Reduzierung der Abhängigkeit von der Bildvorverarbeitung und Merkmalsextraktion 
    * Robust gegenüber Umgebungen mit unterschiedlichen Elementen, z.B. Beleuchtung und Verdeckung
    * Deep-Learning-basierte Ansätze übertreffen konventionelle Ansätze mit einem Durchschnitt von 72,65% gegenüber 63.2%
* Fähigkeit, große Datenmengen zu verarbeiten

Mögliche Deep-Learnig Netze:


### Convolutional Neural Network (CNN)
* State-of-the-art in FER
* "End-to-End"-Modell: Lernen direkt von den Eingabedaten zum Klassifikationsergebnis 
* Merkmale:
    * Weniger Netzwerkparameter durch lokale Konnektivität und die gemeinsame Nutzung von Gewichten 
    * Schnelle Trainingsgeschwindigkeit und Regularisierungseffekt
* CNN enthält drei Arten von heterogenen Schichten:
    * Convolutional Layer: Eingabebilder werden mit Hilfe von Filtern gefaltet und es wird eine Feature Map erzeugt
    * Max Pooling Layer: Maxpooling-Layers (Subsampling) senken die räumliche Auflösung der gegebenen Feature Maps
    * Fully Connected Layer: berechnen die Klassen-Scores auf dem gesamten Originalbild und ein einzelner Gesichtsausdruck wird basierend auf der Ausgabe von Softmax-Algorithmus erkannt

![CNN FER-Ansatz](./images/CNN_model.png)




### Long Short-Term Memory (LSTM)
* Art von RNN (Recurrent Neural Networks), das aus LSTM-Einheiten besteht
* Lösen das Problem der verschwindenden Gradienten, welches in RNNs vorkommt
* Geeignet für die zeitliche Merkmalsextraktion von aufeinanderfolgenden Frames 
* LSTM-Netz hat drei Gates, die die Zellzustände aktualisieren und steuern
    1. Forget-Gate: steuert, welche Informationen im Zellzustand vergessen werden sollen, wenn neue Informationen in das Netzwerk gelangen 
    2. Input-Gate: steuert, welche neuen Informationen in den Zellzustand kodiert werden, wenn die neuen Input-Informationen vorliegen 
    3. Output-Gate: steuert, welche im Zellzustand kodierte Information im folgenden Zeitschritt als Input an das Netzwerk gesendet wird
![LSTM](./images/LSTM.gif)
* Gates verwenden hyperbolische Tangens und Sigmoide Aktivierungsfunktionen

* Durch Vorhandensein der Aktivierungen des Forget-Gates wird dem LSTM ermöglicht, bei jedem Zeitschritt zu entscheiden, dass bestimmte Informationen nicht vergessen werden sollen und die Parameter des Modells entsprechend zu aktualisieren
* Mögliche Einsatzgebiet: 3D Convolutional Neural Network (3DCNN)
    * LSTM berücksichtigt zeitlichen Relationen und verwendet diese Informationen zur Klassifizierung der Sequenzen
    

 


<!-- #endregion -->

### Weitere Modelle

Es gibt viele Ansätze, die auf einem eigenständigen CNN oder einer Kombination aus LSTM und CNN basieren:
* 3D Convolutional Neural Network (3DCNN)
* 3D Inception-ResNet
* DRML (Deep Region und Multi-Label-Learning)
* Candide-3
* Multi-Angle FER
* Hybrid CNN-RNN

