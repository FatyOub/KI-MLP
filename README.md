# KI-MLP

# NN.MLP.01: Perzeptron-Netze

P1:

   x_1-Gewicht : 0

   x_2-Gewicht : 1

   Bias : -3

   Funktion x_2 >= 3

    Ausgabe: h_1 = sign(0 * x_1 + 1 * x_2 - 3).
   
P2:

   x1-Gewicht : 1

   x2-Gewicht : 0

   Bias       : -2

   Funktion   :  x_1 >= 2

   Ausgabe:   h_2 = sign(1 * x_1 + 0 * x_2 - 2).

P3:

   x1-Gewicht : 1

   x2-Gewicht  : 1

   Bias        : 1

   Funktion    : ODER VERknüpfung 

   Ausgabe : y = sign(1 * h_1 + 1 * h_2 + 1).
   
# NN.MLP.02: Vorwärtslauf im MLP 

# die Dimensionen der Gewichtsmatrizen und der Bias-Vektoren:

schicht 1:

     verbindung : Input (25) -> Hidden 1 (64)

     Gewichtsmatrix W :  64 * 25 

     Bias-Vektor b : 64 * 1

schicht 2:

    verbindung : Hidden 1 (64) -> Hidden 2 (32)

     Gewichtsmatrix W : 32 * 64

     Bias-Vektor b : 32 * 1

schicht 3:

    verbindung : Hidden 2 (32) -> Output (4)

     Gewichtsmatrix W : 4 * 32

     Bias-Vektor b : 4 * 1


#  Matrix-Notation 

Der Vorwärtslauf berechnet die Aktivierungen jeder Schicht sukzessive. Als Aktivierungsfunktion wird für alle Schichten die ReLU-Funktion (ReLU(z) = max(0, z)) verwendet. 

Sei a^[0] = x der Eingabevektor:

Erste versteckte Schicht:

z^[1] = W^[1] * a^[0] + b^[1]

a^[1] = ReLU(z^[1])

Zweite versteckte Schicht:

z^[2] = W^[2] * a^[1] + b^[2]

a^[2] = ReLU(z^[2])

Ausgabeschicht:

z^[3] = W^[3] * a^[2] + b^[3]

y = ReLU(z^[3])

Die Ausgabe besteht aus 4 Werten. Da es sich um ein Klassifikationsnetzwerk handelt, repräsentiert jedes Neuron in der Ausgabeschicht eine bestimmte Klasse. Ein hoher Wert an einem Ausgang deutet darauf hin, dass das Netzwerk die Eingabe dieser spezifischen Kategorie zuordnet.

Das Netzwerk ist aufgrund der zwei versteckten Schichten in der Lage, nichtlineare Zusammenhänge zu modellieren und somit komplexe Klassifikations- oder Regressionsprobleme zu lösen.


# NN.MLP.03: Tensorflow Playground:

# 1. logistisches Regressionsmodell

Zunächst wurde ein logistisches Regressionsmodell ohne versteckte Schichten trainiert.
Auf dem Gaussian-Datensatz  konnte eine klare lineare Entscheidungsgrenze gelernt werden. Die Trainings- und Testkosten sanken schnell und blieben stabil auf einem niedrigen Niveau. Eine Überanpassung war nicht zu beobachten, da Modellkomplexität und Datenstruktur gut zusammenpassen.

Auf nicht-linear separierbaren Datensätzen wie Circle und Spiral war das Modell hingegen nicht in der Lage, eine sinnvolle Entscheidungsgrenze zu finden. Die Entscheidungsgrenze blieb nahezu linear, viele Datenpunkte wurden falsch klassifiziert und sowohl Trainings- als auch Testkosten blieben hoch. Dies zeigt die grundsätzliche Limitation logistischer Regression bei komplexen Datenstrukturen.

# 2. MLP:

Mit zunehmender Anzahl an Neuronen und Schichten wurde die Entscheidungsgrenze deutlich komplexer und besser an die Daten angepasst.

2–3 Neuronen (1 Schicht): nur grobe, unzureichende Approximation

5 Neuronen (1 Schicht): brauchbare Trennung bei Circle, Spiral weiterhin schwierig

2–3 Schichten: deutlich bessere Anpassung, insbesondere beim Spiral-Datensatz

4 Schichten mit 7 Neuronen: sehr komplexe Entscheidungsgrenzen, nahezu perfekte Trennung möglich

Einfluss der Aktivierungsfunktion

ReLU: schnellste Konvergenz, stabile Trainingskosten, klare Entscheidungsgrenzen

tanh: glatte, symmetrische Entscheidungsgrenzen, etwas langsamer

Sigmoid: langsamste Konvergenz, teilweise Sättigungseffekte, schlechtere Performance bei tiefen Netzen

Die Aktivierungsfunktion hat sowohl Einfluss auf die Form der Entscheidungsgrenze als auch auf die Trainingsgeschwindigkeit.

# 3. Noise-Level auf 15


Bei erhöhtem Noise-Level konnten komplexe Netzwerke die Trainingsdaten weiterhin sehr gut approximieren, jedoch stiegen die Testkosten deutlich an, während die Trainingskosten weiter sanken. Dies ist ein klares Zeichen für Überanpassung.

Überanpassung tritt insbesondere auf bei:

vielen Schichten

vielen Neuronen

sehr niedrigen Trainingskosten bei gleichzeitig hohen Testkosten

Einfachere Modelle generalisieren in diesem Fall besser, auch wenn sie die Trainingsdaten nicht perfekt klassifizieren.

# Alle Experimente:

Nicht alle Datenpunkte lassen sich in jedem Fall korrekt klassifizieren, insbesondere bei hohem Rauschanteil oder einer zu geringen Modellkomplexität. Während kleine Netzwerke in der Regel sehr schnell konvergieren, benötigen tiefere Netzarchitekturen aufgrund ihrer höheren Komplexität deutlich mehr Trainingszeit.
In den ersten versteckten Schichten reagieren die Neuronen überwiegend auf einfache, nahezu lineare Merkmale, etwa Trennungen entlang der x- oder y-Achse. In den tieferen Schichten hingegen bilden sich zunehmend komplexe Aktivierungsmuster heraus, die hochabstrakte Merkmale der Daten repräsentieren. Dies verdeutlicht die schrittweise Merkmalsextraktion, die charakteristisch für tiefe neuronale Netze ist.
