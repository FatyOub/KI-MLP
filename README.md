# KI-MLP

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
