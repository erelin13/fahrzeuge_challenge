[EN]

### Computer Vision Challange

To solve the task on the meer base level we should at least to benchmark the easiest way, so to say, define a starting point, from which in the future we can more further, improve model and/or algorithm and comapre it with the easiest benchmark.
<br> That is why as a first solution I used a well-known, "old but gold" model architecture ResNet-50 as the main backbone (I only chnaged last layers to suit our task).
<br> In the future we could either work with a benchmark model (learning rate scheduler application, custom loss functions development etc.) or try another model.
<br> If we would like to optimize any model on inference time, we could also perform experimentations with half-presicion or model weights prunging, as well as other optimization techniques, f.i. converting model to ONNX format.

To be continued

[DE]

Um die Aufgabe auf der Basisebene zu lösen, sollten wir zumindest den einfachsten Weg vergleichen, also einen Ausgangspunkt definieren, von dem aus wir in Zukunft das Modell und/oder den Algorithmus weiter verbessern und mit dem einfachsten vergleichen können Benchmark.
<br> Aus diesem Grund habe ich als erste Lösung eine bekannte, „alte, aber goldene“ Modellarchitektur ResNet-50 als Haupt-Backbone verwendet (ich habe nur die letzten Schichten geändert, um sie an unsere Aufgabe anzupassen).