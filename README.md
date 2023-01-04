# Vestibular contribution to path integration deficits in ‘at-genetic-risk’ for Alzheimer’s disease

by
Gillian Coughlan,
William Plumb,
Peter Zhukovsky,
Min Hane Aung,
Michael Hornberger


This paper has been published in [PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278239).

## Abstract

Path integration changes may precede a clinical presentation of Alzheimer’s disease by several years. Studies to date have focused on how spatial cell changes affect path integration in preclinical AD. However, vestibular input is also critical for intact path integration. Here, we developed the vestibular rotation task that requires individuals to manually point an iPad device in the direction of their starting point following rotational movement, without any visual cues. Vestibular features were derived from the sensor data using feature selection. Machine learning models illustrate that the vestibular features accurately classified Apolipoprotein E ε3ε4 carriers and ε3ε3 carrier controls (mean age 62.7 years), with 65% to 79% accuracy depending on task trial. All machine learning models produced a similar classification accuracy. Our results demonstrate the cross-sectional role of the vestibular system in Alzheimer’s disease risk carriers. Future investigations should examine if vestibular functions explain individual phenotypic heterogeneity in path integration among Alzheimer’s disease risk carriers.

![](https://user-images.githubusercontent.com/75738319/210575863-202a3c83-2601-42ac-88dc-bc7c6b5f1157.PNG)

*A) F1 scores for the best performing algorithm are shown. A random predictor would score 0.57, with a score of above 0.57 representing better-than-chance APOE status classification performance. Blue line includes all features. Red line excludes the path integration feature, end error. B) Importance scores are represented by the circle diameter and were derived for the best performing model on each of the trials shown. Scores vary between 0 and 1 depending on the proportion of influence the feature has for that trial. RF = Random Forest, SVM = Support Vector Machine, MLP = Multi-Layer Perception.*

## Software implementation
All source code used to generate the results in the paper are in
the `code` folder.
The pre-processing and modelling are all run using Python.
The data used in this study is provided in `data`. Each participant has their own folder with compass, gyroscope and accelerometer measurements. Deomographic information is found within the `data\Personal`.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/WillPlumb/Vestibular-PreAD.git

or [download a zip archive](https://github.com/WillPlumb/Vestibular-PreAD/archive/refs/heads/master.zip).

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

## Contact
For questions regarding the manuscript, please contact Michael Hornberger (m.hornberger@uea.ac.uk).

For questions on this GitHub, Please contact William Plumb (w.plumb20@imperial.ac.uk)
