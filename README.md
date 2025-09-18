# RTFVE: Realtime Face Video Enhancement
Official repository for the CAIP 2025 paper [RTFVE: Realtime Face Video Enhancement](https://link.springer.com/chapter/10.1007/978-3-032-04968-1_12).

![The RTFVE Model](/assets/RTFVE.png)

[Download the demo of the model here.](/assets/supp.mp4)

### Dataprep
Data needs to be prepared like in the example provided in sample_data/train. The high quality frames for the videos go in the hq folder and their corresponding low quality frames go in the lq folder. The naming convention for these frames is vidnum_framenum.png and the high quality reference frames goes in the ref/vidnum folder. Our model can use one or more high quality reference images for each input video frame. We recommend trying 1-5 reference images. Once you have prepared the dataset, go into configs/train.yaml and update the train and valid paths.

### Training
To train an SR model simply run the following from the terminal:
```
$ python train.py -c configs/train.yaml
```
All models will be saved in the /checkpoints folder. If you are training multiple models remember to change the *name* parameter in the config files otherwise it may just override the previous experiment. Also, if your training crashed in the middle for whatever reason, you can resume from a checkpoint by changing the *epoch_start* parameter in the config. 

### Inference on Video
To perform inference on a low quality face video run:
```
$ python video_face.py
```
But before doing this update the *fpath* and *rpath* parameters at the top of the file. 

### Citation
Please cite our work if it was helpful in any way. 

@InProceedings{10.1007/978-3-032-04968-1_12,
author="Jois, Varun Ramesh
and DiLillo, Antonella
and Storer, James",
editor="Castrill{\'o}n-Santana, Modesto
and Travieso-Gonz{\'a}lez, Carlos M.
and Deniz Suarez, Oscar
and Freire-Obreg{\'o}n, David
and Hern{\'a}ndez-Sosa, Daniel
and Lorenzo-Navarro, Javier
and Santana, Oliverio J.",
title="RTFVE: Realtime Face Video Enhancement",
booktitle="Computer Analysis of Images and Patterns",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="136--148",
isbn="978-3-032-04968-1"
}
