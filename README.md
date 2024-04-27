# Manual for Automatic Speech and Push-To-Talk detection

This manual contains an explanation how to work with source files.

## The /src folder can be divided into two parts.

### First part contains own experimental models, as well as modules for data pre-processing and module for evaluation.

For the training of the FNN model, it is necessary to run the script:

```
./python3.10 neural_network.py -net
```

For the training of the CNN model, this argument is not needed.

For the evaluation of the model:

```
./python3.10 evaluationnets.py 
```

Again, for the FNN model, the ```-net``` parameter is required and for CNN model, it is not.

The modules ```balanced_dataset.py``` serves for distribution between speech, non-speech, PTT and non-PTT frames.

Module ```dataprocessing.py``` is suited for ATCO2 dataset. The module loads all the necessary files and processes the data to Mel Spectrograms and as well it creates labels for speech segments.

The module ```main.py``` was commented the call function for loading the dataset because this model is called by ```neural_network.py``` so it prevents opening dataset two times.

The dataset can be parsed with a call

```
./python3.10 main.py 
```

which processes the data as for CNN model.

The push-to-talk is labelled with the help of ```PEEK.py``` module which contains detection algorithm.

The module ```peaks.py``` is an implementation provided from http://billauer.co.il/blog/2009/01/peakdet-matlab-octave/

### The second part contains adaptation of the GPVAD implementation.

The modified files were ```forward.py, run.py, losses.py, dataset.py, prepare_labels.py```.

These modules contains modifications to be able to evaluate the detection of the PTT.

Folders ```csv_labels, softlabels``` and ```hdf5``` contains labels and extracted spectrograms for fine-tuning the neural network on the push-to-talk detection.

All other implementation details about GPVAD model can be found on:

https://github.com/RicherMans/Datadriven-GPVAD

The module ```forward.py``` evaluated the provided audio. The implicit model is MTL and can be changed to PTT model.

```
./python3.8 forward.py -model ptt -w ../wavfile/LKPR_RUZYNE_Radar_120_520MHz_20201024_221711.wav
```

The module ```run.py``` is a training module prepared for fine-tuning the sre model. The models takes as input the labels and features from ```csv_labels, softlabels``` and ```hdf5``` files.

```
./python3.8 run.py train configs/example.yaml
```

## Folder saved_models

The folder contains saved pre-trained models which were mentioned in the thesis report.

Experimental neural network folders contain pretrained model called ```model.pth```, and ROC curve plotted into image ```ROC.png```.

The GPVAD folders contain pretrained models and log reports during training

## The folder datasets

This folder contains all of the negative push-to-talk events in the ATCO2 dataset.

There might be problem with requirements because some of the used functions in GPVAD implementation did not work so it is compiled only with ```python3.8```. The other experiments are compiled with ```python3.10```. The rule is, if it is the older version it is for ```python3.8```.
