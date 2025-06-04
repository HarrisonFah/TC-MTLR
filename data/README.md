## Recreating Data Files
While we do not provide the data used in the original experiments, we do provide instructions and code for recreating the files for each dataset.

Below are the download links and specific instructions for each dataset. After downloading each file, use the associated script in the preprocessing folder to build the .pkl files.

### PBC

Download: https://www.mayo.edu/research/documents/pbcseqhtml/doc-10027141

After downloading, rename the file to a .dat extension

### AIDS

Download: https://github.com/drizopoulos/JM/tree/master/data

After downloading, run the following R code before using the python preprocessing script:
```
load("aids.rda")
write.csv(aids, "aids.csv", row.names = TRUE)
```

### SmallRW/LargeRW

Neither random walk dataset requires downloading any files. Simply use the preprocessing script with the small and large hyperparameters.

### LastFM

Download: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

### NASA

Download: https://www.kaggle.com/code/vinayak123tyagi/damage-propagation-modeling-for-aircraft-engine/input

Download only the train_FD001.txt and test_FD001.txt files.


