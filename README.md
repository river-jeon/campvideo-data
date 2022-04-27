# campvideo-data
Replication data for ["Automated Coding of Political Campaign Advertisement Videos: An Empirical Validation Study"]() by Alexander Tarr, June Hwang, and Kosuke Imai.

## Overview
Full replication of the results in the paper is a laborious process, involving significant setup and computation time on the part of the user. To simplify the procedure, we have split replication into two parts: [Feature Extraction](#Feature-Extraction) and [Validation](#Validation). For those seeking only to validate the results in the paper, it is **highly recommended** to ignore feature extraction and follow the steps for validation, which uses pre-computed features from the feature extraction step.

## Repository Layout
This repository is split into several folders: ``data``, ``figs``, ``results``, ``scripts`` and ``tables``.
- ``data``: This folder contains all data needed to perform both feature extraction and validation.
  * ``ids``: Numpy vectors for face encodings corresponding to Senate candidates in the 2012 and 2014 elections.
  * ``intermediate``: Extracted feature data for each YouTube video in the study. This data includes Numpy vectors for audio features, keyframe indices, auto-generated transcripts, and detected image text. Data in this folder is created in the [Feature Extraction](#Feature-Extraction) step.
  * ``mturk``: CSV files containing results from the Amazon Mechanical Turk studies.
  * ``validation``: CSV files results from the validation analyses given in the appendix.
  * ``videos``: MP4 files corresponding to YouTube videos used in the study. Data in this folder is used in the [Feature Extraction](#Feature-Extraction) step.
  * ``wmp``: DTA files containing WMP/CMAG data. Data in this folder is used in the [Validation](#Validation) step.
- ``figs``: PDFs for figures generated by the code that are displayed in the paper.
- ``results``: CSV files containing predicted labels for tasks studied in the paper. There are also raw text files showing general statistics about the performance of our methods that are discussed in the main text of the paper.
- ``scripts``: All code needed to generate data, extract features, validate results, and create figures and tables.
-  ``tables``: Raw text files showing confusion matrices corresponding to tables in the paper.

## Data
Replication relies on two datasets. [Feature Extraction](#Feature-Extraction) requires the collection of YouTube videos in MP4 format, while [Validation](#Validation) requires the human-coded labels provided by WMP. Unfortunately, neither of these datasets can be provided publicly.

- YouTube Videos: We provide a list of the YouTube Video IDs used in the study in <JUNE'S FILE: TBD>. Users able to obtain these videos should place them in the ``data\videos`` folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.
- WMP Data: The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the ``data\wmp`` folder.

## Validation
To replicate the results in the paper, follow the instructions below in order as they appear. 

### Installation
Recreating all figures, tables and results in the [Validation](#Validation) step requires working installations of [Python](https://www.python.org/downloads/) and [R](https://cran.r-project.org/src/base/R-4/). All code in this repo was tested under Python version 3.9.7 and R version 4.0.5 on a Windows 10 machine. 

#### Python Dependencies
Most Python package dependencies can be installed by installing the project-related package, ``campvideo``, which is available on [TestPyPi package repository](https://test.pypi.org/project/campvideo/). This package can be installed within a Python environment via the command

    pip install -i https://test.pypi.org/simple/ campvideo

Additionally, the Python code depends on ``matplotlib, seaborn``, which can be installed via command line:

    pip install <PACKAGE_NAME>
    
#### R Dependencies
All R code uses the following packages: ``quanteda, readstata13, readtext, stringr, xtable``, which can be installed from within the R environment via

    install.packages("<PACKAGE_NAME>")

#### spaCy Model Download
The ``spacy`` text modeling package requires downloading a model. After installing the Python packages, enter the following in the command line:

    python -m spacy download en_core_web_md
    
### Preprocessing the WMP Data
Before any results can be produced, the WMP data must be cleaned. After placing the Stata files into ``data\wmp``, clean the data via

    Rscript scripts/Replication_Preprocess.R
    
This file may also be sourced from within an IDE, such as RStudio. Be sure to set the working directory to repo folder, ``campvideo-data``.

### Result Replication
The following commands recreate the tables and figures in the paper. The generated figures are found in the ``figs`` folder, while the tables are stored in raw text files in the ``tables`` folder. Additionally, performance metrics discussed in the paper as well as our predicted labels are stored in the ``results`` folder.

#### Coverage Tables
This section gives instructions for replicating the coverage tables (Section 2.2, Appendix S1).
- Table 1 in the main text is replicated via

      Rscript scripts/table1.R

- Table S1.1 in the appendix is replicated via

      Rscript scripts/tableS1-1.R

#### Text Validation
This section gives instructions for replicating issue mention (Section 4.1, Appendix S11), opponent mention (Section 4.2, Appendix S12), and ad negativity classification (Section 4.5, Appendix S14.2, Appendix S14.3) results.
- Table 2, Table 3, Table 6, and Table S14.2 are replicated via

      python scripts/text_validation.py
  
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables. To recreate the data in ``results``, type the command
  
      python scripts/text_validation.py --calculate
      
  The ``calculate`` flag forces the script to scan the auto-generated transcipts for issue and opponent mentions and to retrain the text models described in the paper using the WMP data as ground truth. The resulting predictions are then saved to the ``results`` folder.
  
- Figure 5 is replicated via

      Rscript scripts/fig5.R
      
- Performance metrics for issue mentions and opponent mentions are found in ``results\issue_results.txt`` and ``results\issue_results.txt``, which are replicated with

      python scripts/text_validation.py

#### Face Recognition Validation
This section gives instructions for replicating face recognition results (Section 4.3, Appendix S13).
- Table 4 and Figure S13.8 are replicated via

      python scripts/facerec_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables and figures. To recreate the data in ``results``, type the command
  
      python scripts/facerec_validation.py --calculate
      
  The ``calculate`` flag forces the script to detect and recognize faces in the keyframes of each video and to recompute the distance threshold. The resulting predictions are then saved to the ``results`` folder.
  
- Performance metrics for face recognition are found in ``results\facerec_results.txt``, which are replicated with

      python scripts/text_validation.py

#### Music Mood Validation
This section gives instructions for replicating music mood classificaiton results (Section 4.4, Appendix S14.1).

- Table 5, and Table S14.5 are replicated via

      python scripts/mood_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables. To recreate the data in ``results``, type the command
  
      python scripts/mood_validation.py --calculate
      
  The ``calculate`` flag forces the script to retrain the music mood models described in the paper using the WMP data as ground truth.. The resulting predictions are then saved to the ``results`` folder.
      
- Figure 8, Figure S14.9, and Figure S14.10 are replicated via

      Rscript scripts/figs8_14-9_14-10.R
      
- Performance metrics for music mood classification are found in ``results\mood_results.txt``, which are replicated with

      python scripts/mood_validation.py

#### Video Summary Validation
This section gives instructions for replicating results in the summary validation study (Appendix S7).

- Figure S7.4 is replicated via

      python scripts/summary_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the figure. To recreate the data in ``results``, type the command
  
      python scripts/summary_validation.py --calculate
      
  The ``calculate`` flag forces the script to compute all relevants metrics for each video summary. The results are then saved to the ``results`` folder.
  
#### Ad Negativity Classification with LSD
This section gives instructions for replicating ad negativity classification results using LSD (Appendix S14.3).

- Table S14.7 is replicated via

      Rscript scripts/tableS14-7.R

#### Kaplan *et al.* (2006) Replication
This section gives instructions for replicating the issue convergence study using our predicted labels (Appendix S14.4).

- Table S14.7 is replicated via
      
      Rscript scripts/tableS14-7.R

## Feature Extraction
To replicate the feature extraction step for creating all data in ``data\intermediate``, follow the instructions below in order as they appear.

### Data
As stated previously, this section can only be replicated if the user has access to the YouTube videos in the study. If a user manages to obtain access to all videos, they must be placed in the ``data\videos`` folder. The videos should be in MP4 format and titled ``<YouTubeID>.mp4``.

### Installation
Recreating the intermediate results in the [Feature Extraction](#Feature-Extraction) step requires a working installations of [Python](https://www.python.org/downloads/). All code in this repo was tested under Python version 3.9.7 on a Windows 10 machine.

#### CUDA and cuDNN
We **strongly recommended** that users with access to a dedicated GPU for computing install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). While optional, using a dedicated GPU for face recognition will greatly improve accuracy and computation time.

#### Google Cloud Platform (GCP)
Image text recognition and speech transcription are performed using GCP. Enabling GCP on your machine requires creating a project and setting up a billing account [here](https://cloud.google.com/docs/get-started). Once the account is setup, be sure to enable the following APIs:
- Google Cloud Vision API
- Google Cloud Video Intelligence API

**Note that using GCP costs money**. Setting up a GCP account and replicating this section will result in charges being made to your billing account.

#### Python Dependencies
All Python package dependencies can be installed by installing the project-related package, ``campvideo``, which is available on [TestPyPi package repository](https://test.pypi.org/project/campvideo/). This package can be installed within a Python environment via the command

    pip install -i https://test.pypi.org/simple/ campvideo

#### dlib
The Python package ``dlib`` must be compiled from source in order to use CUDA and cuDNN. See [this link](http://dlib.net/compile.html) for instructions on how to do this.

#### Model Download
After installing the ``campvideo`` package, download the relevant models via the command

    download_models
    
#### Feature Extraction
The intermediate data in ``data\intermediate`` can be replicated via

    python scripts/generate_data.py --overwrite
    
The ``overwrite`` flag signals the script to replace existing data in ``data\intermediate``. Without this flag, the script will skip over videos with existing data. If the user wishes to do partial replication of the feature extraction step **without** GCP, the command

    python scripts/generate_data.py --overwrite --no-gcp
    
will compute audio features and video features only.

## Additional Notes
- Feature extraction, model training, and prediction require significant processing time. Expect full replication of the results in the paper to take several days. Conversely, recreating all figures and tables using pre-computed results and features takes very little time.
- Image text recognition and speech transcription with GCP require a stable internet connection. Service interruptions during execution of ``scripts/generate_data.py`` may lead to missing data.
- Exact replication for label prediction is only guaranteed for the models we train. Face recognition, image text recognition, and speech transcription all rely on external models which we have no control over. Future updates to these models may lead to slightly different results.
- 'File not found' errors are likely due to issues with working directory. All code assumes this repo, `campvideo-data`, is the working directory.
