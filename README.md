# campvideo-data
Replication data for ["Automated Coding of Political Campaign Advertisement Videos: An Empirical Validation Study"]() by Alexander Tarr, June Hwang, and Kosuke Imai.

## Table of Contents
1. [Overview](#Overview)
2. [Repository Structure](#Repository-Structure)
3. [Data](#Data)
4. [Installation](#Installation)
5. [Preprocessing the WMP Data](#Preprocessing-the-WMP-Data)
6. [Figure and Table Replication](#Figure-and-Table-Replication)
7. [Additional Notes](#Additional-Notes)

## Overview
Full replication of the results in the paper is a laborious process, involving significant setup and computation time on the part of the user. To simplify the procedure, we have split replication into three steps: 
1. [Feature Extraction](README-FE.md#Feature-Extraction)
2. [Prediction](README-PR.md#Prediction)
3. [Validation](#Validation) 

Each step may also be executed separately using pre-compute results provided in this repository. For those seeking only to validate the results in the paper, it is **highly recommended** to ignore the first two steps, feature extraction and prediction, and follow the steps for validation.

We provide instructions for replicating the [Validation](#Validation) step in this document, while instructions for replicating feature extraction and prediction are found in [README-FE.md](README-FE.md) and [README-FE.md](README-PR.md), respectively. 

## Repository Structure
This repository is split into several folders: ``data``, ``figs``, ``results``, ``scripts`` and ``tables``.
- ``data``: This folder contains all data needed to perform both feature extraction and validation.
- ``figs``: PDFs for figures generated by the code that are displayed in the paper.
- ``results``: CSV files containing predicted labels for tasks studied in the paper. There are also raw text files showing general statistics about the performance of our methods that are discussed in the main text of the paper.
- ``scripts``: All code needed to generate data, extract features, validate results, and create figures and tables.
- ``tables``: Raw text files showing confusion matrices and coverage tables corresponding to tables in the paper.

## Data
Replication in the [Validation](#Validation) step requires the human-coded labels provided by WMP, which cannot be shared publicly. This data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.

# Validation
To replicate the validation step for creating all [figures](figs) and [tables](tables), follow the instructions below in order as they appear.

## Installation
Recreating all figures, tables and results requires working installations of
- [Python](https://www.python.org/downloads/), version 3.9 or greater. We recommend using the [Anaconda distribution](https://www.anaconda.com/products/distribution) if unfamiliar with Python.
- [R](https://cran.r-project.org/src/base/R-4/), version 4.0 or greater.

All code in this repo was tested under Python version 3.9.7 and R version 4.0.5 on a Windows 10 machine. 

### Python Dependencies
All Python code in the validation step uses the following packages: ``matplotlib, numpy, pandas, scikit-learn, seaborn``, all of which can be installed via

```
pip install <PACKAGE_NAME>
````

### R Dependencies
All R code uses the following packages: ``dplyr, here, lme4, quanteda, quanteda.sentiment, readstata13, readtext, stargazer, xtable``, most of which can be installed from within the R environment via

```r
install.packages("<PACKAGE_NAME>")
```

``quanteda.sentiment`` is not available on CRAN and must be installed via

```r
devtools::install_github("quanteda/quanteda.sentiment")
```
    
## Preprocessing the WMP Data
Before any results can be produced, the WMP data must be cleaned. After placing the Stata files into [``data/wmp``](data/wmp), clean the data via

```sh
Rscript scripts/preprocess_CMAG.R
```

This file may also be sourced from within an IDE, such as RStudio. Be sure to set the working directory to repo folder, [``campvideo-data``](https://github.com/atarr3/campvideo-data). After running, a file called ``wmp_final.csv`` should be created in [``data/wmp``](data/wmp).

## Figure and Table Replication
All figure and table replication scripts are in the [``scripts``](scripts) folder. The files are named after the figures and tables they replicate. For example, [``figure5.R``](scripts/figure5.R) recreates Figure 5, and [``tableS14-6.py``](scripts/tableS14-6.py) recreates Appendix Table S14.6. Note that some scripts create multiple tables or figures.

The full list of figures and tables and associated replication code is given below.

| Result                                 | Description                                                | Language | Script                                                       |
| :------------------------------------- | :--------------------------------------------------------- | :------- | :----------------------------------------------------------- |
| [Figure 5](figs/figure5.pdf)           | MTurk results for issue mentions                           | R        | [``figure5.R``](scripts/figure5.R)                           |
| [Figure 8](figs/figure8.pdf)           | MTurk results for ominous/tense mood classification        | R        | [``figure8_S14-9_S14-10.R``](scripts/figure8_S14-9_S14-10.R) |
| [Figure S7.4](figs/figureS7-4.pdf)     | Video summarization validation study results               | Python   | [``figureS7-4.py``](scripts/figureS7-4.py)                   |
| [Figure S13.8](figs/figureS13-8.pdf)   | ROC plots for face recognition                             | Python   | [``figureS13-8.py``](scripts/figureS13-8.py)                 |
| [Figure S14.9](figs/figureS14-9.pdf)   | MTurk results for uplifting mood classification            | R        | [``figure8_S14-9_S14-10.R``](scripts/figure8_S14-9_S14-10.R) |
| [Figure S14.10](figs/figureS14-10.pdf) | MTurk results for sad/sorrowful mood classification        | R        | [``figure8_S14-9_S14-10.R``](scripts/figure8_S14-9_S14-10.R) |
| [Table 1](tables/table1.txt)           | Matched video coverage table                               | R        | [``table1.R``](scripts/table1.R)                             |
| [Table 2](tables/table2.txt)           | Confusion matrices for issue mentions                      | Python   | [``table2.py``](scripts/table2.py)                           |
| [Table 3](tables/table3.txt)           | Confusion matrices for opponent mentions                   | Python   | [``table3.py``](scripts/table3.py)                           |
| [Table 4](tables/table4.txt)           | Confusion matrices for face recognition                    | Python   | [``table4.py``](scripts/table4.py)                           |
| [Table 5](tables/table5.txt)           | Confusion matrices for mood classiification                | Python   | [``table5.py``](scripts/table5.py)                           |
| [Table 6](tables/table6.txt)           | Confusion matrices for ad negativity classification (NSVM) | Python   | [``table6.py``](scripts/table6.py)                           |
| [Table S1.1](tables/tableS1-1.txt)     | YouTube channel coverage table                             | R        | [``tableS1-1.R``](scripts/tableS1-1.R)                       |
| [Table S14.5](tables/tableS14-5.txt)   | Confusion matrix for mood MTurk results                    | Python   | [``tableS14-5.py``](scripts/tableS14-5.py)                   |
| [Table S14.6](tables/tableS14-6.txt)   | Confusion matrices for ad negativity classification (All)  | Python   | [``tableS14-6.py``](scripts/tableS14-6.py)                   |
| [Table S14.3](tables/tableS14-7.txt)   | Confusion matrix for LSD results                           | R        | [``tableS14-7.R``](scripts/tableS14-7.R)                     |
| [Table S14.8](tables/tableS14-8.txt)   | Regression coefficients for issue convergence study        | R        | [``tableS14-8.R``](scripts/tableS14-8.R)                     |

Python scripts can be executed via

```
python scripts/<SCRIPT>
```

and R scripts can be executed via

```
Rscript scripts/<SCRIPT>
```

where ``<SCRIPT>`` is given by the name in the "Script" column in the table above.

## Additional Notes
- Recreating the figures and tables using pre-computed results only takes a few minutes.
- Some confusion matrices will differ slightly from what is displayed in the paper. This is due to significant figure truncation, which does not guarantee the sum of the confusion matrices add up to 100% after rounding. The values in the paper have been adjusted to add up to 100%.
- 'File not found' errors are likely due to issues with working directory. All code assumes this repo, `campvideo-data`, is the working directory.
