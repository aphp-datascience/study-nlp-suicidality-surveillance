<div align="center">
<p align="left">
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://python-poetry.org/" target="_blank">
    <img src="https://img.shields.io/badge/reproducibility-poetry-blue" alt="Poetry">
</a>
<a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-%3E%3D%203.7.10%20%7C%20%3C%3D%203.7.13-brightgreen" alt="Supported Python versions">
</a>
<a href="https://zenodo.org/badge/latestdoi/683012195"><img src="https://zenodo.org/badge/683012195.svg" alt="DOI"></a>
</p>
</div>



# Natural language processing of multi-hospital electronic health records for public health surveillance of suicidality

## Study

This repositoy contains the computer code that has been executed to generate the results of the article:
```
@unpublished{suicideattempt,
author = {Romain Bey and Ariel Cohen and Vincent Trebossen and Basile Dura and Pierre-Alexis Geoffroy and Charline Jean and Benjamin Landman and Thomas Petit-Jean and Gilles Chatellier and Kankoe Sallah and Xavier Tannier and Aurélie Bourmaud and Richard Delorme},
title = {Natural language processing of multi-hospital electronic health records for public health surveillance of suicidality},
note = {Manuscript submitted for publication},
year = {2022}
}
```
The code has been executed on the database of the <a href="https://eds.aphp.fr/" target="_blank">Greater Paris University Hospitals</a>

- IRB number: CSE210013

:warning:
This repository is not maintained. It contains computer code that is specific to a research study.


## Version 1.0.0
- Code of article after review.

## Setup

You should run the file `set_environment.py`  in order to create a conda environment and an associated jupyter kernel.

```
python set_environment.py -n env_cse_210013
conda activate env_cse_210013
pip install --upgrade pip

cd cse_210013
poetry install
```
## How to run the code on AP-HP's data platform

You can run all the analysis pipelines with the `./bash/run_analysis.sh` command:

```
bash bash/run_analysis.sh <conf_name>
```

Example:

```
bash bash/run_analysis.sh conf_article
```

It requires the prior training/import of the machine learning model for SA detection.

## Project structure

### Repository organization
- `bash`: Bash files to execute the pipelines and tests
- `conf`: Configuration files
- `data`: Intermediate data and export results
- `figures`: Figures and their associated tables
- `notebooks`: Tutorials and examples
- `suicide_attempt`: Source code (functions and pipelines)

### Pipelines
1. Stay & Document selection:
Retrieve documents that mention a lexical variant of Suicide Attempt for the stays that fulfill the inclusion criteria
2. `Rule-based entity classification
3. Machine learning (ML) entity classification
4. Stay classification using text data
5. Stay classification using claim data
6. Retrieve documents with a risk factor (RF) mention for the previously SA visits (text data).
7. Rule-based entity classification for the RF
8. Make plots
9. Evaluate configuration & data description
10. Train ML model

### Configuration file
- `debug`: (Boolean) If set to `True`, the pipelines will be executed using only a sample of data. Useful for debuging.
- `schema`: Name of the schema to query.
- `admission_mode`: Admissions mode to keep. For example: [`2-URG`] for admission through the emergency department. If `None`, no criterion is applied.
- `type_of_visit`: Type of visit to keep. For example: [`I`,`U`] for hospitalizations and emergency visits, respectively. If `None`, all visits will be considered.
- `only_cat_docs`: List of text document categories to use exclusively. If `None`, no action is applied.
- `rule_select_docs`: Method used to select one document per visit. If `None`, no selection is applied.
- `text_classification_method`: name of the method used to classify an identified SA entity as positive (`is_true_instance` variable).
- `rule_icd10`: Name of the rule used to classify a visit as positive for SA using claim data.
- `icd10_type`: Source database that is considered for claim data (either `ORBIS` or `AREM`).
- `threshold_positive_instances`: Minimum number of positive suicide attempt text instances found in text to classify the visit as positive.
- `delta_min_visits`: timedelta used to tag recurrent visits related to the same SA event (string with the accepted format of pd.to_timedelta). If `None`, no action is applied.
- `delta_history`: timedelta used to discard SA detected by NLP algorithms but that are related to a patient's history. If the algorithm detects the date of a SA and if the date is before the admission date minus `delta_history`, the visit is not tagged as a SA-caused visit. If `None`, no action is applied.
- `date_from`: Consider only visits fulfilling `start_date` >= `date_from`.
- `date_upper_limit`: date up to which analysis is carried out. Only visits that start strictly before `date_upper_limit` are considered. Also used to fill values of visits with no `visit_end_date` for the Kaplan-Meier estimator.
- `hospitals_train`: List of hospital considered in the training set (trigrams).
- `hospitals_test`: List of hospital considered in the testing set (trigram). If `None`, no action is applied.
- `ehr_deployement_file`: Name of the file containing information on the deployement dates of the electronic health record used for data collection.
- `encounter_subset`: List of encounter numbers to consider exclusively. If `None`, no action is applied.


## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
