---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: env_cse_210013
    language: python
    name: env_cse_210013
---

```python
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

#Stop current Spark context
sc.stop()

#Loading pyspark
conf = SparkConf().setAppName("test_new_terms")
conf.set("spark.yarn.max.executor.failures", "10")
conf.set("spark.executor.memory", '2g')
conf.set("spark.dynamicAllocation.enabled",True)
conf.set("spark.dynamicAllocation.minExecutors","10")
conf.set("spark.dynamicAllocation.maxExecutors","15")
conf.set("spark.executor.cores","5")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sql = spark.sql
```

```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext lab_black
import pandas as pd

pd.set_option("max_columns", None)
```

```python
from suicide_attempt.functions.utils import get_conf
```

```python
from suicide_attempt.functions.retrieve_data import retrieve_docs, retrieve_stays
```

```python
from suicide_attempt.functions.text_utils import spark_filter_docs_by_regex
```

```python
from suicide_attempt.functions.stats_utils import get_sa_data
```

```python
from suicide_attempt.functions.constants import regex_sa
```

```python
conf_name = "conf_article"
```

```python
parameters = get_conf(conf_name)
```

<!-- #region tags=[] -->
## 1 Patients with at least one SA keword detected in their docs
<!-- #endregion -->

```python
# Retrieve visits
visits = retrieve_stays(
    list_type_of_visit=parameters["type_of_visit"],
    date_from=parameters["date_from"],
    date_to=parameters["date_upper_limit"],
    schema=parameters["schema"],
)

# Select columns
visits = visits.withColumnRenamed("start_date", "visit_start_date")
visits = visits.withColumnRenamed("end_date", "visit_end_date")
visits = visits.select(["encounter_num", "patient_num"])


# Retrieve documents
documents = retrieve_docs(schema=parameters["schema"])

documents = documents.drop("patient_num").join(visits, how="inner", on="encounter_num")

# Filter documents by regex
documents = spark_filter_docs_by_regex(documents, regex_pattern_dictionary=regex_sa)


# Nbr stays (with docs that match)
n_stay_1_docs_match = (
    documents.cache().select("encounter_num").drop_duplicates().count()
)

# Nbr docs that match
n_documents_1 = documents.count()

# Number of patients
patients_set = documents.select("patient_num").drop_duplicates()
n_patient_1 = patients_set.cache().count()

# All visits for these patients
all_visits_patients = visits.join(patients_set, on="patient_num", how="inner")
n_stay_1 = all_visits_patients.count()


print("N patient 1:", n_patient_1)
print("N stays 1 (with docs that match):", n_stay_1_docs_match)
print("N documents 1 (that match):", n_documents_1)
print("N stays 1 (with docs that match or not):", n_stay_1)
```

## 2 stays in the 15 hospitals with advanced deployement of the EHR software

```python
list_hospitals = parameters["hospitals_train"] + parameters["hospitals_test"]
```

```python
visits_hosp = retrieve_stays(
    list_type_of_visit=parameters["type_of_visit"],
    date_from=parameters["date_from"],
    date_to=parameters["date_upper_limit"],
    schema=parameters["schema"],
    list_hospitals=list_hospitals,
)

visits_hosp_f = visits_hosp.join(patients_set, on="patient_num", how="inner")
```

```python
# Stays
n_stays_2 = visits_hosp_f.count()

# Patients
n_patients_2 = visits_hosp_f.select("patient_num").drop_duplicates().count()

# Documents with at least one match for these visits
documents_hosp_f = documents.join(
    visits_hosp_f.select("encounter_num"), how="inner", on="encounter_num"
)

n_documents_2 = documents_hosp_f.cache().count()

# Nbr Stays of docs that match
n_stay_2_docs_match = documents_hosp_f.select("encounter_num").drop_duplicates().count()
```

```python
print("N documents 2 (with at least one match of regex):", n_documents_2)
print("N patient 2:", n_patients_2)
print("N stays 2:", n_stays_2)
print("N stays 2 (with docs that match):", n_stay_2_docs_match)
```

## 3 Stays caused by SA

```python
path = os.path.expanduser(
    f"~/cse_210013/data/{conf_name}/stay_classification_text_data_{conf_name}"
)
df = pd.read_pickle(path)
df = df.loc[df.date < parameters["date_upper_limit"]]
df_nlp_positive = df.loc[df.nlp_positive]
```

```python
n_stay_3 = df_nlp_positive.encounter_num.nunique()
print("N stays 3:", n_stay_3)
```

```python
n_patient_3 = df_nlp_positive.patient_num.nunique()
print("N patient 3:", n_patient_3)
```

## 4 Patients with valid birth date or sex

```python
df_valid_age_sex = df_nlp_positive.loc[
    ~(
        (df_nlp_positive.age.isna())
        | (df_nlp_positive.age < 0)
        | (df_nlp_positive.sex_cd == "Unknown")
    )
]
```

```python
n_stay_4 = df_valid_age_sex.encounter_num.nunique()
print("N stays 4:", n_stay_4)
```

```python
n_patient_4 = df_valid_age_sex.patient_num.nunique()
print("N patient 4:", n_patient_4)
```

## 5 Stays w patient aged >=8 at admission

```python
df_8years = df_valid_age_sex.loc[df_valid_age_sex.age_cat != "Unknown"]
```

```python
n_stay_5 = df_8years.encounter_num.nunique()
print("N stays 5:", n_stay_5)
```

```python
n_patient_5 = df_8years.patient_num.nunique()
print("N patient 5:", n_patient_5)
```

# 6 Stays with no previous stay in the preceding 15 days

```python
stays = df_8years.loc[~df_8years.recurrent_visit]
```

```python
n_stay_6 = stays.encounter_num.nunique()
print("N stays 6:", n_stay_6)
```

```python
n_patient_6 = stays.patient_num.nunique()
print("N patient 6:", n_patient_6)
```

```python

```
