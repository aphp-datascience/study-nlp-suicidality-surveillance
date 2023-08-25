---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: env_cse_210013
    language: python
    name: env_cse_210013
---

# Preliminary
Probe of the deployment of the main electronic health record (EHR) software.
This script can be executed by "data curators" only and not by the "investigators", because of data access rights.

# Imports and setup
```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext lab_black
import pandas as pd
```

```python
pd.set_option("max_columns", None)
```

```python
from suicide_attempt.functions.ehr_deployement import ProbeEHRDeployement
```

```python
from suicide_attempt.functions.constants import dict_code_UFR
```

```python
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

# Stop current Spark context
sc.stop()

# Loading pyspark
conf = SparkConf().setAppName("ProbeEHRDeployement")
conf.set("spark.yarn.max.executor.failures", "10")
conf.set("spark.executor.memory", "2g")
conf.set("spark.dynamicAllocation.enabled", True)
conf.set("spark.dynamicAllocation.minExecutors", "6")
conf.set("spark.dynamicAllocation.maxExecutors", "15")
conf.set("spark.executor.cores", "5")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession.builder.enableHiveSupport().getOrCreate()
sql = spark.sql
```

# Definition
**Visits of type HC (I)**

Doc in the following list :
     "CRH-CHIR",
    "CRH-HOSPI",
    "CRH-J",
    "CRH-NEUROL",
    "CRH-PEDIA",
    "CRH-S",
    "LT-SOR",
    "INCONNU",

```python
kwargs = dict(
    schema="",
    only_cat_docs=[
        "CRH-CHIR",
        "CRH-HOSPI",
        "CRH-J",
        "CRH-NEUROL",
        "CRH-PEDIA",
        "CRH-S",
        "LT-SOR",
        "INCONNU",
    ],
    only_type_visit=[
        "I",
    ],
)
```

# Structures 

```python
hospit = {
    "BCH": "HOPITAL BICHAT",
    "LMR": "HOPITAL LOUIS MOURIER",
    "SAT": "HOPITAL SAINT ANTOINE",
    "APR": "HOPITAL AMBROISE PARE",
    "SLS": "HOPITAL SAINT LOUIS",
    "NCK": "GROUPE HOSPITAL.NECKER ENFANTS MALADES",
    "PSL": "GROUPE HOSPITALIER PITIE-LA SALPETRIERE",
    "ABC": "HOPITAL ANTOINE BECLERE",
    "BCT": "HOPITAL DE BICETRE",
    "JVR": "HOPITAL JEAN VERDIER",
    "TRS": "GH ARMAND TROUSSEAU-LA ROCHE GUYON",
    "TNN": "HOPITAL TENON",
    "CFX": "HOPITAL CHARLES FOIX",
    "HMN": "GH A.CHENEVIER-H.MONDOR",
    "LRB": "GH LARIBOISIERE FERNAND WIDAL",
    "PBR": "HOPITAL PAUL BROUSSE",
}
```

```python
# Import labels of hospitls codes
key_hospitals = pd.DataFrame(
    dict_code_UFR.items(), columns=["care_site_id", "care_site_name"]
)
```

```python
list_hospitals = key_hospitals.loc[
    key_hospitals.care_site_name.isin(hospit.keys())
].care_site_id.to_list()
```


# Probe 

```python
cse210013_probe = ProbeEHRDeployement.from_spark(sql, **kwargs)
```

```python
cse210013_probe.head()
```

# Export

```python
cse210013_probe = cse210013_probe.loc[cse210013_probe.care_site_id.isin(list_hospitals)]

```
```python
pd.DataFrame(cse210013_probe).to_pickle(
    "~/cse_210013/data/export/ehr_deployement/ratio_doc_hospit_092022.pickle"
)
```
