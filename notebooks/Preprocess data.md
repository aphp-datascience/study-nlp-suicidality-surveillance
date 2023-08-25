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
%load_ext lab_black
```

```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
```

```python
import pandas as pd

pd.set_option("max_columns", None)

from suicide_attempt.functions.nlp_ml import ml_utils
from spacy.tokens import Token, Doc
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
```

```python
from spacy.tokens import Span
import spacy
```

```python
import os
```

```python
from edsnlp.utils.filter import filter_spans
```

```python
from sklearn.metrics import classification_report
```

```python
# Read Data


paths = {
    "conf19": f"/export/home/cse210013/cse_210013/data/annotation/conf19/annotation_results_conf19_clean.pickle",
    "conf14": f"/export/home/cse210013/cse_210013/data/annotation/conf14/annotation_results_conf14_clean.pickle",
    "conf13": f"/export/home/cse210013/cse_210013/data/annotation/conf13/annotation_results_conf13_clean.pickle",
    "conf13bis": f"/export/home/cse210013/cse_210013/data/annotation/conf13bis/annotation_results_conf13bis_clean.pickle",
}
dfs_t = []

for conf, path in paths.items():
    df = pd.read_pickle(path)

    df = df.loc[df.is_instance_ts]

    df.rename(
        columns={
            "gold_label_name": "lexical_variant",
            "gold_label_value": "gold_lexical_variant_value",
            "is_true_instance": "label",
            "gold_antecedent_value": "gold_history_value",
        },
        inplace=True,
    )
    # Select columns to keep
    columns = [
        "patient_num",
        "encounter_num",
        "index_snippet",
        "snippet",
        "note_id",
        "note_text",
        "offset_begin",
        "offset_end",
        "lexical_variant",
        "gold_lexical_variant_value",
        "gold_?_value",
        "gold_negated_value",
        "gold_family_value",
        "gold_history_value",
        "gold_hypothesis_value",
        "gold_rspeech_value",
        "label",
        "prediction",
        "n_annotator",
        "annotator",
    ]
    df = df[columns]
    df["label"] = df.label.astype(int)
    df["prediction"] = df.prediction.astype(int)
    df.rename(
        columns={"prediction": "rule_based_prediction", "snippet": "snippet_ann"},
        inplace=True,
    )

    idx = df.loc[
        (df.lexical_variant != "start_doc_icd10") & (~df["gold_?_value"])
    ].index
    df = df.loc[idx]
    df["conf"] = conf
    print(f"Conf: {conf} - Number of lines:", len(df))

    dfs_t.append(df)

df = pd.concat(dfs_t, axis=0)

sorter = [
    "ariel",
    "romain",
    "thomas",
    "basile",
    "charline",
    "keyvan",
]
df.annotator = df.annotator.astype("category")
df.annotator = df.annotator.cat.set_categories(sorter)
df.drop_duplicates(
    subset=[
        "note_id",
        "offset_begin",
    ],
    inplace=True,
    keep="first",
)

print("Number of lines:", len(df))
df.reset_index(inplace=True, drop=True)
df.head()
```

```python
# Evaluate rule based prediction (all entities)
classification_report(
    y_true=df.label, y_pred=df.rule_based_prediction, output_dict=True
)
```

```python
df.encounter_num.nunique()
```

```python
df.note_id.nunique()
```

```python
df.conf.value_counts(normalize=True)
```

```python
# Create a spacy object
nlp = spacy.blank("eds")
```

```python
col_label = "label"
# val_size = 0.20
extraction_parameters = dict(k1=35, k2=10, mode="asymmetric")
label_all_tokens = False
```

```python
# Add the extension 'col_label'
if not Token.has_extension("word_label"):
    Token.set_extension("word_label", default=-100)  # -1
```

```python tags=[]
# Convert raw text into spacy document
texts = df.drop_duplicates("note_id")[["note_id", "note_text"]]
texts["doc"] = list(nlp.pipe(texts.note_text))
```

```python tags=[]
# Add the entities to the spacy document with the model_label
if not Span.has_extension("idx"):
    Span.set_extension("idx", default=None)
for note_id, doc in zip(texts.note_id, texts.doc):
    doc._.note_id = note_id
    sub_df = df.loc[df.note_id == note_id]

    spans = []
    for lexical_variant, offset_begin, offset_end, label, idx in zip(
        sub_df.lexical_variant,
        sub_df.offset_begin,
        sub_df.offset_end,
        sub_df[col_label],
        sub_df.index,
    ):

        span = doc.char_span(
            offset_begin,
            offset_end,
            label=lexical_variant,
            alignment_mode="expand",
        )

        span._.idx = idx

        if label_all_tokens:
            for token in span:
                token._.word_label = int(label)
                token._.rule_based_prediction = int()
        else:
            token = span[0]
            token._.word_label = int(label)

        spans.append(span)
        spans = filter_spans(spans)
    doc.ents = spans
    # doc.spans["sa_entities"] = spans
```

```python
def _flatten(list_of_lists):
    """
    Flatten a list of lists to a combined list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def pick_results(doc, mode, k1, k2):
    ents = []

    for ent in doc.ents:
        if mode == "symmetric":
            text_to_model = doc[max(0, ent.start - k1) : ent.end + k1]
        elif mode == "asymmetric":
            text_to_model = doc[max(0, ent.start - k1) : ent.end + k2]

        relative_index = ent.start - text_to_model.start

        d = dict(
            note_id=doc._.note_id,
            text_to_model=text_to_model,
            relative_index=relative_index,
            start_char=ent.start_char,
            end_char=ent.end_char,
            label=ent[0]._.word_label,
            span_text=text_to_model.text,
            idx=ent._.idx,
        )

        ents.append(d)

    return ents
```

```python
# Extract a span around the entity
# k is the number of spacy tokens to keep before/after the start/end of the entity

entities = texts.doc.apply(pick_results, **extraction_parameters)

entities = _flatten(entities)

entities = pd.DataFrame(entities)
```

```python tags=[]
entities.set_index("idx", inplace=True)
```

```python
entities.index.is_unique
```

```python
df = df.merge(
    entities[
        [
            "text_to_model",
            "relative_index",
            "span_text",
        ]
    ],
    right_index=True,
    left_index=True,
    how="inner",
    validate="many_to_one",
)
```

```python
# Split info into arrays (labels, vectors, etc)
df = ml_utils.split_into_lists(df, col_text="text_to_model")
```

```python
# Split into Train and validation by patient

val_patients = df.loc[df.conf == "conf14"].patient_num
df["is_validation"] = df.patient_num.isin(val_patients)

print(
    "Number of examples of each set:\n", df.is_validation.value_counts(normalize=True)
)
```

```python
df.head(2)
```

```python
# Evaluation dans validetion set
df_val = df.loc[df.is_validation]
classification_report(
    y_true=df_val.label, y_pred=df_val.rule_based_prediction, output_dict=True
)
```

```python
df_val.label.value_counts()
```

```python
len(df)
```

```python tags=[]
# Create a Hugging Face dataset
_cols = [
    "words",
    "span_text",
    "word_label",
    "rule_based_prediction",
    "relative_index",
    col_label,
]
dataset_train = Dataset.from_pandas(df.loc[~df.is_validation, _cols])
dataset_val = Dataset.from_pandas(df.loc[df.is_validation, _cols])
datasets = DatasetDict({"train": dataset_train, "validation": dataset_val})
```

```python
# Exaple of data
example = next(iter(dataset_train))
example.keys()
```

```python
datasets
```

```python
1216 + 355
```

```python
type(datasets)
```

```python
# Save to disk
k1 = extraction_parameters["k1"]
k2 = extraction_parameters["k2"]
path_dataset = os.path.expanduser(
    f"~/cse_210013/data/datasets/annotation_results_clean_k1={k1}_k2={k2}"
)
path_dataset
```

```python
del df["text_to_model"]
```

# Add visit_start_datetime


```python tags=[]
from suicide_attempt.functions.retrieve_data import retrieve_stays
```

```python
visit_info = retrieve_stays(
    schema="cse_210013_20220201", encounter_subset=list(df.encounter_num.unique())
)
visit_info = (
    visit_info.select(["encounter_num", "start_date"])
    .withColumnRenamed("start_date", "visit_start_date")
    .toPandas()
)
```

```python
df = df.merge(visit_info, on="encounter_num", how="left", validate="many_to_one")
df.head()
```

```python tags=[]
visit_info2 = retrieve_stays(
    schema="cse_210013_20210726", encounter_subset=list("-1144987298496248335")
)
visit_info2 = (
    visit_info2.select(["encounter_num", "start_date"])
    .withColumnRenamed("start_date", "visit_start_date")
    .toPandas()
)
```

```python
df.loc[df.visit_start_date.isna()]
```

```python
df.head()
```

# Save

```python
df.to_pickle(
    os.path.expanduser(
        f"~/cse_210013/data/annotation/annotation_results_ts_clean_k1={k1}_k2={k2}.pickle"
    )
)
```

```python
datasets.save_to_disk(path_dataset)
```

```python
path = "/export/home/cse210013/cse_210013/data/annotation/annotation_results_ts_clean_k1=35_k2=10.pickle"
df = pd.read_pickle(path)
```

```python

```
