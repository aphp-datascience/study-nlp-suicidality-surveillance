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
    display_name: kernel_ts_local
    language: python
    name: kernel_ts_local
---

```python
from suicide_attempt.functions.annotation_utils import ValidationTool
```

# Parameters

```python
params = dict(
    annotator='vincent',
    annotation_subset='SA-RB',
    from_save=True,
    conf_name='conf_article',
    supplementary=False,
    display_height=400,
)
vt = ValidationTool(**params)

```

# Annotation tool

```python
vt.LabellingTool.run()
```

# Read annotations

```python
results = vt.read_annotations()
results

```
