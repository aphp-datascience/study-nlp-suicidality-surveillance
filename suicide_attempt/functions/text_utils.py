import re
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import spacy
from edsnlp.pipelines.misc.dates.models import RelativeDate
from pendulum.datetime import DateTime
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from suicide_attempt.functions.constants import snippet_window_length


def pattern_matcher(pattern: str, window=20):
    """

    Parameters
    ----------
    pattern : string
        Regex pattern
    window : int, default=20
        Number of characters to keep before and after the match

    Returns
    -------
    df : function
        Function to match the text
    """

    def matcher(text):
        matches = []
        for match in re.finditer(pattern, text):

            lexical_variant = match.group()
            start = match.start()
            end = match.end()

            snippet = text[max(0, start - window) : end + window]  # noqa: E203

            matches.append(
                dict(
                    lexical_variant=lexical_variant,
                    start=start,
                    end=end,
                    snippet=snippet,
                )
            )
        return matches

    return matcher


def spark_filter_docs_by_regex(documents, regex_pattern_dictionary) -> DataFrame:
    """
    Filter documents that match a lowercase
    regex pattern or a case sensitive regex pattern.

    Parameters
    ----------
    documents: spark DataFrame
        the filter will be done over the column `note_text`
    regex_pattern_dictionary: dictionary
        A regex_pattern_dictionary of regex patterns to match.
        All items of the list will be joined with the OR operator.

    Returns
    -------
    documents_match_regexp: spark DataFrame with the rows that fulfill the conditions.

    """

    # Join all the terms wiht the OR operator
    regex_patterns = regex_pattern_dictionary.values()
    regex_patterns = [f"({i})" for pattern in regex_patterns for i in pattern]
    regex_patterns = r"|".join(regex_patterns)

    # Filter documents
    documents_match_regexp = documents.where(col("note_text").rlike(regex_patterns))

    return documents_match_regexp


def _use_sections_getter(nlp) -> bool:
    """Returns True if "sections" is in the pipeline

    Parameters
    ----------
    nlp : spacy Language object

    Returns
    -------
    bool
    """
    return "sections" in nlp.pipe_names


def _direction_getter(date):
    if type(date._.date) is RelativeDate:
        return date._.date.direction.value
    else:
        return None


def _to_datetime_getter(date, visit_start_date):
    try:
        dt = date._.date.to_datetime(
            note_datetime=visit_start_date, infer_from_context=True, tz=None
        )

        if isinstance(dt, (datetime, DateTime)):
            return dt
        else:
            return None

    except:  # noqa: E722
        return None


def nlp_baseline(regex: str, attr_text: str = "NORM"):
    """
    Define the NLP pipeline to:
        - Split the text in sentences
        - Run Name entity recognition algorithm (suicide attempts)
        - Identify a negation structure that modifies each lexical variant
        - Identify an hypothesis structure that modifies each lexical variant
        - Identify a family context structure that modifies each lexical variant
        - Identify an history structure that modifies each lexical variant
        - Identify an reported speech structure that modifies each lexical variant
        - Detect mentions of dates


    Parameters
    ----------
    regex: str, a regex of lexical variants to match
    attr_text: str, default="NORM". Attribute of text to match

    Returns
    -------
    nlp: a Spacy pipeline that could be applied to texts
    """

    # Initialize a tokenizer
    nlp = spacy.blank("eds")
    # nlp.tokenizer = custom_tokenizer(nlp)

    # Add a pipeline of NLP operations ##
    if attr_text == "NORM":
        # Normalize text
        nlp.add_pipe(
            "eds.normalizer",
            config=dict(
                accents=True,
                lowercase=True,
                quotes=True,
                pollution=True,
            ),
        )
        # Split into sections
        nlp.add_pipe("eds.sections", config=dict(attr=attr_text))

    # Split text into sentences
    nlp.add_pipe("eds.sentences")
    # Identify the lexical variants of interest
    nlp.add_pipe("eds.matcher", config=dict(regex=regex, attr="TEXT"))
    # Classify the lexical variants as negated
    nlp.add_pipe("eds.negation", config=dict(attr=attr_text))
    # Classify the lexical variants as hypothesis
    nlp.add_pipe("eds.hypothesis", config=dict(attr=attr_text))
    # Classify the lexical variants as Family Context
    nlp.add_pipe("eds.family", config=dict(attr=attr_text))
    # Classify the lexical variants as history
    nlp.add_pipe(
        "eds.history",
        config=dict(attr=attr_text, use_sections=_use_sections_getter(nlp)),
    )
    # Classify the lexical variants as Reported Speech
    nlp.add_pipe("eds.reported_speech", config=dict(attr=attr_text))
    # Identify dates
    nlp.add_pipe("eds.dates")

    return nlp


def pick_dict(ent, _snippet) -> Dict[str, Any]:
    """
    Function to return different attributes of an entity as a dictionary
    Useful to export to a pandas dataframe

    Parameters
    ----------
    ent : entity
    _snippet : span around the entity

    Returns
    -------
    Dictionary with different attributes of the entity
    """

    # Convert _snippet (span) to a list
    words = [token.text for token in _snippet]

    return dict(
        note_id=ent.doc._.note_id,
        span_type="ent",
        lexical_variant=ent.text,
        std_lexical_variant=ent.label_,
        start=ent.start,
        start_char=ent.start_char,
        end=ent.end,
        end_char=ent.end_char,
        snippet=_snippet.text,
        words=words,
        word_index=ent.start - _snippet.start,
        start_char_snippet_relative=ent.start_char - _snippet.start_char,
        end_char_snippet_relative=ent.end_char - _snippet.start_char,
        negated=ent._.negation,
        family=ent._.family,
        history=ent._.history,
        hypothesis=ent._.hypothesis,
        rspeech=ent._.reported_speech,
        sent_id=ent.sent.start,
    )


def pick_date(date, _snippet) -> Dict[str, Any]:
    """
    Function to return different attributes of a date as a dictionary
    Useful to export to a pandas dataframe

    Parameters
    ----------
    date : span of date

    Returns
    -------
    Dictionary with different attributes of the entity
    """

    if date.doc.has_extension("visit_start_date"):
        visit_start_date = date.doc._.get("visit_start_date")
    else:
        visit_start_date = None

    return dict(
        note_id=date.doc._.note_id,
        span_type="date",
        lexical_variant=date.text,
        std_lexical_variant=date.label_,
        snippet=_snippet.text,
        date_dt=_to_datetime_getter(date, visit_start_date),
        year=date._.date.year,
        month=date._.date.month,
        day=date._.date.day,
        direction=_direction_getter(date),
        sent_id=date.sent.start,
        start=date.start,
        start_char=date.start_char,
        end=date.end,
        end_char=date.end_char,
    )


def pick_results(doc, k1=35, k2=10):
    """
    Function used well Paralellizing tasks via joblib
    This functions will store all extracted entities
    """
    ents = []
    for ent in doc.ents:
        _snippet = doc[max(0, ent.start - k1) : ent.end + k2]  # noqa: E203
        # _snippet = ent.sent
        ents.append(pick_dict(ent, _snippet))

    dates = []
    if (len(ents) > 0) & ("dates" in list(doc.spans.keys())):
        dates = []
        for date in doc.spans["dates"]:
            _snippet = doc[
                max(0, date.start - snippet_window_length) : date.end  # noqa: E203
                + snippet_window_length
            ]
            dates.append(pick_date(date, _snippet))

    return ents + dates


def lv_co_occurrence(df, output_format="wide"):
    """
    Function to compute the matrix of co-occurrence from a dataframe containing
    the columns 'encounter_num' and 'std_lexical_variant'.

    Parameters
    ----------
    df: pd.DataFrame,
        It should have the columns `std_lexical_variant` and `encounter_num`
    output_format: one of {'wide','long','list'},
        The desired output format

    Returns
    -------
    dfr: an upper diagonal co-occurrence matrix
    """

    # Keep one line by visit & std_lexical_variant
    df_ = df.drop_duplicates(["encounter_num", "std_lexical_variant"]).copy()

    # Add a column to count the presence of the lexical variant
    df_["present"] = True

    # Pivot table to have one line by visit and
    # one boolean column by std_lexical_variant
    dfp = df_.pivot(
        index="encounter_num", columns="std_lexical_variant", values="present"
    )

    # df = df.drop(columns=['present'])

    # Fill na with False value
    dfp.fillna(False, inplace=True)

    # Count the co occurrence of the std lexical variants
    col_names = dfp.columns
    length = len(col_names)
    r = []

    for col1 in range(0, length):
        for col2 in range(col1, length):

            n = len(dfp.loc[(dfp.iloc[:, col1]) & (dfp.iloc[:, col2])])
            r.append((col_names[col1], col_names[col2], n))

    # Count visits that are identified only by one std lexical variant
    r_O = []
    for col1 in range(0, length):
        n = len(
            dfp.loc[
                (dfp.iloc[:, col1])
                & (~(dfp.loc[:, dfp.columns != col_names[col1]]).max(axis=1))
            ]
        )

        r_O.append((col_names[col1], "O", n))

    if output_format == "list":
        return r, r_O

    # Make a df from the list.
    # `n` is the number of visits that are identified by the pair (lv1,lv2)
    dfr = pd.DataFrame(r, columns=["lv1", "lv2", "n"])
    dfr_O = pd.DataFrame(r_O, columns=["lv1", "lv2", "n"])

    if output_format == "long":
        return dfr, dfr_O

    # Pivot table to have an upper-diagonal matrix of co-occurrence
    dfr = dfr.pivot("lv1", "lv2", "n")
    dfr.fillna(0, inplace=True)
    dfr = dfr.astype("int")

    dfr_O = dfr_O.pivot("lv1", "lv2", "n")
    dfr_O.fillna(0, inplace=True)
    dfr_O = dfr_O.astype("int")

    return dfr, dfr_O


def count_visits_by_lv(df):
    """
    Function to retrieve the histogram of number of detected visits by lexical variant.
    Note than visits could be counted in multiple categories at the same time.

    Parameters
    ----------
    df: pd.DataFrame,
        It should have the columns `std_lexical_variant` and `encounter_num`

    Returns
    -------
    hist: pd.DataFrame,
        Histogram of number of detected visits by lexical variant.
        Note than visits could be counted in multiple categories at the same time.
    """
    hist = (
        df.groupby("std_lexical_variant")
        .agg(n_visits=("encounter_num", "nunique"))
        .sort_values("n_visits", ascending=False)
    )
    return hist
