from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, CamembertTokenizerFast


def extract_text_window(doc, k1=35, k2=10, mode="symmetric"):
    """
    Function to extract a text window of k tokens around the entity.
    It supposes one entity by document.

    Parameters
    ----------
    doc: spacy document
    k: number of tokens before and after the entity to extract from the document.

    Returns
    -------
    text_to_model: span of text that will be the input to the model
    """
    ent = doc.ents[0]

    if mode == "symmetric":
        text_to_model = doc[max(0, ent.start - k1) : ent.end + k1]
    elif mode == "asymmetric":
        text_to_model = doc[max(0, ent.start - k1) : ent.end + k2]

    return text_to_model


def split_into_lists(df, col_text="text_to_model"):
    df["words"] = df[col_text].apply(lambda doc: [token.text for token in doc])

    if col_text == "text_to_model":
        df["word_label"] = df[col_text].apply(
            lambda doc: [token._.word_label for token in doc]
        )

    return df


def add_ents_to_doc(df, col_label, label_all_tokens=False):
    # Add a span information for each document.
    for _, row in df.iterrows():
        doc = row.doc

        spans = []
        span = doc.char_span(
            row["offset_begin"],
            row["offset_end"],
            label=row["lexical_variant"],
            alignment_mode="expand",
        )
        spans.append(span)

        if label_all_tokens:
            for token in span:
                token._.word_label = int(row[f"{col_label}"])
        else:
            token = span[0]
            token._.word_label = int(row[f"{col_label}"])

        doc.ents = spans


def collate_fn_inference(batch):
    words = [item["words"] for item in batch]

    if "word_index" in batch[0].keys():
        word_index = [item["word_index"] for item in batch]
        return dict(words=words, word_index=word_index)
    else:
        return dict(words=words)


def tokenize_and_align_labels(
    text: Union[List[str], List[List[str]]],
    tags: Union[List[int], List[List[int]], List[str], List[List[str]]],
    tokenizer: CamembertTokenizerFast,
    ids: Union[List[int], List[List[int]], List[str], List[List[str]]] = None,
    truncation: bool = True,
    is_split_into_words: bool = True,
    padding: bool = True,
    max_length: int = 512,
    label_all_tokens: bool = False,
    tag2id: Dict = None,
) -> BatchEncoding:
    """
    Function that tokenizes a given text and aligns the corresponding tags, according
    to the alignement strategy `label_all_tokens`.
    Inspired from the HuggingFace library.

    Parameters
    ----------
    text: Union[List[str], List[List[str]]]
        List of raw text (if `is_split_into_words = False`) or pretokenized text
        (if `is_split_into_words = True`) to be tokenized.
    tags: Union[List[int], List[List[int]], List[str], List[List[str]]]
        List of tags to assign to each word in the text.
    tokenizer: CamembertTokenizerFast
        Huggingface tokenizer to use for text tokenization.
    ids: Union[List[int], List[List[int]], List[str], List[List[str]]]
        List of ids to assign to each example.
    truncation: bool
        Whether to truncate long texts during tokenization or not.
    is_split_into_words: bool
        If the given text is already split into words (with pre-tokenization).
    padding: bool
        Whether to add padding during tokenization or not.
    max_length: int
        Maximum length of the returned tokenized sequences.
    label_all_tokens: bool
        Whether to label all tokens corresponding to a word or only the first token
        of each word.
    tag2id: Dict
        Dictionnary to map tags to their id if tags are given as `str`
        (rather than `int`).

    Returns
    -------
    tokenized_inputs: batch of tokenized text and labels aligned with the tokenization.
    """

    tokenized_inputs = tokenizer(
        text,
        truncation=True,
        is_split_into_words=is_split_into_words,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = []

    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100
            # so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if tag2id is not None:
                    label_ids.append(tag2id[label[word_idx]])
                else:
                    label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the
            # current label or -100, depending on
            # the label_all_tokens flag.
            else:

                if tag2id is not None:
                    label_ids.append(
                        tag2id[label[word_idx]] if label_all_tokens else -100
                    )
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = torch.tensor(labels)
    if ids is not None:
        tokenized_inputs["original_ids"] = ids
    return tokenized_inputs


class PLDataset(Dataset):
    """
    Custom dataset to be given as input of a PytorchLightning module.

    Parameters
    ----------
    tokens: Dict[str, List[Any]]
        Dictionnary of tokens and their attributes.
    """

    def __init__(self, tokens: Dict[str, List[Any]]):
        self.tokens = tokens

    def __len__(self):
        key = list(self.tokens.keys())[0]
        return len(self.tokens[key])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.tokens.items()}
        return item
