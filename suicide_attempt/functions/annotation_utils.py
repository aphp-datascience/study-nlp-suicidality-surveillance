import os
from typing import List, Optional

import nbformat as nbf
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Output  # type: ignore
from labeltool.labelling import GlobalLabels, Labelling, Labels  # type: ignore
from pyspark.sql import functions as F

import suicide_attempt.functions.utils as utils
from suicide_attempt.functions.constants import covid_date, regex_rf
from suicide_attempt.functions.data_wrangling import (
    import_patient_rf_data,
    reduce_rule_based_attributes,
)
from suicide_attempt.functions.stats_utils import add_rf_data, get_sa_data
from suicide_attempt.functions.utils import (
    extract_global_labels,
    get_conf,
    initiate_spark,
)

out = Output()

spark, sql = initiate_spark()


class ValidationDatasetCreation:
    def __init__(
        self,
        conf_name: str,
        annotator_names: List[str],
        n1: int,
        share_p: float = 0.1,
        avoid_list: Optional[List[str]] = None,
        annotation_subset: str = "SA-ML",
        supplementary: Optional[str] = None,
    ):

        self.conf_name = conf_name
        self.annotator_names = annotator_names
        self.share_p = share_p

        self.n1 = n1

        self.annotation_subset = annotation_subset

        self.avoid_list = avoid_list

        self.parameters = get_conf(conf_name)

        assert annotation_subset in ["SA-ML", "SA-RB", "RF"]
        if annotation_subset == "SA-ML":
            self.stays = get_sa_data(
                self.conf_name, source="text_ml", keep_only_positive=True
            )
            self.stays = add_rf_data(self.stays, self.conf_name)

        if annotation_subset == "SA-RB":
            self.stays = self._get_rb_stays()

        if annotation_subset == "RF":
            self.stays = self._get_rf_stays()

        self.stays = self._preprocess(self.stays)

    def _get_rf_stays(self):
        stays = get_sa_data(self.conf_name, source="text_ml", keep_only_positive=True)
        stays = add_rf_data(
            stays,
            self.conf_name,
        )

        stays_rf = []
        keys = list(regex_rf.keys())
        keys.append("has_history")
        for key in keys:
            rf = stays.loc[stays[key]].copy()
            rf["rf_name"] = key
            stays_rf.append(rf)

        return pd.concat(stays_rf)

    def _get_rb_stays(self):
        # Get stays positively calssified by the Rule based algorithm
        stays = get_sa_data(
            conf_name=self.conf_name,
            keep_only_positive=True,
            source="text_rule_based",
        )

        # Get stays positively calssified by the ML algorithm
        stays_ml = get_sa_data(
            source="text_ml",
            conf_name=self.conf_name,
            keep_only_positive=True,
        )
        stays_ml = stays_ml[["encounter_num", "nlp_positive"]].copy()
        stays_ml.rename(columns={"nlp_positive": "ml_positive"}, inplace=True)

        # Merge both
        stays = stays.merge(stays_ml, on="encounter_num", how="left")

        # Keep stays where the ML algorithm is not True (False or None)
        stays = stays[~stays.ml_positive.fillna(False)].copy()

        return stays

    def sample_stays(self):
        cols = [
            "visit_start_date",
            "encounter_num",
            "note_id",
            "patient_num",
            "after_covid_outbreak",
        ]

        merge_cols = ["note_id"]
        if self.annotation_subset == "RF":
            gb_cols = ["after_covid_outbreak", "rf_name"]
            cols.append("rf_name")
            merge_cols.append("rf_name")
        else:
            gb_cols = ["after_covid_outbreak"]

        if self.avoid_list is not None:
            avoid_list = list(self.avoid_list)
            print("Length of avoid list", len(avoid_list))
            sample_sa_stays = (
                self.stays.loc[~self.stays.encounter_num.isin(avoid_list)]
                .groupby(gb_cols)
                .sample(int(self.n1 / 2))
            )
            print("Length of sample_sa_stays", len(sample_sa_stays))
            assert sum(sample_sa_stays.encounter_num.isin(avoid_list)) == 0
        else:
            sample_sa_stays = self.stays.groupby(gb_cols).sample(int(self.n1 / 2))

        sample_sa_stays = sample_sa_stays[cols]

        sample_sa_stays["annotation_subset"] = self.annotation_subset

        if self.annotation_subset == "SA-ML":
            results_ent = self._get_ml_entity_results()
        if self.annotation_subset == "SA-RB":
            results_ent = self._get_rule_based_entity_results()
        if self.annotation_subset == "RF":
            results_ent = self._get_rf_entity_results()

        sample_sa_ent = results_ent.merge(
            sample_sa_stays, on=merge_cols, how="inner", validate="many_to_one"
        )

        sample_sa_ent[f"{self.annotation_subset} stay"] = True

        sample_sa_ent = self.process_for_annotation_util(sample_sa_ent)

        stay_ids = sample_sa_stays[["encounter_num"]].drop_duplicates()

        distribution = self._split_dataset_into_annotators_v2(stay_ids=stay_ids)
        assert len(stay_ids) == distribution.encounter_num.nunique()
        assert len(stay_ids) < len(distribution)

        sample_sa_ent = sample_sa_ent.merge(
            distribution, on="encounter_num", how="inner", validate="many_to_many"
        )

        return sample_sa_ent

    def _preprocess(self, df):
        hospitals_test = self.parameters["hospitals_test"]  # noqa: F841
        df = df.query("hospital_label.isin(@hospitals_test)").copy()
        df["after_covid_outbreak"] = df.date >= covid_date
        df = df.explode(column="note_ids")
        df.rename(columns={"note_ids": "note_id"}, inplace=True)

        return df

    def _get_ml_entity_results(self):
        results_bert = pd.read_pickle(
            os.path.expanduser(
                f"~/cse_210013/data/{self.conf_name}/result_ent_classification_ml_{self.conf_name}"  # noqa: E501
            )
        )

        return results_bert

    def _get_rule_based_entity_results(self):
        file_path = os.path.expanduser(
            f"~/cse_210013/data/{self.conf_name}/result_ent_classification_rule_based_{self.conf_name}"  # noqa: E501
        )

        results_rb = pd.read_pickle(file_path)

        results_rb = reduce_rule_based_attributes(results_rb, "base", verbose=False)
        results_rb = results_rb[
            ["note_id", "start_char", "end_char", "rule_based_prediction"]
        ]
        return results_rb

    def _get_rf_entity_results(self):

        rf_entities = import_patient_rf_data(
            self.conf_name,
            subcats=list(regex_rf.keys()),
            text_modifers=None,
        )

        rf_entities = reduce_rule_based_attributes(rf_entities, method="rf")

        rf_entities = rf_entities[
            [
                "std_lexical_variant",
                "note_id",
                "start_char",
                "end_char",
                "rule_based_prediction",
            ]
        ]

        rf_entities.rename(
            columns={
                "std_lexical_variant": "rf_name",
            },
            inplace=True,
        )

        # SA History
        file_path = os.path.expanduser(
            f"~/cse_210013/data/{self.conf_name}/result_ent_classification_rule_based_{self.conf_name}"  # noqa: E501
        )

        results_rb = pd.read_pickle(file_path)
        results_rb["rule_based_prediction"] = np.logical_and(
            np.logical_not(
                np.logical_or.reduce(
                    (results_rb.negated, results_rb.family, results_rb.hypothesis)
                )
            ),
            results_rb.history,
        )
        results_rb["rf_name"] = "has_history"
        results_rb = results_rb[
            ["rf_name", "note_id", "start_char", "end_char", "rule_based_prediction"]
        ]

        rf_entities = pd.concat([rf_entities, results_rb])
        return rf_entities

    def get_documents(self, docs_ids_pd):
        # Use the same docs retrieved in pipe 'stay_document_selection'
        documents = spark.read.parquet(
            f"cse_210013/pipeline_results/stay_document_selection_{self.conf_name}.parquet"  # noqa: E501
        )
        cols = ["note_id", "concept_cd", "note_text"]
        documents = documents.select(cols)

        documents = documents.where(
            F.col("note_id").isin(list(docs_ids_pd.note_id.unique()))
        )

        return documents.toPandas()

    @staticmethod
    def data_description(df):
        docs_by_annotator = (
            df.drop_duplicates(subset=["note_id_dedup", "annotator"])
            .groupby("annotator")
            .size()
        )

        n_unique_notes = df.note_id.nunique()
        shared_notes = df.query("n_annotator>1").note_id.nunique()

        print("----- ### Description ### -----")
        print("Number of unique notes:", n_unique_notes)
        print("Number of shared notes:", shared_notes)
        print("Distribution of notes by annotator:\n", docs_by_annotator)

    def get_data(self):

        # Get entities
        sample_sa_stays = self.sample_stays()
        self.data_description(sample_sa_stays)

        ents = pd.concat(
            [sample_sa_stays],
        )

        # Get Documents
        notes = self.get_documents(ents[["note_id"]].drop_duplicates())

        ents = ents.merge(notes, on="note_id", how="left", validate="many_to_one")

        return ents

    def process_for_annotation_util(self, df):
        # Rename columns to work with the annotation util
        df.rename(
            columns={
                "start_char": "offset_begin",
                "end_char": "offset_end",
                # "annotation_subset": "label_name",
                "bert_token_prediction": "label_value",
                "rule_based_prediction": "label_value",
            },
            inplace=True,
        )

        # Cast to int
        df.offset_begin = df.offset_begin.astype("int")
        df.offset_end = df.offset_end.astype("int")
        df.label_value = df.label_value.astype("bool")

        # Concat
        df["label_name"] = df.annotation_subset + "-" + df.label_value.astype(str)
        if self.annotation_subset == "RF":
            df["title"] = (
                "Début du séjour: "
                + df.visit_start_date.dt.date.astype(str)
                + f" - Tâche : {self.annotation_subset} "
                + df.rf_name
            )
            df["note_id_dedup"] = df["note_id"] + "-" + df["rf_name"]
        else:
            df["title"] = (
                "Début du séjour: "
                + df.visit_start_date.dt.date.astype(str)
                + f" - Tâche : {self.annotation_subset}"
            )
            df["note_id_dedup"] = df["note_id"]
        return df

    def _split_dataset_into_annotators(self, stay_ids):
        """
        stay_ids : vector of unique identifiers
        """

        n_common_visits = int(len(stay_ids) * self.share_p)

        # Check that there are enough visits in common
        assert n_common_visits <= len(stay_ids)
        print("Number of common stays: ", n_common_visits)

        # Generate a vector to assign randomly each encounter_num to an annotator
        _annot_dist1 = np.random.choice(self.annotator_names, size=len(stay_ids))

        # Generate a vector to assign randomly each encounter_num to a second annotator
        _annot_dist2 = np.random.choice(self.annotator_names, size=len(stay_ids))

        # Keep n_common_visits where annotator1 != annotator2
        predistribution = pd.DataFrame(
            {
                "annotator1": _annot_dist1,
                "annotator2": _annot_dist2,
                "encounter_num": stay_ids.encounter_num,
            }
        )
        distribution_common = predistribution.loc[
            predistribution.annotator1 != predistribution.annotator2
        ].iloc[:n_common_visits]

        # Visits non selected
        _enc_num2 = stay_ids.loc[
            ~stay_ids.encounter_num.isin(distribution_common.encounter_num)
        ].encounter_num.unique()

        # Asign these visits to annotators
        _annot_dist3 = np.random.choice(self.annotator_names, size=len(_enc_num2))
        distribution_non_common = pd.DataFrame(
            {"annotator": _annot_dist3, "encounter_num": _enc_num2}
        )

        # Convert the df 'distribution_common' to long format
        distribution_common["annotator"] = None

        distribution_common1 = distribution_common[["encounter_num", "annotator1"]]
        distribution_common1.columns = ["encounter_num", "annotator"]

        distribution_common2 = distribution_common[["encounter_num", "annotator2"]]
        distribution_common2.columns = ["encounter_num", "annotator"]

        # Final distribution (concatenation of )
        final_dist = pd.concat(
            [distribution_common1, distribution_common2, distribution_non_common]
        )

        # Count how many annotators by visit
        count_annotators = pd.DataFrame(
            data=final_dist.groupby("encounter_num")["annotator"].nunique()
        )
        count_annotators.columns = ["n_annotator"]
        count_annotators.reset_index(inplace=True)

        # Merge this info
        final_dist = final_dist.merge(count_annotators, on="encounter_num")
        final_dist.reset_index(inplace=True, drop=True)

        return final_dist

    def _split_dataset_into_annotators_v2(self, stay_ids):
        """
        stay_ids : vector of unique identifiers
        """
        # Randomize
        stay_ids = stay_ids.sample(frac=1.0)

        # Select shared stays
        n_shared_stays = int(len(stay_ids) * self.share_p)

        # Split into shared stays and not shared
        shared_stays = stay_ids.iloc[:n_shared_stays].encounter_num.values
        non_shared_stays = stay_ids.iloc[n_shared_stays:].encounter_num.values

        # Distribution of shared stays
        annot_dist_shared = np.tile(self.annotator_names, n_shared_stays)
        enc_dist_shared = np.repeat(shared_stays, len(self.annotator_names))

        distribution_shared = pd.DataFrame(
            {"annotator": annot_dist_shared, "encounter_num": enc_dist_shared}
        )
        distribution_shared["n_annotator"] = len(self.annotator_names)

        # Distribution of non shared
        annot_dist_non_shared = np.resize(self.annotator_names, len(non_shared_stays))
        distribution_non_shared = pd.DataFrame(
            {"annotator": annot_dist_non_shared, "encounter_num": non_shared_stays}
        )

        distribution_non_shared["n_annotator"] = 1

        # Final distribution (concatenation of )
        final_dist = pd.concat([distribution_shared, distribution_non_shared])

        final_dist.reset_index(inplace=True, drop=True)

        return final_dist

    def generate_annotation_notebooks(
        self,
    ):
        nb = nbf.v4.new_notebook()

        code1 = (
            """from suicide_attempt.functions.annotation_utils import ValidationTool"""
        )

        text1 = """# Parameters"""

        text2 = """# Annotation tool"""
        code3 = """vt.LabellingTool.run()"""
        text3 = """# Read annotations"""

        code4 = """\
results = vt.read_annotations()
results
"""
        for annotator in self.annotator_names:
            code2 = f"""\
params = dict(
    annotator='{annotator}',
    annotation_subset='{self.annotation_subset}',
    from_save=True,
    conf_name='{self.conf_name}',
    supplementary=False,
    display_height=400,
)
vt = ValidationTool(**params)
"""
            nb["cells"] = [
                nbf.v4.new_code_cell(code1),
                nbf.v4.new_markdown_cell(text1),
                nbf.v4.new_code_cell(code2),
                nbf.v4.new_markdown_cell(text2),
                nbf.v4.new_code_cell(code3),
                nbf.v4.new_markdown_cell(text3),
                nbf.v4.new_code_cell(code4),
            ]

            kernelspec = {
                "name": "kernel_ts_local",
                "language": "python",
                "display_name": "kernel_ts_local",
            }

            nb.metadata["kernelspec"] = kernelspec
            fname = "test.ipynb"

            fname = os.path.expanduser(
                f"~/cse_210013/notebooks/annotation_validation/{annotator}/{annotator}_{self.annotation_subset}.ipynb"  # noqa: E501
            )

            _directory = utils.get_dir_path(fname)
            if not os.path.isdir(_directory):
                os.makedirs(_directory)

            with open(fname, "w") as f:
                nbf.write(nb, f)


class ValidationTool:
    def __init__(
        self,
        annotator,
        annotation_subset,
        conf_name,
        supplementary=False,
        from_save=False,
        display_height=500,
        labels_height=0,
        window_snippet=50,
        sort_column=None,
        reverse=False,
        add_grid=False,
        add_fn_handling=False,
    ):

        path_raw = os.path.expanduser(
            f"~/cse_210013/data/annotation/validation/raw_validation/{conf_name}/{annotation_subset}/raw_{annotation_subset}_{conf_name}"  # noqa: E501
        )

        save_path = os.path.expanduser(
            f"~/cse_210013/data/annotation/validation/annotated/{conf_name}/{annotation_subset}/{annotation_subset}_{conf_name}_{annotator}.pickle"  # noqa: E501
        )

        if supplementary:
            path_raw += "_supplementary"
            save_path = os.path.expanduser(
                f"~/cse_210013/data/annotation/validation/annotated/{conf_name}/{annotation_subset}/{annotation_subset}_{conf_name}_{annotator}_supplementary.pickle"  # noqa: E501
            )
        data = pd.read_pickle(path_raw)
        data = data.query(f"annotator=='{annotator}'")
        assert len(data) > 0

        _directory = utils.get_dir_path(save_path)
        if not os.path.isdir(_directory):
            os.makedirs(_directory)

        print("Results will be saved at:\n", save_path)
        self.save_path = save_path

        self.mapping = data[["label_name", "label_value"]].drop_duplicates()

        self.ref = {
            "SA-ML": {"label_name": "SA-ML Gold Standard", "label_value": "SA-ML stay"},
            "SA-RB": {"label_name": "SA-RB Gold Standard", "label_value": "SA-RB stay"},
            "RF": {"label_name": "RF Gold Standard", "label_value": "RF stay"},
        }

        self.label_name = self.ref[annotation_subset]["label_name"]
        self.label_value = self.ref[annotation_subset]["label_value"]
        self.annotation_subset = annotation_subset

        self.LabellingTool = Labelling(
            data,
            save_path=save_path,
            groupby_col="note_id_dedup",
            labels_dict=self.get_labels_ents(),
            global_labels_dict=self.get_labels_doc(),
            global_labels_only=False,
            from_save=from_save,
            display_height=display_height,
            out=out,
            display=display,
            labels_height=labels_height,
            window_snippet=window_snippet,
            sort_column=sort_column,
            add_grid=add_grid,
            add_fn_handling=add_fn_handling,
            reverse=reverse,
        )

    @staticmethod
    def _get_color(_class):
        if _class is False:
            return "#FC7457"  # red
        if _class is True:
            return "#0AC998"  # green
        else:
            return "#5EB4DC"  # blue

    def get_labels_ents(self):
        labels = Labels()
        for std_lexical_variant, _class in self.mapping.values:

            labels.add(
                name=std_lexical_variant,
                color=self._get_color(_class),
                selection_type="button",
            )

        return labels.dict

    def get_labels_doc(self):
        # Add labels to the util
        global_labels = GlobalLabels()

        global_labels.add(
            name=self.label_name, selection_type="check", value=self.label_value
        )
        global_labels.add(name="Remarque", selection_type="text")

        return global_labels.dict

    def read_annotations(self):
        df = pd.read_pickle(self.save_path)
        df = df.loc[df.vu].copy()
        df = extract_global_labels(df)
        df.drop_duplicates("note_id_dedup", inplace=True)
        df.reset_index(inplace=True, drop=True)

        cols = [
            "note_id",
            "note_id_dedup",
            "visit_start_date",
            "encounter_num",
            "patient_num",
            "after_covid_outbreak",
            "annotation_subset",
            self.label_value,
            "annotator",
            "n_annotator",
            "concept_cd",
            self.label_name,
            "Remarque",
        ]
        if self.annotation_subset == "RF":
            cols.append("rf_name")
        return df[cols]
