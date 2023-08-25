#!/bin/bash
set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONF_NAME=$1

python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'SA-ML' 150 --annotator-names benjamin --annotator-names vincent
python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'SA-RB' 40 --annotator-names benjamin --annotator-names vincent
python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'RF' 50 --annotator-names benjamin --annotator-names vincent


python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'SA-ML' 150 --annotator-names benjamin --annotator-names vincent --supplementary
python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'SA-RB' 40 --annotator-names benjamin --annotator-names vincent --supplementary
python $SCRIPT_DIR/../suicide_attempt/pipelines/generate_annotation_data_validation.py $CONF_NAME 'RF' 50 --annotator-names benjamin --annotator-names vincent --supplementary
