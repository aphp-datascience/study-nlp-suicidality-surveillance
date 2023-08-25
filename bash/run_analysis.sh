set -e
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export ARROW_LIBHDFS_DIR=/usr/local/hadoop/usr/lib/
export HADOOP_HOME=/usr/local/hadoop/
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

CONF_NAME=$1
# Print date
date

# Stay document selection
bash $SCRIPT_DIR/spark_submit.sh $SCRIPT_DIR/../suicide_attempt/pipelines/stay_document_selection.py $CONF_NAME
hdfs dfs -ls cse_210013/pipeline_results/

# Entity classification Rule based
python $SCRIPT_DIR/../suicide_attempt/pipelines/ent_classification_rule_based.py $CONF_NAME

# Entity classification ML
sbatch $SCRIPT_DIR/inference_ml_model.sh $CONF_NAME

# Stay classification (text data)
python $SCRIPT_DIR/../suicide_attempt/pipelines/stay_classification_text_data.py $CONF_NAME

# Stay classification (claim data)
bash $SCRIPT_DIR/spark_submit.sh $SCRIPT_DIR/../suicide_attempt/pipelines/stay_classification_claim_data.py $CONF_NAME

# Risk factor document selection
bash $SCRIPT_DIR/spark_submit.sh $SCRIPT_DIR/../suicide_attempt/pipelines/rf_document_selection.py $CONF_NAME 

# Risk factor entity classification (rule based)
python $SCRIPT_DIR/../suicide_attempt/pipelines/ent_classification_rule_based.py $CONF_NAME --regex rf --file-name-in rf_document_selection_ --file-name-out result_classification_rf_

# Make plots
python $SCRIPT_DIR/../suicide_attempt/pipelines/make_plots.py  $CONF_NAME

# Validation metrics (table 2) and age description
python $SCRIPT_DIR/../suicide_attempt/pipelines/validation_and_age_description.py $CONF_NAME

# Print date
date
