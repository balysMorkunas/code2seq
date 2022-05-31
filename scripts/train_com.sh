###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.
type=small
dataset_name=small-processed-comments
data_dir=data/small-processed-comments
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.val.c2s
model_dir=models/${type}

mkdir -p ${model_dir}
set -e
python -u code2seq.py --data ${data} --test ${test_data} --save_path ${model_dir} --model_path ${model_dir}
