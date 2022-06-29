#!/bin/bash
for id in 0 1 2 3 4 5 6
do
        python ./one_experiment.py --dataset_name 14lap --id $id --save_dir_name hard
done
for id in 0 1 2 3 4 5 6
do
        python ./one_experiment.py --dataset_name 14res --id $id  --save_dir_name hard
done
for id in 0 1 2 3 4 5 6
do
        python ./one_experiment.py --dataset_name 15res --id $id --save_dir_name hard
done
for id in 0 1 2 3 4 5 6
do
        python ./one_experiment.py --dataset_name 16res --id $id  --save_dir_name hard
done