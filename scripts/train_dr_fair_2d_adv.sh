#!/bin/bash
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
MODEL_TYPE='efficientnet' # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-5 # 5e-5 for rnflt | 1e-4 for ilm
NUM_EPOCH=20 # 10
BATCH_SIZE=18 #( 10 12 14 16 18 20 ) best 6, best 18 for large scale 
STRETCH_RATIO=5 #( 0.5 1 2 5 10 26 ) # best 5
MODALITY_TYPE=('oct_bscans' ) # 'rpet' | 'fundus' | 'oct_bscans'
ATTRIBUTE_TYPE=( race gender hispanic ) # race|gender|hispanic
MODEL_TYPE=( efficientnet vit resnet ) # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
NEED_BALANCE=false
IMBALANCE_BETA=0.9999
IMBALANCE_BETA=-1
SPLIT_RATIO=1 # ( 0.2 0.3 .4 .5 .6 .8 .9 1 )
for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
do
for (( q=0; q<${#MODALITY_TYPE[@]}; q++ ));
do
for (( i=0; i<1; i++ ));
do
PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE[$q]}_${ATTRIBUTE_TYPE[$a]}.csv
python train_dr_fair_adversial.py \
		--data_dir ${PROJECT_DIR}/datasets/harvard/Harvard-DR/ \
		--result_dir ./results/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}_Adversial_Fair/fullysup_${MODEL_TYPE[$j]}_${MODALITY_TYPE}_Task${TASK}_beta${IMBALANCE_BETA} \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE[$q]} \
		--split_seed 5 \
		--imbalance_beta ${IMBALANCE_BETA} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE[$a]} \
		--split_ratio ${SPLIT_RATIO} \
		--need_balance ${NEED_BALANCE}
done
done
done
done