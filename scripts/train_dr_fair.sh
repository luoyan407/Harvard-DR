#!/bin/bash
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-5
NUM_EPOCH=10
BATCH_SIZE=18
MODALITY_TYPE='oct_bscans' # 'rpet' | 'fundus' | 'oct_bscans'
ATTRIBUTE_TYPE=hispanic # race|gender|hispanic
MODEL_TYPE=( efficientnet vit resnet swin vgg resnext wideresnet convnext ) # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
	PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
	for (( i=0; i<1; i++ ));
	do
	python train_dr_fair.py \
			--data_dir ${PROJECT_DIR}/datasets/harvard/Harvard-DR/ \
			--result_dir ./results/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_baseline/fullysup_${MODEL_TYPE[$j]}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE} \
			--model_type ${MODEL_TYPE[$j]} \
			--image_size 200 \
			--loss_type ${LOSS_TYPE} \
			--lr ${LR} --weight-decay 0. --momentum 0.1 \
			--batch-size ${BATCH_SIZE} \
			--task ${TASK} \
			--epochs ${NUM_EPOCH} \
			--modality_types ${MODALITY_TYPE} \
			--perf_file ${PERF_FILE} \
			--attribute_type ${ATTRIBUTE_TYPE} 
	done
done