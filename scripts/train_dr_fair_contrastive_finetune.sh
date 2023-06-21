#!/bin/bash
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
MODEL_TYPE='efficientnet' # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-5 # 5e-5 for rnflt | 1e-4 for ilm
NUM_EPOCH=10 # 10
BATCH_SIZE=18 #( 10 12 14 16 18 20 ) best 6, best 18 for large scale 
STRETCH_RATIO=5 #( 0.5 1 2 5 10 26 ) # best 5
MODALITY_TYPE='fundus' # 'rpet' | 'fundus' | 'oct_bscans'
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
MODEL_TYPE=vit # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
NEED_BALANCE=false
CONT_TYPE=FSCL
ATTRIBUTE_TYPE=hispanic # race|gender|hispanic
IMBALANCE_BETA=-1
SPLIT_RATIO=1 # ( 0.2 0.3 .4 .5 .6 .8 .9 1 )
for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( q=0; q<${#ATTRIBUTE_TYPE[@]}; q++ ));
do
python train_dr_fair.py \
		--data_dir ${PROJECT_DIR}/datasets/harvard/Harvard-DR/ \
		--result_dir ./results_fair_cont_finetune/${MODALITY_TYPE}_cont_finetune_${CONT_TYPE}_${ATTRIBUTE_TYPE[$q]}_${MODEL_TYPE[$j]} \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--task ${TASK} \
		--cont_method ${CONT_TYPE} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--split_seed 5 \
		--imbalance_beta ${IMBALANCE_BETA} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE[$q]} \
		--split_ratio ${SPLIT_RATIO} \
		--need_balance ${NEED_BALANCE} 
	
done
done