#!/bin/bash
# # HAVO
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
MODEL_TYPE='efficientnet' # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-5 
NUM_EPOCH=10 # 10
BATCH_SIZE=2 #( 10 12 14 16 18 20 ) best 6, best 18 for large scale 
STRETCH_RATIO=5 #( 0.5 1 2 5 10 26 ) # best 5
MODALITY_TYPE='oct_bscans' # 'rpet' | 'fundus' | 'oct_bscans'
ATTRIBUTE_TYPE=gender # race|gender|hispanic
EXPR=train_predictor_longitudinal
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
MODEL_TYPE=resnet18 
NEED_BALANCE=false
CONV_TYPE=Conv3d  
IMBALANCE_BETA=-1
SPLIT_RATIO=1 
for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( q=0; q<${#CONV_TYPE[@]}; q++ ));
do
for (( i=0; i<1; i++ ));
do
python train_dr_fair_3d.py \
		--data_dir ${PROJECT_DIR}/datasets/harvard/Harvard-DR/ \
		--result_dir ./results/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${MODEL_TYPE[$j]}_${CONV_TYPE[$q]}_3D_baseline \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 200 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--split_seed 5 \
		--imbalance_beta ${IMBALANCE_BETA} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE} \
		--split_ratio ${SPLIT_RATIO} \
		--exp_name dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_3D_baseline_Conv_${CONV_TYPE[$b]}_Model_${MODEL_TYPE[$j]}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE} \
		--need_balance ${NEED_BALANCE} \
		--conv_type ${CONV_TYPE[$q]} 
		
done
done
done