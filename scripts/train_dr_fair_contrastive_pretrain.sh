#!/bin/bash
# # HAVO
# PROJECT_DIR=/shared/ssd_16T/yl535/project/python/
PROJECT_DIR=/data/home/luoy/project/python/
TASK=cls # md | tds | cls
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=5e-5 # 5e-5 for rnflt | 1e-4 for ilm
NUM_EPOCH=20 # 10
# BATCH_SIZE=16 # 16 is best
BATCH_SIZE=18 #( 10 12 14 16 18 20 ) best 6, best 18 for large scale 
STRETCH_RATIO=5 #( 0.5 1 2 5 10 26 ) # best 5
MODALITY_TYPE='fundus' # 'rpet' | 'fundus' | 'oct_bscans'
ATTRIBUTE_TYPE=( race gender hispanic ) # race|gender|hispanic
PROGRESSION_TYPE=progression_outcome_md_fast_no_p_cut # ( progression_outcome_td_pointwise_no_p_cut progression_outcome_md_fast_no_p_cut )
EXPR=train_predictor_longitudinal
PROGRESSION_TYPE=( 'progression.outcome.md' 'progression.outcome.vfi' 'progression.outcome.td.pointwise' 'progression.outcome.md.fast' 'progression.outcome.md.fast.no.p.cut' 'progression.outcome.td.pointwise.no.p.cut' )
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
MODEL_TYPE=vit # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
NEED_BALANCE=false
CONT_TYPE=FSCL # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
IMBALANCE_BETA=0.9999
IMBALANCE_BETA=-1
SPLIT_RATIO=1 # ( 0.2 0.3 .4 .5 .6 .8 .9 1 )
for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
do
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}.csv
python train_dr_fair_contrast_pretrain.py \
		--data_dir ${PROJECT_DIR}/datasets/harvard/Harvard-DR/ \
		--result_dir ./results/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}_baseline/fullysup_${MODEL_TYPE}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE}_beta${IMBALANCE_BETA}_split${SPLIT_RATIO}_balance${NEED_BALANCE} \
		--model_type ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--task ${TASK} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--split_seed 5 \
		--imbalance_beta ${IMBALANCE_BETA} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE[$a]} \
		--split_ratio ${SPLIT_RATIO} \
		--need_balance ${NEED_BALANCE} \
		--cont_method ${CONT_TYPE} \
		# --seed 13 \ 
done