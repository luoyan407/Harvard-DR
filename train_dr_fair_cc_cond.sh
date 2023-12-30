#!/bin/bash

#SBATCH -n 8
#SBATCH -t 96:00:00
#SBATCH -p nvidia
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=cw3437@nyu.edu
#SBATCH --output=vit_%j.out

#Activating conda
module purge
module load all
source ~/.bashrc
cd /scratch/cw3437/Harvard-DR/
conda activate earth_py

PROJECT_DIR=/scratch/cw3437/Data/DR/
TASK=cls # md | tds | cls
LOSS_TYPE='cos' # mse | cos | kld | mae | gaussnll | bce 
LR=( 1e-5 5e-6 1e-6 )
NUM_EPOCH=20
NUM_LAYERS=1
BATCH_SIZE=18
MODALITY_TYPE='slo_fundus' # 'rpet' | 'fundus' | 'oct_bscans'
ATTRIBUTE_TYPE=race # race|gender|hispanic
MODEL_TYPE=( cond_vit ) # efficientnet | vit | cond_vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
for (( j=0; j<${#LR[@]}; j++ ));
do
	PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${LR[$j]}_${NUM_EPOCH}_${NUM_LAYERS}.csv
	for (( i=3; i<4; i++ ));
	do
	python train_dr_fair_cc_cond.py \
			--data_dir ${PROJECT_DIR} \
			--result_dir ./results/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_baseline/fullysup_${MODEL_TYPE}_${NUM_EPOCH}_NUMLayer_{$i}_${MODALITY_TYPE}_Task${TASK}_lr${LR[$j]}_bz${BATCH_SIZE} \
			--model_type ${MODEL_TYPE} \
			--number_layer $i \
			--image_size 224 \
			--loss_type ${LOSS_TYPE} \
			--lr ${LR[$j]} --weight-decay 0. --momentum 0.1 \
			--batch-size ${BATCH_SIZE} \
			--task ${TASK} \
			--epochs ${NUM_EPOCH} \
			--modality_types ${MODALITY_TYPE} \
			--perf_file ${PERF_FILE} \
			--attribute_type ${ATTRIBUTE_TYPE} 
	done
done