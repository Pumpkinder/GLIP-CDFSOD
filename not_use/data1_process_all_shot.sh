#!/bin/bash

SHOT=5
DIR=/root/autodl-tmp/GLIP
SHOT_ES=2
SHOT_ITER=(1 2 3 4)
the=0.4
################
ALL=10
REAP=$(($ALL / $SHOT))
#echo ${SHOT_1_ES}

sed -i "s/${SHOT}_shot_pseu.json/${SHOT}_shot.json/g" ${DIR}/configs/cdfsod/dataset1.yaml ##有需要则测试文件修改为训练文件用于打伪标签
sed -i "s/${SHOT}_shot.json/00tmpp/g" ${DIR}/configs/cdfsod/dataset1_train.yaml ##有需要则测试文件修改为训练文件用于打伪标签

for ITER in "${SHOT_ITER[@]}"
do
#echo ${ITER}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
      --config-file ${DIR}/configs/pretrain/glip_Swin_L.yaml  --ft-tasks ${DIR}/configs/cdfsod/dataset1.yaml --skip-test \
      --custom_shot_and_epoch_and_general_copy ${SHOT}_200_${REAP} \
      --evaluate_only_best_on_test --push_both_val_and_test \
      --e_stop ${SHOT_ES} --t_task_id ${ITER} --use_prepared_data\
      MODEL.WEIGHT /root/autodl-tmp/GLIP/weights/glip_large_model.pth \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.05 SOLVER.BASE_LR 0.00002 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full

MODEL_NAME=`cat ${DIR}/OUTPUT/ft_task_${ITER}/last_checkpoint`
#echo ${MODEL_NAME}
sed -i "s/00tmpp/${SHOT}_shot.json/g" ${DIR}/configs/cdfsod/dataset1_train.yaml ##有需要则测试文件修改为训练文件用于打伪标签
python tools/test_grounding_net.py --config-file /root/autodl-tmp/GLIP/configs/pretrain/glip_Swin_L.yaml --weight ${DIR}/OUTPUT/ft_task_${ITER}/${MODEL_NAME} \
--task_config ${DIR}/configs/cdfsod/dataset1_train.yaml \
TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
TEST.EVAL_TASK detection \
DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
DATASETS.USE_OVERRIDE_CATEGORY True \
DATASETS.USE_CAPTION_PROMPT True
sed -i "s/${SHOT}_shot.json/00tmpp/g" ${DIR}/configs/cdfsod/dataset1_train.yaml ##有需要则测试文件修改为训练文件用于打伪标签

python pseuod.py ${DIR}/OUTPUT/eval/${MODEL_NAME:0:-4}/inference/test/bbox.json ${DIR}/DATASET/dataset1/annotations/${SHOT}_shot.json ${DIR}/DATASET/dataset1/annotations/${SHOT}_shot_pseu.json ${the}

sed -i "s/${SHOT}_shot.json/${SHOT}_shot_pseu.json/g" ${DIR}/configs/cdfsod/dataset1.yaml ##有需要则测试文件修改为训练文件用于打伪标签

SHOT_ES=`expr $SHOT_ES + 1`
#SHOT_ES=3
the=0.5
#sleep 10

done

##########测试
echo ${ITER}
python tools/test_grounding_net.py --config-file /root/autodl-tmp/GLIP/configs/pretrain/glip_Swin_L.yaml --weight ${DIR}/OUTPUT/ft_task_${ITER}/${MODEL_NAME} \
--task_config ${DIR}/configs/cdfsod/dataset1.yaml \
TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
TEST.EVAL_TASK detection \
DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
DATASETS.USE_OVERRIDE_CATEGORY True \
DATASETS.USE_CAPTION_PROMPT True
