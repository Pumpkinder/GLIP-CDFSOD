#!/bin/bash

DIR=/root/autodl-tmp/GLIP

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
      --config-file ${DIR}/configs/pretrain/glip_Swin_L.yaml  --ft-tasks ${DIR}/configs/cdfsod/dataset3.yaml --skip-test \
      --custom_shot_and_epoch_and_general_copy 5_200_2 \
      --evaluate_only_best_on_test --push_both_val_and_test --use_prepared_data --e_stop_detail 40 --seed 0 --task_name ${DIR}/OUTPUT/eval/data3_5shot/ \
      MODEL.WEIGHT ${DIR}/weights/glip_large_model.pth\
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.25 SOLVER.BASE_LR 0.05 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE language_prompt_v2 

MODEL_NAME=`cat ${DIR}/OUTPUT/ft_task_1/last_checkpoint`

##########test

python tools/test_grounding_net.py --config-file ${DIR}/configs/pretrain/glip_Swin_L.yaml --weight ${DIR}/OUTPUT/ft_task_1/${MODEL_NAME} \
      --task_config ${DIR}/configs/cdfsod/dataset3.yaml \
      TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True \
      MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER True

#cp --path ${DIR}/OUTPUT/eval/${MODEL_NAME:0:-4}/inference/test/bbox.json ${DIR}/GLIP/predict_result/dataset3_1shot/dataset3_1shot.json

