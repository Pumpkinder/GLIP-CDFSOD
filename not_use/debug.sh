CUDA_VISIBLE_DEVICES=0,1,2,3 python -m  debugpy --listen localhost:57001 --wait-for-client tools/finetune.py \
      --config-file /root/autodl-tmp/GLIP/configs/pretrain/glip_Swin_L.yaml  --ft-tasks /root/autodl-tmp/GLIP/configs/cdfsod/dataset3.yaml --skip-test \
      --custom_shot_and_epoch_and_general_copy 1_200_10 \
      --evaluate_only_best_on_test --push_both_val_and_test --use_prepared_data\
      MODEL.WEIGHT /root/autodl-tmp/GLIP/weights/glip_large_model.pth\
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.25 SOLVER.BASE_LR 0.05 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE language_prompt_v2 