CUDA_VISIBLE_DEVICES=0,1,2,3 python -m  debugpy --listen localhost:57001 --wait-for-client tools/test_grounding_net.py --config-file /root/autodl-tmp/GLIP/configs/pretrain/glip_Swin_L.yaml --weight /root/autodl-tmp/GLIP/OUTPUT/ft_task_1/model_0000002.pth \
      --task_config /root/autodl-tmp/GLIP/configs/cdfsod/dataset2.yaml \
      TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True