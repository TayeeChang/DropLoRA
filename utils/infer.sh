#!/bin/bash


#     "output-cms3-rank/3e-4/v6-20250411-144157/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v7-20250411-181857/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v8-20250411-215412/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v9-20250412-012444/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v10-20250412-045545/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v11-20250412-082656/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v12-20250412-115810/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v13-20250412-152907/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v14-20250412-185950/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v15-20250412-223021/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v16-20250413-020128/checkpoint-3459-merged",
#     "output-cms3-rank/3e-4/v17-20250413-053206/checkpoint-3459-merged"

# python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py >infer.txt 2>&1
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v7-20250411-181857/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v8-20250411-215412/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v9-20250412-012444/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v10-20250412-045545/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v11-20250412-082656/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v12-20250412-115810/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v13-20250412-152907/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v14-20250412-185950/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v15-20250412-223021/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v16-20250413-020128/checkpoint-3459-merged
python /common-data/zhanghaojie/DropLora/utils/gen_vllm_gs.py --model output-cms3-rank/3e-4/v17-20250413-053206/checkpoint-3459-merged

while true; do
    sleep 10
done