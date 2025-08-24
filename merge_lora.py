
from swift.llm import InferArguments, merge_lora

# best_model_checkpoint = 'output-python-8b/3e-4/v2-20250401-123521/checkpoint-2457'
# infer_args = InferArguments(ckpt_dir=best_model_checkpoint)
# merge_lora(infer_args, device_map='auto')
# print("END merge lora.")

for best_model_checkpoint in [
                #    "output-python-8b/3e-4/v0-20250331-181647/checkpoint-1638",
                #    "output-python-8b/3e-4/v0-20250331-181647/checkpoint-2457",
                #    "output-python-8b/3e-4/v1-20250401-011252/checkpoint-1638",
                #    "output-python-8b/3e-4/v2-20250401-123521/checkpoint-1638",
                #    "output-python-8b/3e-4/v2-20250401-123521/checkpoint-2457",
                #    "output-python-8b/3e-4/v3-20250401-194222/checkpoint-1638",
                #    "output-python-8b/3e-4/v3-20250401-194222/checkpoint-2457",
                #    "output-python-8b/3e-4/v4-20250402-024053/checkpoint-1638",
                #    "output-python-8b/3e-4/v4-20250402-024053/checkpoint-2457",
                #    "output-python-8b/3e-4/v5-20250402-093636/checkpoint-1638",
                #    "output-python-8b/3e-4/v5-20250402-093636/checkpoint-2457",
                #    "output-python-8b/3e-4/v6-20250402-163039/checkpoint-1638",
                #    "output-python-8b/3e-4/v6-20250402-163039/checkpoint-2457",
                #    "output-python-8b/3e-4/v7-20250402-232418/checkpoint-1638",
                #    "output-python-8b/3e-4/v7-20250402-232418/checkpoint-2457"
                    #  "output-python-8b/3e-4/v8-20250403-061816/checkpoint-1638",
                    #  "output-python-8b/3e-4/v8-20250403-061816/checkpoint-2457"
                    #  "output-cms3-8b-dynamic/3e-4/v0-20250401-162835/checkpoint-3459"
                      # "output-cms3-8b-dynamic/3e-4/v0-20250401-162835/checkpoint-2306"
                      # "output-cms3-8b-dynamic/3e-4/v0-20250401-162835/checkpoint-2306"
                      # "output-metamath-rank/3e-4/v0-20250403-172224/checkpoint-3086",
                      # "output-metamath-rank/3e-4/v2-20250404-095625/checkpoint-3086"
                        # "output-metamath-rank/3e-4/v4-20250404-225057/checkpoint-3086",
                        # "output-metamath-rank/3e-4/v5-20250405-051549/checkpoint-3086",
                        # "output-metamath-rank/3e-4/v6-20250405-114047/checkpoint-3086",
                        # "output-metamath-rank/3e-4/v7-20250405-180525/checkpoint-3086",
                      # "output-conversations-8b/3e-4/v0-20250403-131130/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v1-20250403-235933/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v2-20250404-173652/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v3-20250405-042114/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v4-20250405-150953/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v5-20250406-015800/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v6-20250406-124528/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v7-20250406-233252/checkpoint-3351"
                      # "output-conversations-8b/3e-4/v5-20250406-015800/checkpoint-2234",
                      # "output-conversations-8b/3e-4/v6-20250406-124528/checkpoint-2234",
                      # "output-conversations-8b/3e-4/v7-20250406-233252/checkpoint-2234",
                      # "output-conversations-8b/3e-4/v8-20250407-101924/checkpoint-2234",
                      # "output-conversations-8b/3e-4/v8-20250407-101924/checkpoint-3351",
                      # "output-conversations-8b/3e-4/v0-20250403-131130/checkpoint-2234"
                      "output-cms3-rank/3e-4/v0-20250410-172624/checkpoint-3459",
                      "output-cms3-rank/3e-4/v1-20250410-205905/checkpoint-3459",
                      "output-cms3-rank/3e-4/v2-20250411-003137/checkpoint-3459",
                      "output-cms3-rank/3e-4/v3-20250411-040342/checkpoint-3459",
                      "output-cms3-rank/3e-4/v4-20250411-073614/checkpoint-3459",
                      "output-cms3-rank/3e-4/v5-20250411-110907/checkpoint-3459",
                      "output-cms3-rank/3e-4/v6-20250411-144157/checkpoint-3459",
                      "output-cms3-rank/3e-4/v7-20250411-181857/checkpoint-3459",
                      "output-cms3-rank/3e-4/v8-20250411-215412/checkpoint-3459",
                      "output-cms3-rank/3e-4/v9-20250412-012444/checkpoint-3459",
                      "output-cms3-rank/3e-4/v10-20250412-045545/checkpoint-3459",
                      "output-cms3-rank/3e-4/v11-20250412-082656/checkpoint-3459",
                      "output-cms3-rank/3e-4/v12-20250412-115810/checkpoint-3459",
                      "output-cms3-rank/3e-4/v13-20250412-152907/checkpoint-3459",
                      "output-cms3-rank/3e-4/v14-20250412-185950/checkpoint-3459",
                      "output-cms3-rank/3e-4/v15-20250412-223021/checkpoint-3459",
                      "output-cms3-rank/3e-4/v16-20250413-020128/checkpoint-3459",
                      "output-cms3-rank/3e-4/v17-20250413-053206/checkpoint-3459"
                   ]:
    print("best_model_checkpoint=", best_model_checkpoint)
    infer_args = InferArguments(ckpt_dir=best_model_checkpoint)
    merge_lora(infer_args, device_map='auto')
    print("END merge lora.")