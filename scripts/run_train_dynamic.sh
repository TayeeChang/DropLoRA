#!/bin/bash

cd /common-data/zhanghaojie/DropLora
# bash /common-data/zhanghaojie/DropLora/scripts/llama3_8b/run_cms.sh >logs/cms_llama3-8b.txt 2>&1 

bash /common-data/zhanghaojie/DropLora/scripts/llama_dynamic/run_dynamic_8b.sh >logs/llama3-8b-dynamic.txt 2>&1
bash /common-data/zhanghaojie/DropLora/scripts/llama_dynamic/run_dynamic_7b.sh >logs/llama2-7b-dynamic.txt 2>&1

while true; do
    sleep 10
done