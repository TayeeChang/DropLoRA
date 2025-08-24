#!/bin/bash

cd /common-data/zhanghaojie/DropLora
# bash /common-data/zhanghaojie/DropLora/scripts/llama3_8b/run_cms.sh >logs/cms_llama3-8b.txt 2>&1 

bash /common-data/zhanghaojie/DropLora/scripts/llama2_7b/rank/run_cms_r_64.sh >logs/run_cms_r_64.txt 2>&1

while true; do
    sleep 10
done