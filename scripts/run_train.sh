#!/bin/bash

cd /common-data/zhanghaojie/DropLora
bash /common-data/zhanghaojie/DropLora/scripts/run_cms.sh >logs/cms_llama3-8b.txt 2>&1 
bash /common-data/zhanghaojie/DropLora/scripts/run_metamath.sh >logs/math_llama3-8b.txt 2>&1
bash /common-data/zhanghaojie/DropLora/scripts/run_python.sh >logs/python_llama3-8b.txt 2>&1
bash /common-data/zhanghaojie/DropLora/scripts/run_conv.sh >logs/conv_llama3-8b.txt 2>&1

while true; do
    sleep 10
done