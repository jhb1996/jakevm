#!/bin/bash

wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh

bash Anaconda2-5.1.0-Linux-x86_64.sh

# Answer yes to all questions, say no to VSCode

echo -e "export PATH=/home/ubuntu/anaconda2/bin:$PATH\n" > ~/.bashrc

source ~/.bashrc

rm Anaconda2-5.1.0-Linux-x86_64.sh

conda create --name pa5 python=2.7
source activate pa5
conda install keras jupyter scikit-image tensorflow nose
