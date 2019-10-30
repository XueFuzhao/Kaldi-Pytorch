#!/bin/bash

##  Copyright (C) 2017 Huang Hengguan
##  hhuangaj [at] ust [dot] hk
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


set -e

nj=4
. cmd.sh
. path.sh

## Configurable directories
## Please change these according to the dataset you selected, these para are used for timit
train=data/train
dev=data/dev
test=data/test
lang=data/lang
# gmm depends on which ali you choose
gmm=exp/tri3
exp=exp/sru_1024_dnn_12layer
#lm=tgpr_5k


export DEVICE=cuda

## tune learning rate
## Train
for lr in 0.25 ; do
    python steps_sru/train_sru_nolap_12layer_1024_fbank.py $dev ${gmm}_ali_dev $train ${gmm}_ali $gmm ${exp}_$lr $lr
    ## Make graph
    [ -f ${gmm}/graph/HCLG.fst ] || utils/mkgraph.sh ${lang}_test_bg $gmm ${gmm}/graph
    ## Decode
    echo "tune acoustic scale"
    for ac in 0.1  ; do
    [ -f ${exp}_$lr/decode.done ] || steps_sru/decode_myseq_nolap_sru_dnn_fbank.sh --nj $nj --acwt $ac --scoring-opts "--min-lmwt 4 --max-lmwt 15"  \
        --add-deltas "true" --norm-vars "true" --splice-size "20" --splice-opts "--left-context=0 --right-context=4"  \
        $test $gmm/graph ${exp}_$lr ${exp}_$lr/decode_$ac
    done
done

    
