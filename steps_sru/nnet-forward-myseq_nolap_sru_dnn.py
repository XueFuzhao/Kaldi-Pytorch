#!/usr/bin/python3

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
import theano
import theano.tensor as T
import json
import sys
import numpy
import kaldiIO
from signal import signal, SIGPIPE, SIG_DFL
from lib.ops  import SpeechNet_Sru_Test as SpeechNet
import lib as mylib
import time
if __name__ == '__main__':
    model = sys.argv[1]
    priors = sys.argv[2]
    # the default time step is 20
    spliceSize = int(sys.argv[3])
    cfgFile = sys.argv[4]
    
    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')
    
    learning=[]
    with open(cfgFile,'r') as load_f:
        learning = json.load(load_f)

    

    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors

    arkIn = sys.stdin.buffer
    arkOut = sys.stdout.buffer
    encoding = sys.stdout.encoding
    signal (SIGPIPE, SIG_DFL)

    ## Load a feature matrix (utterance)
    uttId, featMat = kaldiIO.readUtterance(arkIn)
    row,col = featMat.shape
    inputs_variable = T.fmatrix(name="inputs variable")
    t_c0 = T.tensor3(name="init_c0_test",dtype=theano.config.floatX)
    
    ##load the model for testing
    predicts,c1 = SpeechNet(inputs_variable,t_c0,learning['layerNum'], 1,"haha_I_am_useless",learning['historyNum'],learning['singFeaDim'],learning['hiddenDim'],learning['modelOrder'],learning['dilDepth'],learning['targetDim'])
    mylib.load_params(model)
    predict_fn = theano.function(
    [inputs_variable,t_c0],
    [predicts],
    on_unused_input='warn')
    
    #init value for initial state/cell of SRU
    c0=numpy.zeros((learning['layerNum'],1,learning['hiddenDim']),dtype=theano.config.floatX)

    while uttId:
        start_time = time.time()
        logProbMat = numpy.log (predict_fn(featMat,c0) / p)
        logProbMat [logProbMat == -numpy.inf] = -100
        row = logProbMat.flatten().shape[0]//learning['targetDim']
        logProbMat = numpy.reshape(logProbMat,(row,learning['targetDim']))
        ## Write utterance
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        ## Load another feature matrix (utterance)
        uttId, featMat = kaldiIO.readUtterance(arkIn)


