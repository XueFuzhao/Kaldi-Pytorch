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




from subprocess import Popen, PIPE, DEVNULL
import tempfile
import kaldiIO
import pickle
import numpy
import os
import shutil
import torch.utils.data as data
import torch

## Data generator class for Kaldi

class dataGenSequences(data.Dataset):

    def __init__(self, data, ali, exp, timeSteps=20, inputDim=195, left=0, right=0):
        self.data = data
        self.ali = ali
        self.exp = exp

        self.lable_list = [0]
        self.left = left
        self.right = right
        self.timeSteps = timeSteps

        ## Number of utterances loaded into RAM.
        ## Increase this for speed, if you have more memory.
        self.maxSplitDataSize = 100

        ## Parameters for initialize the iteration
        self.item_counter = 0
        self.timeSteps_Num = 0


        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
        #aliPdf = '/home/glen/alipdf.txt'
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)

        ## Determine the number of steps
        ## need to re calculate The last patch will be deleted


        self.numSteps = -(-self.numFeats // ( self.timeSteps))
      
        self.inputFeatDim = inputDim ## IMPORTANT: HARDCODED. Change if necessary.
        self.singleFeatDim = inputDim//(1+self.left+self.right)
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        #print out the configuration
        print ("NumFeats:%d"%(self.numFeats))
        print("NumSteps:%d" % (self.numSteps))
        print ("FeatsDim:%d"%(self.inputFeatDim))
        print ("TimeSteps:%d"%(self.timeSteps))

        
        self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        self.y = numpy.empty (0, dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)
        print("numUtterances:%d"%(self.numUtterances))
        print("numSplit:%d" % (self.numSplit))

        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        if os.path.isdir (data + 'split' + str(self.numSplit)):
            shutil.rmtree (data + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()

        ## Save split labels and delete label
        self.splitSaveLabels(labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.exp], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        FilledNumFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1

            if (len(line)-1)%self.timeSteps!=0:
                FilledNumFeats += (self.timeSteps -(len(line)-1)%self.timeSteps) 
            
            labels[line[0]] = numpy.array([int(i) for i in line[1:]], dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        return labels, numFeats+FilledNumFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + 'utt/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)


    ## Return split of data to work on
    ## There
    def getNextSplitData (self):
        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE, stderr=DEVNULL)

        p2 = Popen (['splice-feats','--print-args=false','--left-context='+str(self.left),'--right-context='+str(self.right),'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()	
        

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
        #print(labels)
        featList = []
        labelList = []
        while True:
            uid, featMat = kaldiIO.readUtterance (p2.stdout)

            if uid == None:
                self.lable_list = labelList
                return (numpy.vstack(featList), numpy.hstack(labelList))
            if uid in labels:
                row,col = featMat.shape
                fillNum = self.timeSteps - (row % self.timeSteps)
                fillRight = fillNum//2
                fillLeft = fillNum - fillRight
                featMat = numpy.concatenate([numpy.tile(featMat[0],(fillLeft,1)), featMat, numpy.tile(featMat[-1],(fillRight,1))])
                #print(featMat.shape)
                labels4uid = labels[uid]
                labels4uid = numpy.concatenate([numpy.tile(labels4uid[0],(fillLeft,)), labels4uid, numpy.tile(labels4uid[-1],(fillRight,))])
                featList.append (featMat)
                labelList.append (labels4uid)


    def __len__(self):
        return self.numSteps


    def __getitem__(self, item):

        while (self.item_counter >= self.timeSteps_Num):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True

                # return the last group of data, may repeated several times but not matter
                return (self.xMini,self.yMini)
                # break

            self.splitDataCounter += 1
            x, y = self.getNextSplitData()
            self.split_counter = 0

            self.batchPointer = len(self.x) - len(self.x) % self.timeSteps
            self.timeSteps_Num = self.batchPointer//self.timeSteps
            self.x = numpy.concatenate((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate((self.y[self.batchPointer:], y))
            self.item_counter = 0


            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False

        item = item % ((len(self.x) - len(self.x) % self.timeSteps)//self.timeSteps)

        self.xMini = self.x[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.yMini = self.y[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.item_counter += 1

        self.xMini = torch.from_numpy(self.xMini)
        self.yMini = self.yMini.astype(numpy.int16)
        self.yMini = torch.from_numpy(self.yMini)


        return (self.xMini,self.yMini)



