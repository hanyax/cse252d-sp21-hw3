import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io

parser = argparse.ArgumentParser()
# Model Param
parser.add_argument('--imageRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_false', help='whether to do spatial pyramid or not' )
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--imHeight', type=int, default=320, help='height of input image')
parser.add_argument('--imWidth', type=int, default=480, help='width of input image')

# Training Param
parser.add_argument('--experiment', default='checkpoint_dil', help='the path to store sampled images and models')
parser.add_argument('--batchSize', type=int, default=8, help='the size of a batch')
parser.add_argument('--nepoch', type=int, default=50, help='the training epoch')
parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
parser.add_argument('--iterationEnd', type=int, default=9150, help='the iteration to end training')

# The detail network setting
opt = parser.parse_args()
print(opt)

# Save all the codes
os.system('mkdir %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
'''
# Initialize network
if opt.isDilation:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
elif opt.isSpp:
    encoder = model.encoderSPP()
    decoder = model.decoderSPP()
else:
'''
encoder = model.encoderDilation()
decoder = model.decoderDilation()

if not opt.noCuda:
    device = 'cuda'
else:
    device = 'cpu'

model.loadPretrainedWeight(encoder)
encoder = encoder.to(device)
decoder = decoder.to(device)

# Initialize optimizer
optimizer = optim.Adam([{'params': encoder.parameters()},
                        {'params': decoder.parameters()}], lr = 0.0001)

# Initialize dataLoader
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList,
        imWidth = opt.imWidth, 
        imHeight = opt.imHeight
        )

segLoader = DataLoader(segDataset, batch_size = opt.batchSize, num_workers=8, shuffle=False)

lossArr = []
accuracyArr = []
iteration = 0
confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
for epoch in range(0, opt.nepoch):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(segLoader):
        iteration += 1

        # Read data
        imBatch = Variable(dataBatch['im']).to(device)
        labelBatch = Variable(dataBatch['label']).to(device)
        labelIndexBatch = Variable(dataBatch['labelIndex']).to(device)
        maskBatch = Variable(dataBatch['mask']).to(device)

        # Test network
        x1, x2, x3, x4, x5 = encoder(imBatch)
        pred = decoder(imBatch, x1, x2, x3, x4, x5)

        # Train network
        optimizer.zero_grad()
        loss = torch.mean( pred * labelBatch )
        hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch)
        confcounts += hist

        loss.backward()
        optimizer.step()

        for n in range(0, opt.numClasses):
            rowSum = np.sum(confcounts[n, :] )
            colSum = np.sum(confcounts[:, n] )
            interSum = confcounts[n, n]
            accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)

        # Output the log information
        lossArr.append(loss.cpu().data.item())
        meanLoss = np.mean(np.array(lossArr[:]))
        meanAccuracy = np.mean(accuracy)

        print('Epoch %d iteration %d: Loss %.5f Mean Loss %.5f' % ( epoch, iteration, lossArr[-1], meanLoss ) )
        print('Epoch %d iteration %d: Mean Accuracy %.5f' % ( epoch, iteration, meanAccuracy ) )
        trainingLog.write('Epoch %d iteration %d: Loss %.5f Mean Loss %.5f \n' % ( epoch, iteration, lossArr[-1], meanLoss ) )
        trainingLog.write('Epoch %d iteration %d: Mean Accuracy %.5f \n' % ( epoch, iteration, meanAccuracy ) )

        if iteration == opt.iterationEnd:
            np.save('%s/loss.npy' % opt.experiment, np.array(lossArr))
            np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracy) )
            torch.save(encoder.state_dict(), '%s/encoderFinal_%d.pth' % (opt.experiment, epoch+1))
            torch.save(decoder.state_dict(), '%s/decoderFinal_%d.pth' % (opt.experiment, epoch+1))
            break

    trainingLog.close()
    if iteration >= opt.iterationEnd:
        break
    if (epoch+1) % 2 == 0:
        np.save('%s/loss.npy' % opt.experiment, np.array(lossArr))
        np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracy) )
        torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1) )
        torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1) )