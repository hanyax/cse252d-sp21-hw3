import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', help='path to input images' )
parser.add_argument('--experiment', default='test_combined', help='the path to store sampled images and models' )
parser.add_argument('--modelRoot', default='checkpoint_spp', help='the path to store the testing results')
parser.add_argument('--epochId', type=int, default=50, help='the number of epochs being trained')
parser.add_argument('--batchSize', type=int, default=1, help='the size of a batch' )
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_false', help='whether to do spatial pyramid or not' )
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

# The detail network setting
opt = parser.parse_args()
print(opt)

colormap = io.loadmat(opt.colormap )['cmap']

assert(opt.batchSize == 1 )

# Save all the codes
os.system('mkdir %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initialize image batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300) )
labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, 300, 300) )
maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300) )
labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )


encoder_dil = model.encoderDilation()
decoder_dil = model.decoderDilation()

encoder_spp = model.encoderSPP()
decoder_spp = model.decoderSPP()

encoder = model.encoder()
decoder = model.decoder()

encoder.load_state_dict(torch.load('%s/encoderFinal_%d.pth' % ('checkpoint', opt.epochId)))
decoder.load_state_dict(torch.load('%s/decoderFinal_%d.pth' % ('checkpoint', opt.epochId)))
encoder_dil.load_state_dict(torch.load('%s/encoderFinal_%d.pth' % ('checkpoint_dil', opt.epochId)))
decoder_dil.load_state_dict(torch.load('%s/decoderFinal_%d.pth' % ('checkpoint_dil', opt.epochId)))
encoder_spp.load_state_dict(torch.load('%s/encoderFinal_%d.pth' % ('checkpoint_spp', opt.epochId)))
decoder_spp.load_state_dict(torch.load('%s/decoderFinal_%d.pth' % ('checkpoint_spp', opt.epochId)))

encoder = encoder.eval()
decoder = decoder.eval()
encoder_dil = encoder_dil.eval()
decoder_dil = decoder_dil.eval()
encoder_spp = encoder_spp.eval()
decoder_spp = decoder_spp.eval()

# Move network and containers to gpu
if not opt.noCuda:
    device = 'cuda'
else:
    device = 'cpu'

imBatch = imBatch.to(device)
labelBatch = labelBatch.to(device)
labelIndexBatch = labelIndexBatch.to(device)
maskBatch = maskBatch.to(device)

encoder = encoder.to(device)
decoder = decoder.to(device)
encoder_dil = encoder_dil.to(device)
decoder_dil = decoder_dil.to(device)
encoder_spp = encoder_spp.to(device)
decoder_spp = decoder_spp.to(device)


# Initialize dataLoader
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList
        )
segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=0, shuffle=True )

lossArr = []
iteration = 0
epoch = opt.epochId
confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
for i, dataBatch in enumerate(segLoader ):
    iteration += 1

    # Read data
    imBatch = Variable(dataBatch['im']).to(device)
    labelBatch = Variable(dataBatch['label']).to(device)
    labelIndexBatch = Variable(dataBatch['labelIndex']).to(device)
    maskBatch = Variable(dataBatch['mask']).to(device)

    # Test network
    x1, x2, x3, x4, x5 = encoder(imBatch )
    pred = decoder(imBatch, x1, x2, x3, x4, x5 )
    
    # Test network
    x1_dil, x2_dil, x3_dil, x4_dil, x5_dil = encoder_dil(imBatch)
    pred_dil = decoder_dil(imBatch, x1_dil, x2_dil, x3_dil, x4_dil, x5_dil)
    
    # Test network
    x1_spp, x2_spp, x3_spp, x4_spp, x5_spp = encoder_spp(imBatch)
    pred_spp = decoder_spp(imBatch, x1_spp, x2_spp, x3_spp, x4_spp, x5_spp)
    
    if iteration % 50 == 0:
        vutils.save_image( imBatch.data , '%s/images_%d.png' % (opt.experiment, iteration ), padding=0, normalize = True)
        utils.save_label(labelBatch.data, maskBatch.data, colormap, '%s/labelGt_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
        utils.save_label(-pred.data, maskBatch.data, colormap, '%s/labelPred_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
        utils.save_label(-pred_dil.data, maskBatch.data, colormap, '%s/labelPred_dil_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
        utils.save_label(-pred_spp.data, maskBatch.data, colormap, '%s/labelPred_spp_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )

testingLog.close()
# Save the accuracy
np.save('%s/accuracy_%d.npy' % (opt.experiment, opt.epochId), accuracy )
