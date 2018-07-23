# Code to evaluate forward pass time in Pytorch
# Sept 2017
# Eduardo Romera
#######################

import os
import numpy as np
import torch
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable

import Model as Net

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main(args):
    classes = args.classes
    p = args.p
    q = args.q
    if args.modelType == 2:
        model = Net.ESPNet_Encoder(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'encoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(
            q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        model.load_state_dict(torch.load(model_weight_file))
    elif args.modelType == 1:
        model = Net.ESPNet(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'decoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        model.load_state_dict(torch.load(model_weight_file))
    else:
        print('Model not supported')

    if (not args.cpu):
        model = model.cuda()#.half()    #HALF seems to be doing slower for some reason
    #model = torch.nn.DataParallel(model).cuda()

    model.eval()


    images = torch.randn(args.height, args.width, args.num_channels, args.batch_size)

    if (not args.cpu):
        images = images.cuda()#.half()

    time_train = []

    i=0

    while(True):
    #for step, (images, labels, filename, filenameGt) in enumerate(loader):

        start_time = time.time()

        inputs = Variable(images, volatile=True)
        outputs = model(inputs)

        #preds = outputs.cpu()
        if (not args.cpu):
            torch.cuda.synchronize()    #wait for cuda to finish (cuda is asynchronous!)

        if i!=0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (args.batch_size, fwt/args.batch_size, sum(time_train) / len(time_train) / args.batch_size))
        
        time.sleep(1)   #to avoid overheating the GPU too much
        i+=1

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--classes', default=2, type=int, help='number of classes')
    parser.add_argument('--weightsDir', default='../pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--modelType', type=int, default=1, help='1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. Supported only 3, 5, 8')

    main(parser.parse_args())
