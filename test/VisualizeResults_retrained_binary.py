import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
import Model as Net
import os
import time
from argparse import ArgumentParser

pallete = [128, 64, 128,
           0, 0, 0]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    
    img[img == 0] = 7
    img[img == 1] = 0
    return img


def evaluateModel(args, model, up, image_list):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName).astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        # resize the image to 1024x512x3
        img = cv2.resize(img, (1024, 512))

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        img_variable = Variable(img_tensor, volatile=True)
        if args.gpu:
            img_variable = img_variable.cuda()
        img_out = model(img_variable)

        img_out = up(img_out)

        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()

        if i % 100 == 0:
            print(i)

        name = imgName.split('/')[-1]

        if args.colored:
            classMap_numpy_color = PILImage.fromarray(classMap_numpy)
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'))

        if args.cityFormat:
            classMap_numpy = relabel(classMap_numpy.astype(np.uint8))

        cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)


def main(args):
    # read all the images in the folder
    image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)

    up = None
    if args.modelType == 2:
        up = torch.nn.Upsample(scale_factor=16, mode='bilinear')
        if args.gpu:
            up = up.cuda()
    else:
        up = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        if args.gpu:
            up = up.cuda() 

    p = args.p
    q = args.q
    classes = args.classes
    if args.modelType == 2:
        modelA = Net.ESPNet_Encoder(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'encoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(
            q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    elif args.modelType == 1:
        modelA = Net.ESPNet(classes, p, q)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        model_weight_file = args.weightsDir + os.sep + 'decoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print('Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    else:
        print('Model not supported')
    # modelA = torch.nn.DataParallel(modelA)
    if args.gpu:
        modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    evaluateModel(args, modelA, up, image_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=1, help='1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--savedir', default='./results', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=False,
                        help='True if ESPNet. False for ESPNet-C')  # False for encoder
    parser.add_argument('--weightsDir', default='../pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. Supported only 3, 5, 8')
    parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--classes', default=20, type=int, help='number of classes')

    main(parser.parse_args())