import time

import torch

from retinaface.data import cfg_mnet
from retinaface.models.retinaface import RetinaFace

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    net = checkpoint['net']

    filename = 'retinaface.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(net.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    net = RetinaFace(cfg=cfg_mnet)
    net.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    # filename_scripted = 'retinaface_scripted.pt'
    # print('saving {}...'.format(filename_scripted))
    # start = time.time()
    # torch.jit.save(torch.jit.script(net), filename_scripted)
    # print('elapsed {} sec'.format(time.time() - start))
    #
    # print('loading {}...'.format(filename))
    # start = time.time()
    # net = torch.jit.load(filename_scripted)
    # print('elapsed {} sec'.format(time.time() - start))
