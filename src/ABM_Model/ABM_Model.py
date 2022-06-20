import os
from typing import List

import numpy as np
import torch

from .encoder_decoder import Encoder_Decoder
from .utils import load_checkpoint_part_weight, load_dict, gen_sample_bidirection

path = 'src/ABM_Model/results/models/ABM_params195_lr0.015625_55.12927439532944.pkl'
dictionary = 'src/ABM_Model/dictionary.txt'
end = 1


def abm_model_predict(image):
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 113
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 1

    params['L2R'] = 1
    params['R2L'] = 0

    idx_decoder = params['L2R']  # TODO: make input variable
    if idx_decoder == 1:
        params['L2R'] = 1
        end = 1
    if idx_decoder == 2:
        params['R2L'] = 1
        end = 0

    # load model
    model = Encoder_Decoder(params)
    load_checkpoint_part_weight(model, os.path.join(os.getcwd(), path))
    # model  # .cuda()
    model.eval()

    # load dictionary
    worddicts = load_dict(os.path.join(os.getcwd(), dictionary))
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    xx_pad = np.array([image]).astype(np.float32) / 255.
    print(xx_pad.shape, params, 10, idx_decoder)
    xx_pad = torch.from_numpy(xx_pad[None, :, :, :])  # # .cuda() # (1,1,H,W)
    print(xx_pad.shape)
    # direction
    sample, score, attn_weights, next_alpha_sum = gen_sample_bidirection(model, xx_pad, params, False,
                                                                         k=10, maxlen=1000,
                                                                         idx_decoder=int(idx_decoder))
    score = score / np.array([len(s) for s in sample])
    ss = sample[score.argmin()]
    # write decoding results
    # fpp_sample.write(test_uid_list[test_count_idx])

    prd_strs: List[str] = []
    for vv in ss:
        if vv == end:  # <eos>   # 'L2R' 1
            break
        prd_strs.append(worddicts_r[vv])

    return " ".join(prd_strs)
