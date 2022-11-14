import torch
from architectures import LinearHeadBENDR_from_scratch, BENDRClassification_from_scratch
from torchsummary import summary

samples_tlen = 40

x = torch.zeros(1, 20, 256*samples_tlen)
model = BENDRClassification_from_scratch(1, samples_len=samples_tlen * 256, n_chn=20,
                                        encoder_h=512,
                                        contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None,
                                        mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=False)
a=0
output = model(x)
b=0