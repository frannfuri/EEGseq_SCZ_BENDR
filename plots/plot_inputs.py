import torch
import matplotlib.pyplot as plt
import numpy as np
all_inputs = torch.load('../all_inputs_h_scz_Alpha.pt')
all_labels = torch.load('../all_labels_h_scz_Alpha.pt')
all_inputs = all_inputs.detach().cpu().numpy()
all_labels = all_labels.detach().cpu().numpy()

# ALL DATA (Cz and Pz)
mean_chn1 = np.mean(all_inputs[:,9,:], axis=0)
std_chn1 = np.std(all_inputs[:,9,:], axis=0)
mean_chn2 = np.mean(all_inputs[:,14,:], axis=0)
std_chn2 = np.std(all_inputs[:,14,:], axis=0)
mean_scale = np.mean(all_inputs[:,19,:], axis=0)
std_scale = np.std(all_inputs[:,19,:], axis=0)

plt.figure(figsize=(15,2))
plt.fill_between(list(range(all_inputs.shape[2])), mean_chn1-std_chn1, mean_chn1+std_chn1, alpha=0.2, color='r')
plt.plot(list(range(all_inputs.shape[2])), mean_chn1, '-', linewidth=0.5, color='r', label='Cz')
plt.title('Mean of the Cz channel inputs to the network')
plt.figure(figsize=(15,2))
plt.fill_between(list(range(all_inputs.shape[2])), mean_chn2-std_chn2, mean_chn2+std_chn2, alpha=0.2, color='b')
plt.plot(list(range(all_inputs.shape[2])), mean_chn2, '-', linewidth=0.5, color='b', label='Pz')
plt.title('Mean of the Pz channel inputs to the network')
plt.figure(figsize=(15,2))
plt.fill_between(list(range(all_inputs.shape[2])), mean_scale-std_scale, mean_scale+std_scale, alpha=0.2, color='g')
plt.plot(list(range(all_inputs.shape[2])), mean_scale, '-', linewidth=0.5, color='g', label='scale chn')
plt.title('Mean of the scale inputs to the network')
seg_len = all_inputs.shape[2]

# LABEL DIFFERENCIATED
labels = ['low + sympts', 'high + sympts'] # [0, 1]
colors = ['darkorange', 'limegreen']
class_0_ids = [i for i in range(len(all_labels)) if (all_labels[i] == 0)]
class_1_ids = [i for i in range(len(all_labels)) if (all_labels[i] == 1)]
assert len(class_0_ids) + len(class_1_ids) == len(all_labels)

all_inputs_0 = all_inputs[class_0_ids]
all_inputs_1 = all_inputs[class_1_ids]
all_inputs_per_class = [all_inputs_0, all_inputs_1]

all_means_per_class = []
all_stds_per_class = []
for inp in all_inputs_per_class:
    mean_chn1 = np.mean(inp[:,9,:], axis=0)
    std_chn1 = np.std(inp[:,9,:], axis=0)
    mean_chn2 = np.mean(inp[:,14,:], axis=0)
    std_chn2 = np.std(inp[:,14,:], axis=0)
    mean_scale = np.mean(inp[:,19,:], axis=0)
    std_scale = np.std(inp[:,19,:], axis=0)
    all_means_per_class.append((mean_chn1, mean_chn2, mean_scale))
    all_stds_per_class.append((std_chn1, std_chn2, std_scale))
    assert seg_len == inp.shape[2]

plt.figure(figsize=(15,2))
for j in range(len(all_inputs_per_class)):
    plt.fill_between(list(range(seg_len)), all_means_per_class[j][0] - all_stds_per_class[j][0], all_means_per_class[j][0] + all_stds_per_class[j][0], alpha=0.2, color=colors[j])
    plt.plot(list(range(seg_len)), all_means_per_class[j][0], '-', linewidth=0.3, color=colors[j], label=labels[j])
    plt.title('Mean of the Cz channel inputs to the network')
    plt.legend(loc='best')
plt.figure(figsize=(15,2))
for j in range(len(all_inputs_per_class)):
    plt.fill_between(list(range(seg_len)), all_means_per_class[j][1] - all_stds_per_class[j][1], all_means_per_class[j][1] + all_stds_per_class[j][1], alpha=0.2, color=colors[j])
    plt.plot(list(range(seg_len)), all_means_per_class[j][1], '-', linewidth=0.3, color=colors[j], label=labels[j])
    plt.title('Mean of the Pz channel inputs to the network')
    plt.legend(loc='best')
plt.figure(figsize=(15,2))
for j in range(len(all_inputs_per_class)):
    plt.fill_between(list(range(seg_len)), all_means_per_class[j][2] - all_stds_per_class[j][2], all_means_per_class[j][2]+ all_stds_per_class[j][2], alpha=0.2, color=colors[j])
    plt.plot(list(range(seg_len)), all_means_per_class[j][2], '-', linewidth=0.3, color=colors[j], label=labels[j])
    plt.title('Mean of the scale inputs to the network')
    plt.legend(loc='best')

plt.show(block=False)
a = 0