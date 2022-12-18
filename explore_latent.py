import torch
import yaml
import mne
from architectures import ConvEncoderBENDR_from_scratch
from datasets import charge_dataset, recInfoDataset
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import numpy as np

def flat_latent_representation(model, dataset):
    list_sorted_rec_names = []
    targets_sorted = []
    outputs_sorted = []
    first_outputs = []
    model.eval()

    for x, y, rec_name in dataset:
        output_ = model(x.unsqueeze(0))   # size [1, 512, 107]
        new_dim = torch.zeros(output_.shape[2])
        for chn in range(output_.shape[1]):
            id = torch.argmax(output_[0][chn,:])
            new_dim[id] += 1
        output = torch.mean(output_[0], dim=0) # size [107]
        output = torch.cat((output, new_dim)) # size [214]

        list_sorted_rec_names.append(rec_name)
        targets_sorted.append(y)
        outputs_sorted.append(output)
        first_outputs.append(output_)
    return outputs_sorted, targets_sorted, list_sorted_rec_names, first_outputs

def plot_latent_matrix_per_class(enc_output, all_records, class0_names, class1_names, k_reduce=(4,1)):
    pool = torch.nn.AvgPool2d(k_reduce)

    ids_0 = [i for i in range(len(all_records)) if (all_records[i] in class0_names)]
    ids_1 = [i for i in range(len(all_records)) if (all_records[i] in class1_names)]
    class0_outputs = enc_output[ids_0]
    class1_outputs = enc_output[ids_1]
    final_matrix0 = np.zeros((int(enc_output.shape[1]/k_reduce[0]), int(enc_output.shape[2]/k_reduce[1])))
    for sample in range(class0_outputs.shape[0]):
        outp0 = pool(torch.tensor(np.expand_dims(class0_outputs[sample],0)))
        final_matrix0 += outp0.squeeze().detach().numpy()
    # make plot
    fig, ax = plt.subplots(figsize=(8,10))
    # show image
    shw = ax.imshow(final_matrix0/class0_outputs.shape[0], cmap=plt.cm.Blues, aspect='auto')
    # make bar
    bar = plt.colorbar(shw)
    ax.set_title('Mean of latent representations of segments\nfrom class 0', fontsize=10)
    ax.set_xlabel('Embeddings')
    ax.set_ylabel('Reduced (average by kernel {}) features/channels'.format(k_reduce[0]))

    final_matrix1 = np.zeros((int(enc_output.shape[1]/k_reduce[0]), int(enc_output.shape[2]/k_reduce[1])))
    for sample in range(class1_outputs.shape[0]):
        outp1 = pool(torch.tensor(np.expand_dims(class1_outputs[sample],0)))
        final_matrix1 += outp1.squeeze().detach().numpy()
    # make plot
    fig, ax = plt.subplots(figsize=(8,10))
    # show image
    shw = ax.imshow(final_matrix1/class1_outputs.shape[0], cmap=plt.cm.Blues, aspect='auto')
    # make bar
    bar = plt.colorbar(shw)
    ax.set_title('Mean of latent representations of segments\nfrom class 1', fontsize=10)
    ax.set_ylabel('Reduced (average by kernel {}) features/channels'.format(k_reduce[0]))
    ax.set_xlabel('Embeddings')
    plt.show()
    return final_matrix0/class0_outputs.shape[0] , final_matrix1/class1_outputs.shape[0]

def plot_latent_matrix_per_record(enc_output, all_records, record_name):
    rec_ids = [i for i in range(len(all_records)) if all_records[i] == record_name]
    rec_outputs = enc_output[rec_ids]
    final_matrix = np.zeros((enc_output.shape[1], enc_output.shape[2]))
    for sample in range(rec_outputs.shape[0]):
        final_matrix += rec_outputs[sample]
    # make plot
    fig, ax = plt.subplots(figsize=(8,10))
    # show image
    shw = ax.imshow(final_matrix/rec_outputs.shape[0], cmap=plt.cm.Blues, aspect='auto')
    # make bar
    bar = plt.colorbar(shw)
    ax.set_title('Mean of latent representations of segments\nfrom record {}'.format(record_name), fontsize=10)
    plt.show()



if __name__ == '__main__':
    ####PARAMETERS####
    dataset_name = 'decomp_study_SA010'
    w_len = 40
    class_names = ['standard + symptoms', 'high + symptoms'] #['HC', 'SCZ']
    use_3d = True

    ######
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')

    flat_outputs = torch.load('./{}_{}_flat_output.pt'.format(dataset_name, w_len), map_location=device)
    rec_names = torch.load('./{}_{}_rec.pt'.format(dataset_name, w_len), map_location=device)
    targets = torch.load('./{}_{}_target.pt'.format(dataset_name, w_len), map_location=device)
    outputs = torch.load('./{}_{}_output.pt'.format(dataset_name, w_len), map_location=device)

    np_flat_outputs = []
    for e in flat_outputs:
        np_flat_outputs.append(e.detach().numpy())

    np_flat_outputs = np.array(np_flat_outputs)

    assert np_flat_outputs.shape[1] == 2 * np.ceil(w_len*256/96)
    print('Flat outputs shape: {}'.format(np_flat_outputs.shape))

    np_outputs = []
    for e in outputs:
        np_outputs.append(e.detach().numpy().squeeze())
    np_outputs = np.array(np_outputs)

    assert (np_outputs.shape[2] == np.ceil(w_len*256/96)) and np_outputs.shape[1] == 512
    assert (np_flat_outputs.shape[0] == np_outputs.shape[0])
    print('Non-flat outputs shape: {}'.format(np_outputs.shape))

    if dataset_name == 'h_scz_study':
        rec_names_0 = ['SA000_day9_',
                       'SA000_day3_',
                       'SA000_day1_', 'SA000_day4_', 'SA000_day7_', 'SA000_day10',
                       'SA000_day14',
                       'SA000_day11', 'SA000_day6_',
                       'SA000_day12',
                       'SA000_day8_', 'SA000_day5_', 'SA000_day13']
        rec_names_1 = ['SA000_day26',
                       'SA000_day21', 'SA000_day23',
                       'SA000_day28', 'SA000_day30',
                       'SA000_day25',
                       'SA000_day29', 'SA000_day22',
                       'SA000_day32', 'SA000_day33']

    elif dataset_name == 'decomp_study_SA047':
        rec_names_0 = ['SA047_day1_', 'SA047_day2_', 'SA047_day3_', 'SA047_day4_', 'SA047_day5_']
        rec_names_1 = ['SA047_day6_', 'SA047_day7_', 'SA047_day9_', 'SA047_day13']

    elif dataset_name == 'decomp_study_SA010':
        rec_names_0 = ['SA010_day6_', 'SA010_day7_', 'SA010_day9_', 'SA010_day11', 'SA010_day12', 'SA010_day13']
        rec_names_1 = ['SA010_day1_', 'SA010_day3_', 'SA010_day5_']

    else:
        assert 1 == 0
    assert len(list(set(rec_names))) == (len(rec_names_0) + len(rec_names_1))

    ids_0 = [i for i in range(len(rec_names)) if (rec_names[i] in rec_names_0)]
    ids_1 = [i for i in range(len(rec_names)) if (rec_names[i] in rec_names_1)]
    ids_0_per_record = []
    for rec in rec_names_0:
        ids_0_per_record.append([i for i in range(len(rec_names)) if rec_names[i] == rec])
    ids_1_per_record = []
    for rec in rec_names_1:
        ids_1_per_record.append([i for i in range(len(rec_names)) if rec_names[i] == rec])

    print('Lets compute the reduce embedding...')
    if use_3d:
        embedding = umap.UMAP(n_neighbors=5, metric='mahalanobis', n_components=3).fit_transform(np_flat_outputs)
    else:
        embedding = umap.UMAP(n_neighbors=5, metric='mahalanobis', n_components=2).fit_transform(np_flat_outputs)
    print('Ready. Reduce embedding computed.')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'chocolate', 'orchid', 'tomato', 'olive',
              'darkseagreen', 'pink', 'dimgrey']
    # Plots
    ax = plt.axes(projection='3d')
    for x in range(len(rec_names_0)):
        if (x+1) == len(rec_names_0):
            ax.scatter(embedding[ids_0_per_record[x], 0], embedding[ids_0_per_record[x], 1],
                       embedding[ids_0_per_record[x], 2], c='blue', label=class_names[0], alpha=0.5, s=20) #label=rec_names_0[x], alpha=0.5, s=20)
        else:
            ax.scatter(embedding[ids_0_per_record[x], 0], embedding[ids_0_per_record[x], 1],
                       embedding[ids_0_per_record[x], 2], c='blue', alpha=0.5, s=20)
    for x in range(len(rec_names_1)):
        if (x+1) == len(rec_names_1):
            ax.scatter(embedding[ids_1_per_record[x], 0], embedding[ids_1_per_record[x], 1],
                       embedding[ids_1_per_record[x], 2], c='red', label=class_names[1], alpha=0.5, s=20)
        else:
            ax.scatter(embedding[ids_1_per_record[x], 0], embedding[ids_1_per_record[x], 1],
                       embedding[ids_1_per_record[x], 2], c='red', alpha=0.5, s=20)
    ax.legend(loc='best')
    ax.set_title('UMAP latent space representation (of {}s segments) colored by target class'.format(w_len), fontsize=10)

    plt.show(block=False)
    ax2 = plt.axes(projection='3d')

    for x in range(len(rec_names_1)):
        if (x+1) == len(rec_names_1):
            ax2.scatter(embedding[ids_1_per_record[x], 0], embedding[ids_1_per_record[x], 1],
                       embedding[ids_1_per_record[x], 2], c='black', label=class_names[1], alpha=0.35, s=20)
        else:
            ax2.scatter(embedding[ids_1_per_record[x], 0], embedding[ids_1_per_record[x], 1],
                       embedding[ids_1_per_record[x], 2], c='black', alpha=0.35, s=20)
    for x in range(len(rec_names_0)):
        ax2.scatter(embedding[ids_0_per_record[x], 0], embedding[ids_0_per_record[x], 1],
                   embedding[ids_0_per_record[x], 2], c=colors[x], label=rec_names_0[x][6:], alpha=0.5, s=20)

    ax2.legend(fontsize=8)
    ax2.set_title(
        'UMAP latent space representation (of {}s segments)\n{} class colored and diferenciated & {} class all in black'.format(
            w_len, class_names[0], class_names[1]), fontsize=10)

    plt.show(block=False)
    ax3 = plt.axes(projection='3d')

    for x in range(len(rec_names_0)):
        if (x + 1) == len(rec_names_0):
            ax3.scatter(embedding[ids_0_per_record[x], 0], embedding[ids_0_per_record[x], 1],
                        embedding[ids_0_per_record[x], 2], c='black', label=class_names[0], alpha=0.35, s=20)
        else:
            ax3.scatter(embedding[ids_0_per_record[x], 0], embedding[ids_0_per_record[x], 1],
                        embedding[ids_0_per_record[x], 2], c='black', alpha=0.35, s=20)
    for x in range(len(rec_names_1)):
        ax3.scatter(embedding[ids_1_per_record[x], 0], embedding[ids_1_per_record[x], 1],
                    embedding[ids_1_per_record[x], 2], c=colors[x], label=rec_names_1[x][6:], alpha=0.5, s=20)

    ax3.legend(fontsize=8)
    ax3.set_title(
        'UMAP latent space representation (of {}s segments)\n{} class colored and diferenciated & {} class all in black'.format(
            w_len, class_names[1], class_names[0]), fontsize=10)
    plt.show(block=False)
    a = 0
