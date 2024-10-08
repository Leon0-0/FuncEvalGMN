from GMN.handle_dataset.spider_dataset import SpiderDataset
# from GMN.handle_dataset.data_pre_processor.ast_processor import ASTProcessor
from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
from GMN.evaluation import compute_similarity, auc
from GMN.loss import pairwise_loss
from utils import *
from GMN.configure import *
import numpy as np
import torch.nn as nn
import time
import os
import random
import pandas as pd

seed = 8
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# Initialize global variables
highest_pair_auc = 0
highest_tau = 0
flag = False

def model_eval_dev(dev_dataset : SpiderDataset, name):

    global highest_pair_auc, highest_tau, flag

    with (torch.no_grad()):

        similarity_array = torch.empty(0).to(device)
        labels_array = torch.empty(0).to(device)
        dev_dataset.shuffle()
        for edge_tuple, node_tuple, n_graphs, batch_labels in dev_dataset:
            labels = batch_labels.to(device)
            eval_pairs = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

            x, y = reshape_and_split_tensor(eval_pairs,  int(n_graphs / 2))
            similarity = compute_similarity(config, x, y).to(device)

            similarity_array = torch.cat((similarity_array, similarity), dim=0)
            labels_array = torch.cat((labels_array, labels), dim=0)

        pair_auc = auc(similarity_array, labels_array)

        from scipy import stats
        x = similarity_array.cpu().numpy()
        y = labels_array.cpu().numpy()
        # Kendall-Tau Correlation
        tau, p_value_tau = stats.kendalltau(x, y)
        # print(f"Kendall-Tau Correlation: τ = {tau}, p-value = {p_value_tau}")
        # Spearman Correlation
        rho, p_value_rho = stats.spearmanr(x, y)
        # print(f"Spearman Correlation: rₛ = {rho}, p-value = {p_value_rho}")
        pair_auc = round(pair_auc, 4)
        tau = round(tau, 4)
        rho = round(rho, 4)

        # Update global variables and flag
        if name == 'dev_dataset':
            if pair_auc > highest_pair_auc or tau > highest_tau:
                highest_pair_auc = max(pair_auc, highest_pair_auc)
                highest_tau = max(tau, highest_tau)
                flag = True
            else:
                flag = False

        log_file_txt = 'iter %d, loss %.4f, val/pair_auc %.6f, time %.2fs, τ %.4f, rₛ %.4f, %s\n' % (
            i_epoch, loss_mean.item(), pair_auc, time.time() - t_start, tau, rho, name)

        print('iter %d, loss %.4f, val/pair_auc %.4f, time %.2fs, %s' %
              (i_epoch, loss_mean.item(), pair_auc, time.time() - t_start, name))

        return log_file_txt

is_cuda = torch.cuda.is_available()
ONLINE_TRAIN_SETTINGS = {
    'NAME': "train_newlabel_remove_denotation",
    'GPU_DEVICE': torch.device('cuda:0' if is_cuda else 'cpu'),
    'PATH_TO_GMN': "./GMN",
    'IMAGES_FOLDER': "save_file/picture/",
    'LOG_FILE_NAME': "save_file/log/plot",
    'Checkpoint': 'save_file/checkpoints/',
    'Train': './GMN/database_spider/gmn_api_data/Spider_pair_train.xlsx',
    'Dev': './GMN/database_spider/gmn_api_data/Spider_pair_dev.xlsx',
}

print("==================================")
device = ONLINE_TRAIN_SETTINGS['GPU_DEVICE']
name = ONLINE_TRAIN_SETTINGS['NAME']
print("Device:", device)
print("Name:", name)

if is_cuda:
    image_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER'])
    log_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME'])
    checkpoint_folder_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Checkpoint'])

    train_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Train'])
    dev_path = os.path.join(ONLINE_TRAIN_SETTINGS['PATH_TO_GMN'], ONLINE_TRAIN_SETTINGS['Dev'])

    print("Images saved in:", image_folder_path)
    print("Logs saved in:", log_folder_path)
    print("==================================")

else:
    image_folder_path = ONLINE_TRAIN_SETTINGS['IMAGES_FOLDER']
    log_folder_path = ONLINE_TRAIN_SETTINGS['LOG_FILE_NAME']
    checkpoint_folder_path = ONLINE_TRAIN_SETTINGS['Checkpoint']

    train_path = ONLINE_TRAIN_SETTINGS['Train']
    dev_path = ONLINE_TRAIN_SETTINGS['Dev']

torch.set_default_tensor_type(torch.FloatTensor)
# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

train_df = pd.read_excel(train_path, sheet_name="Sheet1").reset_index(drop=True)
dev_df = pd.read_excel(dev_path, sheet_name="Sheet1").reset_index(drop=True)

data_pre_processor = PositionalEncodingProcessor() #ASTProcessor()
pair_list_train, labels_train = data_pre_processor.read_data(train_df)
train_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_train, labels_train)

pair_list_dev, labels_dev = data_pre_processor.read_data(dev_df)
dev_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_dev, labels_dev)

model, optimizer = build_model(config)
pretrain_model = torch.load('./GMN/models/pretrain_model0.1_maskN_0.1_dropN_50_1e5.pt')
model.load_state_dict(pretrain_model)
model.to(device)

training_dataset = SpiderDataset(train_batch_data)
dev_dataset = SpiderDataset(dev_batch_data)

log_txt = ""

for i_epoch in range(5001):
    model.train(True)
    t_start = time.time()

    loss_mean = torch.empty(0).to(device)
    training_dataset.shuffle()
    for edge_tuple, node_tuple, n_graphs, batch_labels in training_dataset:
        labels = batch_labels.to(device)
        graph_vectors = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

        x, y = reshape_and_split_tensor(graph_vectors,  int(n_graphs / 2))
        loss = pairwise_loss(x, y, labels, loss_type=config['training']['loss'], margin= config['training']['margin'])
        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        # add
        loss.backward(torch.ones_like(loss))
        nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
        optimizer.step()

    model.eval()

    dev_log = model_eval_dev(dev_dataset, "dev_dataset")

    log_txt = log_txt + dev_log

    if i_epoch % 2 == 0 and i_epoch != 0:
        log_file_path = f"{log_folder_path}_{name}.txt"
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_txt)
            log_file.flush()
            log_txt = ""
    if flag:
        checkpoint_save_path = f"{checkpoint_folder_path}{name}_{i_epoch}_{highest_pair_auc}_{highest_tau}.pt"
        torch.save(model.state_dict(), checkpoint_save_path)
