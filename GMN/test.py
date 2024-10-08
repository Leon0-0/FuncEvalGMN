from GMN.handle_dataset.spider_dataset import SpiderDataset
from GMN.handle_dataset.data_pre_processor.ast_processor import ASTProcessor
from GMN.handle_dataset.data_pre_processor.positional_encoding_processor import PositionalEncodingProcessor
from GMN.evaluation import compute_similarity, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from utils import *
from GMN.configure import *
import numpy as np
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

def model_eval_dev(dev_dataset : SpiderDataset, name):

    with (torch.no_grad()):
        # df_old = pd.read_excel('./GMN/database_spider/diff_queries/BIRD_gb_new.xlsx')
        similarity_array = torch.empty(0).to(device)
        labels_array = torch.empty(0).to(device)
        # dev_dataset.shuffle()
        for edge_tuple, node_tuple, n_graphs, batch_labels in dev_dataset:
            labels = batch_labels.to(device)
            eval_pairs = model(edge_tuple.to(device), node_tuple.to(device), n_graphs)

            x, y = reshape_and_split_tensor(eval_pairs,  int(n_graphs / 2))
            similarity = compute_similarity(config, x, y).to(device)

            similarity_array = torch.cat((similarity_array, similarity), dim=0)
            labels_array = torch.cat((labels_array, labels), dim=0)


        pair_auc = auc(similarity_array, labels_array)
        print("auc score:",  pair_auc)

        from scipy import stats
        x = similarity_array.cpu().numpy()
        y = labels_array.cpu().numpy()
        # x = 1 / (1 + np.exp(x))

        lower_bound = -2
        upper_bound = -0.1
        filtered_scores = [score for score in x if lower_bound <= score <= upper_bound]
        best_threshold = None
        best_tau = -float('inf')

        thresholds = []
        kendall_taus = []
        for threshold in filtered_scores:
        # 将分数转换为预测标签
            predicted_labels = np.where(x >= threshold, 1, 0)
        # 计算 Kendall’s Tau
            tau, _ = stats.kendalltau(predicted_labels, y)

            thresholds.append(threshold)
            kendall_taus.append(tau)
    
        # 更新最佳阈值
            if tau > best_tau:
                best_tau = tau
                best_threshold = threshold
                auc_score = roc_auc_score(y, predicted_labels)
                rho, p_value_rho = stats.spearmanr(predicted_labels, y)

        print(auc_score)
        print(f'最佳阈值: {best_threshold}')
        print(f'最高 Kendall’s Tau: {best_tau}')
        print("spearman:", rho)

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

dev_df = pd.read_excel(dev_path, sheet_name="Sheet1").reset_index(drop=True)
data_pre_processor = PositionalEncodingProcessor()
pair_list_dev, labels_dev = data_pre_processor.read_data(dev_df)
dev_batch_data = data_pre_processor.pairs_spider(config['training']['batch_size'], pair_list_dev, labels_dev)


model, optimizer = build_model(config)
model_path = torch.load('./GMN/save_file/checkpoints/train_graphcl_927_0.975_0.6689.pt')
model.load_state_dict(model_path)
model.to(device)

dev_dataset = SpiderDataset(dev_batch_data)

model.eval()

model_eval_dev(dev_dataset, "dev_dataset")