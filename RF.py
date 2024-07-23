import re
import os
import argparse
import click
import numpy as np
from sklearn.ensemble import RandomForestClassifier #集成学习中的随机森林
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score,confusion_matrix, classification_report, f1_score
import seaborn as sns
from collections import Counter
import random

args = argparse.ArgumentParser()
args.add_argument('--dir_path', type=str, default="./kdd_results/RF_20240202")
args.add_argument('--model_type', type=click.Choice(["BiasedAD", "BiasedADM", "Gcon"]), default="BiasedAD")
args.add_argument('--dataset_name', type=str, default="nb15")
# The follow three options are useful when the dataset is the fashionMNIST dataset
args.add_argument("--contamination_for_FMNIST", type=float, default=0.02)
args.add_argument("--labeled_target_outlier_number", type=int, default=100)
args.add_argument("--normal_class" , type=int, default = 4)
args.add_argument("--non_target_outlier_class" , type=int, default = 0)
args.add_argument("--target_outlier_class" , type=int, default = 6)

# The follow three options are useful when the dataset is the nb15 dataset with a fixed contamination ratio of 2%.
args.add_argument("--s_normal" , type=float, default = 1)
args.add_argument("--s_non_target" , type=int, default = 100)
args.add_argument("--s_target" , type=int, default = 100)
# Controls the number of non-target categories
args.add_argument("--nb15_non_target_class_num" , type=int, default = 4)
# Controls target categories
args.add_argument("--nb15_target_class", nargs="+", type=str, default=["DoS", "Generic", "Backdoor"], choices=["DoS", "Generic", "Backdoor"])

args.add_argument("--sqb_test_frac" , type=int, default = None)
args = args.parse_args()

contaminationRate = 2

binary_flag = args.model_type == 'BiasedADM'

if args.dataset_name == "nb15":
    if args.model_type == 'BiasedAD':
        file_name = f'processed_data/BiasedAD/s_normal={args.s_normal},s_non_target={args.s_non_target},s_target={args.s_target},nb15_non_target_class_num={args.nb15_non_target_class_num},nb15_target_class=DoS_Generic_Backdoor,contaminationRate={contaminationRate},eta0=20,model_type=BiasedAD,sample_count=100.npz'
    if args.model_type == 'BiasedADM':
        file_name = f'processed_data/BiasedADM/s_normal={args.s_normal},s_non_target={args.s_non_target},s_target={args.s_target},nb15_non_target_class_num={args.nb15_non_target_class_num},nb15_target_class=DoS_Generic_Backdoor,contaminationRate={contaminationRate},eta0=10,model_type=BiasedADM,sample_count=1000.npz'
    
elif args.dataset_name == "fashionmnist":
    if args.model_type == 'BiasedAD':
        file_name = f'intermediate_results/BiasedAD/normal={args.normal_class},non_target={args.non_target_outlier_class},target={args.target_outlier_class},s_normal={args.s_normal},labeled_target_outlier_number={args.labeled_target_outlier_number},contaminationRate=2,eta0=1,model_type=BiasedAD,sample_count=100.npz'
    if args.model_type == 'BiasedADM':
        file_name = f'intermediate_results/BiasedADM/normal={args.normal_class},non_target={args.non_target_outlier_class},target={args.target_outlier_class},s_normal={args.s_normal},labeled_target_outlier_number={args.labeled_target_outlier_number},contaminationRate={args.contamination_for_FMNIST},eta0=1,model_type=BiasedADM,sample_count=100.npz'

elif args.dataset_name == "SQB":
    if args.model_type == 'BiasedAD':
        file_name = f'processed_data/BiasedAD/dataset=sqb,sqb_test_frac=None,contaminationRate=2,eta0=1,model_type=BiasedAD,sample_count=100.npz'
    if args.model_type == 'BiasedADM':
        file_name = f'processed_data/BiasedADM/dataset=sqb,sqb_test_frac=None,contaminationRate=2,eta0=10,model_type=BiasedADM,sample_count=200.npz'
data = np.load(file_name,allow_pickle=True)
output_name = os.path.basename(file_name)
output_name = re.sub("\.npz|eta0=[0-9]+?,|,sample_count=[0-9]+","",output_name)

x_train = data["x_train"]
y_train = data["y_train"]
target_y_train = data["target_y_train"]
x_test = data["x_test"]
y_test = data["y_test"]
target_y_test = data["target_y_test"]

if binary_flag == False:

    random.seed(0)
    non_target_samples = x_train[np.where(target_y_train==-2)[0]]
    target_samples = x_train[np.where(target_y_train==-1)[0]]
    normal_samples = x_train[random.sample(np.where(target_y_train==0)[0].tolist(), len(non_target_samples))]

    # 三类
    rf_x_train = np.concatenate([normal_samples, non_target_samples, target_samples], axis=0)
    rf_y_train = np.concatenate([np.zeros(normal_samples.shape[0]),
                                np.zeros(non_target_samples.shape[0]) - 2,
                                np.zeros(target_samples.shape[0]) - 1,], axis=0)#建立模型
    for i in range(10):
        rfc = RandomForestClassifier()
        rfc = rfc.fit(rf_x_train, rf_y_train)
        y_pred = rfc.predict(x_test)
        print(Counter(y_pred))
        print(Counter(target_y_test))
        y_prob = rfc.predict_proba(x_test)

        cm = confusion_matrix(target_y_test, y_pred)
        print(cm)
        print(classification_report(target_y_test, y_pred))

        precision, recall, threshold = precision_recall_curve(y_test, y_prob[:,1])
        rf_test_AUPRC = auc(recall, precision)
        rf_test_AUROC = roc_auc_score(y_test, y_prob[:,1])
        f1 = np.nanmax(2 * recall * precision / (recall + precision))
        print(rf_test_AUPRC)
        print(rf_test_AUROC)
        with open(f'{args.dir_path}/{output_name}.txt', 'a+') as f:
            f.write('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * rf_test_AUROC, 100. * rf_test_AUPRC, f1 * 100))
            f.write('\n')
        print()

elif binary_flag:
    random.seed(0)
    target_samples = x_train[np.where(target_y_train==-1)[0]]
    normal_samples = x_train[random.sample(np.where(target_y_train==0)[0].tolist(), len(target_samples))]
    rf_x_train = np.concatenate([normal_samples, target_samples], axis=0)
    rf_y_train = np.concatenate([np.zeros(normal_samples.shape[0]),
                                    np.zeros(target_samples.shape[0]) + 1,], axis=0)#建立模型

    for i in range(10):
        rfc = RandomForestClassifier()
        rfc = rfc.fit(rf_x_train, rf_y_train)
        y_pred = rfc.predict(x_test)
        print(Counter(y_pred))
        print(Counter(y_test))
        y_prob = rfc.predict_proba(x_test)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(classification_report(y_test, y_pred))

        precision, recall, threshold = precision_recall_curve(y_test, y_prob[:,1])
        rf_test_AUPRC = auc(recall, precision)
        rf_test_AUROC = roc_auc_score(y_test, y_prob[:,1])
        f1 = np.nanmax(2 * recall * precision / (recall + precision))
        print(rf_test_AUPRC)
        print(rf_test_AUROC)
        with open(f'{args.dir_path}/{output_name}.txt', 'a+') as f:
            f.write('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * rf_test_AUROC, 100. * rf_test_AUPRC, f1 * 100))
            f.write('\n')
        print()    

    np.nanmax(2 * recall * precision / (recall + precision))