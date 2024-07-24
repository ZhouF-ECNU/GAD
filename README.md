# GAD

Implementation of ["GAD: A Generalized Framework for Anomaly Detection at Different Risk Levels"]. (Accepted by CIKM 2024)

## Paper abstract

Anomaly detection is a crucial data mining problem due to its extensive range of applications. In real-world scenarios, anomalies often exhibit different levels of priority. Unfortunately, existing methods tend to overlook this phenomenon and identify all types of anomalies into a single class. In this paper, we propose a generalized formulation of the anomaly detection problem, which covers not only the conventional anomaly detection task, but also the partial anomaly detection task that is focused on identifying target anomalies of primary interest while intentionally disregarding non-target (low-risk) anomalies. One of the challenges in addressing this problem is the overlap among normal instances and anomalies of different levels of priority, which may cause high false positive rates. Additionally, acquiring a sufficient quantity of all types of labeled non-target anomalies is not always feasible. For this purpose, we present a generalized anomaly detection framework flexible in addressing a broader range of anomaly detection scenarios. Employing a dual-center mechanism to handle relationships among normal instances, non-target anomalies, and target anomalies, the proposed framework significantly reduces the number of false positives caused by class overlap and tackles the challenge of limited amount of labeled data. Extensive experiments conducted on two publicly available datasets from different domains demonstrate the effectiveness, robustness and superior labeled data utilization of the proposed framework. When applied to a real-world application, it exhibits a lift of at least 7.08% in AUPRC compared to the alternatives, showcasing its remarkable practicality.

## Full paper source:

Pending update...

## Running environment

Python version 3.7.16.

Create suitable conda environment:

```
conda env create -f environment.yml
```

## Fashion MNIST

For the fashionMNIST dataset, our training data contains three categories: normal, non_target, and target, which need to be explicitly specified in the sh command.

**$\text{GAD}^{f-partial}$**

```sh
python main.py --model_type GADF --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0&
```



**$\text{GAD}^{s-partial}$**

$\text{GAD}^{s-partial}$ is similar to $\text{GAD}^{f-partial}$. Although the command includes ``--non_target_outlier_class``, the number of non-target anomalies is set to 0 during runtime. The required sampling count is set to default 100 and does not need to be explicitly declared.

```sh
python main.py --model_type GADS --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0 &
```



**$\text{GAD}^{con}$**

The code for $\text{GAD}^{con}$ is identical to that of $\text{GAD}^{s-partial}$, but it utilizes data from conventional anomaly detection tasks.

```sh
python main.py --model_type GADS --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 6 --gpu 2 --random_seed 0 &
```

## UNSW_NB15

**$\text{GAD}^{f-partial}$**

```sh
python main.py --model_type GADF --dir_path ./result/nb15 --dataset_name nb15 --gpu 2 --random_seed 0&
```



**$\text{GAD}^{s-partial}$**

$\text{GAD}^{s-partial}$ adds the parameter ``--sample_count`` compared to $\text{GAD}^{f-partial}$. Since the default sampling count is 100, which is specific to the fashionMNIST dataset, it's necessary to declare ``--sample_count 1000``

```sh
python main.py --model_type GADS --dir_path ./result/nb15 --dataset_name nb15 --gpu 0 --sample_count 1000 --random_seed 0 &
```

## Citation

Wei R., He Z., Pavlovski M., Zhou F.,"GAD: A Generalized Framework for Anomaly Detection at Different Risk Levels"., Proceedings of the 33rd ACM International Conference on Information and Knowledge Management(CIKM) , 2024
