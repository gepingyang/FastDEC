from FastDEC import FastDEC
import numpy as np
import time
import Dataprocessing
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
DATA_File = "./datasets/Spiral.csv"
# DATA_File = "./datasets/banana-ball.csv"
# DATA_File = "./datasets/R15.csv"
# DATA_File = "./datasets/S2.csv"
# DATA_File = "./datasets/Flame.csv"
# DATA_File = "./datasets/MFFCCs.csv"
n_clusters = 3
k = 7

current = time.perf_counter()
data, label_true = Dataprocessing.get_data(DATA_File)
fastdec = FastDEC(k=k, n_clusters=n_clusters)
result = fastdec.fit(data)
now_ = time.perf_counter()
time_ = (now_ - current)
############################### fiure 1#####################
g = fastdec.g
core_idx = fastdec.core_idx
topK_idx_core = fastdec.topK_idx_core
cdh_ids = fastdec.cdh_ids
k_th = fastdec.k_th
core_density = fastdec.core_density
density = fastdec.density
label1 = np.full(fastdec.n, 1, np.int32)
Dataprocessing.show_data(data, label1)
Dataprocessing.show_density(data, density)
Dataprocessing.show_DC(data, core_idx,cdh_ids,density)
Dataprocessing.plot_sig1(core_density, g,  k_th, topK_idx_core )
Dataprocessing.show_result(data,  result, core_idx[topK_idx_core])
# # SD = fastdec.SD
# num_SD = len(SD)
# avg_SD = np.mean(SD)
#####################################################
ami = AMI(label_true, result)
ari = ARI(label_true, result)
nmi = NMI(label_true, result)
print("Adj. Rand Index Score(ARI)=", ari)
print("Adj. Mutual Info Score(AMI)=", ami)
print("Norm Mutual Info Score=", nmi)
