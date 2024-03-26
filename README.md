# code_for_master_paper
毕业论文实验三、四的代码，即多分类问题。
1. 文件说明
   
1.1 data文件夹：
	由于github无法上传大文件，则此处只有data空文件夹，对应的数据直接在FedProx提供的链接中下载。
 
 	如FEMNIST：https://drive.google.com/file/d/1tCEcJgRJ8NdRo11UJZR6WSKMNdmox4GC/view?usp=sharing
 
 	举例：如MNIST数据，其文件存储路径为：训练集：data/mnist/data/train/xxx    测试集：data/mnist/data/test/xxx

	存放了synthetic_iid, synthetic_0_0, synthetic_0.5_0.5, synthetic_1_1, mnist, femnist数据，均是FedProx文献处理好的数据。
  
1.2 FedDR、FedQuasiNewton（我们的方法）文件夹：

	分别是FedDR和FedQuasiNewton的实现代码，其中client.py是节点运行的相关代码，server.py是中心端服务器的运行的相关代码，synFedDR.py、FedQuasiNewton.py则是将节点和中心端服务器的功能串起来。\\
 	input.py是算法输入的具体参数。
  
1.3 regularizer文件夹:

	存放的是$\ell_1$正则项的相关计算，如其近端的显式解，近端的偏导数等；proximal_operator.py是FedQuasiNewton中具体求解变尺度近端的一些函数。

1.4 utils文件夹：

    generate_dataset.py，主要是为了将数据集的数据处理成代码可读取的样子;
    
    generate_Hessian.py则是FedQuasiNewton生成SR1矩阵的方法；
    
    model.py则是所用的网络，目前是单层网络，在此处修改代码可以搞成多层网络，但是目前synFedDR、FedQuasiNewton中的代码都是按单层网络设置的，所以也需要对这些模块进行修改（修改难度暂时不详）。
    
    second_grad.py是求真实Hessian矩阵的模块，目前在代码中没有使用；
    
    subroutines.py是半光滑牛顿法等一些求解FedQuasiNewton中变尺度近端计算涉及到的$L(\alpha)=0$的代码；
    
    swith_variable_type.py的功能是类型转换，主要是用来处理代码中涉及到的类型转换；
    
    deal_dataset.py用来整合成global数据集。目前没有用到。
  
1.5 main.py是主文件，在此处运行代码。包括参数初始化、结果画图、读取数据、算法参数设置（如所用的损失函数、迭代轮次、正则项等）。

     画图函数在该文件中，若在作图中要添加算法数量，要在该函数中对应的添加。

     在if __name__ == '__main__'中，各参数的含义：
	client_num：节点总个数；
	item：用以找到对应的网络维度，目前支持输入'synthetic','mnist','femnist'中的一个；
	dataset: 输入具体的数据集名字，目前支持 'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1', 'mnist', 'femnist'；
	max_iter：最大通信轮次
	reg_type：正则项类型，目前仅支持"l1"（即$\ell_1$正则项）
	lambd：正则项系数；
	active_ratio：每轮更新的节点比例；
 
————————————————————————————————————————————————————————————————————————————————————————————————————————————

     算法中的options主要是输入算法所需要的一些参数，包括损失函数等，具体地：
	'loss_func'：光滑损失函数项，多分类问题均为nn.CrossEntropyLoss()，即交叉熵损失；
	'rounds': max_iter，读取上面的最大迭代轮次；
	'local_epoch': 节点局部迭代轮次，如在FedDR中设置为20，即每个节点做20个epoch的随机梯度下降更新局部模型；在FedQuasiNewton中决定了ADMM的迭代轮次，设置为1，即单次ADMM。
	'regularizer': reg_type，读取上面的正则项类型
	'lambd': lambd，读取上面的正则项系数
	'data_item': item，含义同上面的item，读取上面的item。
	'active_ratio': active_ratio
 
	##### FedQuasiNewton 特有的 ############
 
	# 节点ADMM中的参数，alpha是用来保证生成Hessian矩阵正定的参数，即在Hessain矩阵的基础上加上alpha倍的单位阵。rho是ADMM的参数
	'alpha': 1, 
	'rho': 5,  
	# hessian矩阵相关参数，global_hessian默认为None，此时中心端会按算法的方式生成全局Hessian矩阵，若取'SR1'，则会直接在中心端生成一个SR1矩阵的近似全局Hessian（效果不佳）
	# nu_hat 和 eta 是无记忆SR1中的参数
	'global_hessian': None,
	'nu_hat': 10, # 
	'eta': 0.99,
	##### FedDR 特有的 #################
	'batch_size': 50，批量随机梯度下降的batch大小
	'learning_rate': 0.01, 学习率
	'eta': 500, FedDR中的参数
	'alpha': 1.95, 同上，FedDR中的参数
1.6 打开main.py文件运行代码，其中超参数的修改要在这个文件下修改。输出的图会存放在img文件夹中。一张图中包含训练集、测试集上的loss和accuracy的变化趋势。
			
			
		
