# Limited-Linear-Combination
mixup是张宏毅提出的数据增强方法。但是没有给出解释。我基于多任务网络和物理学势能场理论解释了mixup，并
对mixup方法做出改进，提出了limited linear combination方法。limited linear combination不仅包含mixup的线
性内插，也包含线性外推，同时线性外推的系数受到限制，所以叫limited linear combination。在实验中线性外推
的系数为0.04时，可以到达mixup的baseline95.8%，系数为0.02时，准确率为96.3%，比mixup提高0.5%。
在这次项目中，我对于mixup的最初解释是mixup基于添加噪声，对mixup做出改进，但效果不理想。深入研究
mixup在交叉熵函数中的作用，才得出多任务网路和物理学势能场两种解释。
