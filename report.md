
在运行以下代码时：
grad = torch.autograd.grad(loss_fast, fast_parameters, create_graph=True)
报错：
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.

https://discuss.pytorch.org/t/runtimeerror-one-of-the-differentiated-tensors-appears-to-not-have-been-used-in-the-graph-set-allow-unused-true-if-this-is-the-desired-behavior/43679

原因：直接把用以预测学习率的lstm网络附在self上（即self.lstm）会导致self.parameters会导出lstm的参数，这一部分在一开始不参与梯度的计算
一开始被一些信息误导了，网上提供了一些代码可以打印出梯度是None的参数对应的层，但lstm在self的前面，这导致grad中的元素不能与classifier与embedder一一对应，lstm会顶掉一部分参数，导致最后显示梯度为None的层必然是classifier的最后一层或几层。但这实际上并不是问题的根源所在。