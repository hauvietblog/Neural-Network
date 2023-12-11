# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Neural Networks](https://machinelearningcoban.com/2017/02/24/mlp/)
Định nghĩa: Cho $L \in \mathbb{N}, d = (d^{(0)},d^{(l)},\dots,d^{(L)}) \in \mathbb{N}^{L+1}$ và $\sigma:\mathbb{R} \to \mathbb{R}.$ $\sigma$ là activation function, $L$ là số lượng layers, 
và $d^{(0)},d^{(L)},d^{(l)},l \in [L-1]$ lần lượt là số lượng neural đầu vào, đầu ra và layer ẩn thứ $l$. Khi đó $P(d)$ là số lượng tham số.
$$P(d)=\sum_{l=1}^{L} d^{(l)} d^{(l-1)}+ d^{(l)}$$
Xét ánh xạ:

$$\begin{aligned} 
\Phi_\alpha:\mathbb{R}^{d^{(0)}}\times\mathbb{R}^{P(d)} &\to \mathbb{R}^{d^{(L)}} \\\\ (\mathbf{x},\theta) &\mapsto \Phi_\alpha(\mathbf{x},\theta) 
\end{aligned}$$

Với mọi $\mathbf{x} \in \mathbb{R}^{d^{(0)}}$ và tham số $\theta = (\theta^{(l)})_{l=1}^L=((W^{(l)},b^{(l)}))\_{l=1}^L \in \bigtimes\limits\_{l=1}^{L}(\mathbb{R}^{d^{(l)}\times d^{(l-1)}}\times \mathbb{R}^{d\^{(l)}})\cong \mathbb{R}^{P(d)},$ đặt $\Phi\_{\alpha}(\mathbf{x},\theta)=\Phi^L(\mathbf{x},\theta)$ trong đó $\alpha=(d,\sigma)$ thì

$$\Phi^{(1)}:=\mathbf{Z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)},$$   

$$\bar\Phi^{(l)}:=\mathbf{a}^{(l)}=\sigma(\mathbf{Z}^{(l)}),~~l \in [L-1],$$

$$\Phi^{(l+1)}:=\mathbf{Z}^{(l+1)}=\mathbf{W}^{(l+1)}\mathbf{a}^{(l)}+\mathbf{b}^{(l+1)},~~l \in [L-1]$$

$$\bar\Phi^{(l+1)}:=\mathbf{a}^{(l+1)}=\sigma(\mathbf{Z}^{(l+1)}),~~l \in [L-1]$$

Chúng ta coi $\mathbf{W}^{(l)} \in \mathbb{R}^{d^{(l)} \times d^{(l-1)}}$ và $\mathbf{b}^{(l)} \in \mathbb{R}^{d^{(l)}}$ là ma trận trọng số và vector bias.
## [Backpropagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/?fbclid=IwAR2awLv1m6QkU7pDlpusUjOOrv4R61TSFLyllhuTPneuxUTpQhJmB3s3Is8)
