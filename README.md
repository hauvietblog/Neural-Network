# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Neural Networks](https://machinelearningcoban.com/2017/02/24/mlp/)
Định nghĩa: Cho $L \in \mathbb{N}, N = (N_0,N_l,\dots,N_L) \in \mathbb{N}^{L+1}$ và $\sigma:\mathbb{R} \to \mathbb{R}.$ $\sigma$ là activation function, $L$ là số lượng layers, 
và $N_0,N_L,N_l,l \in [L-1]$ lần lượt là số lượng neural đầu vào, đầu ra và layer ẩn thứ $l$. Khi đó $P(N)$ là số lượng tham số.

$$P(N)=\sum_{l=1}^{L} N_l N_{l-1}+ N_{l}$$

  

Xét ánh xạ:

$$\begin{aligned} 
\Phi_\alpha:\mathbb{R}^{N_0}\times\mathbb{R}^{P_N} &\to \mathbb{R}^{N_L} \\\\ (\mathbf{x},\theta) &\mapsto \Phi_\alpha(\mathbf{x},\theta) 
\end{aligned}$$

Với mọi $\mathbf{x} \in \mathbb{R}^{N_0}$ và tham số $\theta = (\theta^{(l)})_{l=1}^L=((W^{(l)},b^{(l)}))\_{l=1}^L \in \bigtimes\limits\_{l=1}^{L}(\mathbb{R}^{N_l\times N\_{l-1}}\times \mathbb{R}^{N\_{l}})\cong \mathbb{R}^{P(N)},$ đặt $\Phi\_{\alpha}(\mathbf{x},\theta)=\Phi^L(\mathbf{x},\theta)$ trong đó $\alpha=(N,\sigma)$ thì

$$\Phi^{(1)}:=\mathbf{Z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)},$$   

$$\bar\Phi^{(l)}:=\mathbf{a}^{(l)}=\sigma(\mathbf{Z}^{(l)}),~~l \in [L-1],$$

$$\Phi^{(l+1)}:=\mathbf{Z}^{(l+1)}=\mathbf{W}^{(l+1)}\mathbf{a}^{(l)}+\mathbf{b}^{(l+1)},~~l \in [L-1]$$

$$\bar\Phi^{(l+1)}:=\mathbf{a}^{(l+1)}=\sigma(\mathbf{Z}^{(l+1)}),~~l \in [L-1]$$

Chúng ta coi $\mathbf{W}^{(l)} \in \mathbb{R}^{N_l \times N_{l-1}}$ và $\mathbf{b}^{(l)} \in \mathbb{R}^{N_l}$ là ma trận trọng số và vector bias.
