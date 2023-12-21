# [Machine Learning cơ bản](https://machinelearningcoban.com/about/)
## [Neural Networks](https://machinelearningcoban.com/2017/02/24/mlp/)
Định nghĩa: Cho $L \in \mathbb{N}, d = (d^{(0)},d^{(l)},\dots,d^{(L)}) \in \mathbb{N}^{L+1}$ và $\sigma:\mathbb{R} \to \mathbb{R}.$ $\sigma$ là activation function, $L$ là số lượng layers, 
và $d^{(0)},d^{(L)},d^{(l)},l \in [L-1]$ lần lượt là số lượng neural đầu vào, đầu ra và layer ẩn thứ $l$. Khi đó $P(d)$ là số lượng tham số.
$$P(d)=\sum_{l=1}^{L} d^{(l-1)} d^{(l)}+ d^{(l)}$$
Xét ánh xạ:

$$\begin{aligned} 
\Phi_\alpha:\mathbb{R}^{d^{(0)}}\times\mathbb{R}^{P(d)} &\to \mathbb{R}^{d^{(L)}} \\\\ (\mathbf{x},\theta) &\mapsto \Phi_\alpha(\mathbf{x},\theta) 
\end{aligned}$$

Với mọi $\mathbf{x} \in \mathbb{R}^{d^{(0)}}$ và tham số $\theta = (\theta^{(l)})_{l=1}^L=((\mathbf{W}^{(l)},\mathbf{b}^{(l)}))\_{l=1}^L \in \bigtimes\limits\_{l=1}^{L}(\mathbb{R}^{d^{(l-1)}\times d^{(l)}}\times \mathbb{R}^{d\^{(l)}})\cong \mathbb{R}^{P(d)},$ đặt $\Phi\_{\alpha}(\mathbf{x},\theta)=\Phi^L(\mathbf{x},\theta)$ trong đó $\alpha=(d,\sigma)$ thì

$$\Phi^{(1)}:=\mathbf{z}^{(1)}=\mathbf{W}^{(l)T}\mathbf{x}+\mathbf{b}^{(1)},$$   

$$\bar\Phi^{(l)}:=\mathbf{a}^{(l)}=\sigma(\mathbf{z}^{(l)}),~~l \in [L-1],$$

$$\Phi^{(l+1)}:=\mathbf{z}^{(l+1)}=\mathbf{W}^{(l+1)T}\mathbf{a}^{(l)}+\mathbf{b}^{(l+1)},~~l \in [L-1]$$

$$\bar\Phi^{(l+1)}:=\mathbf{a}^{(l+1)}=\sigma(\mathbf{z}^{(l+1)}),~~l \in [L-1]$$

Chúng ta coi $\mathbf{W}^{(l)} \in \mathbb{R}^{d^{(l-1)} \times d^{(l)}}$ và $\mathbf{b}^{(l)} \in \mathbb{R}^{d^{(l)}}$ là ma trận trọng số và vector bias.
## [Backpropagation cho Stochastic Gradient Descent](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/?fbclid=IwAR2awLv1m6QkU7pDlpusUjOOrv4R61TSFLyllhuTPneuxUTpQhJmB3s3Is8)
Đặt $\mathbf{e}^{(l)} = \left(e_1^{(l)}, e_2^{(l)}, …, e_{d^{(l)}}^{(l)}\right)^T \in \mathbb{R}^{d^{(l)}\times 1}$. Ta sẽ có quy tắc tính như sau:  
B1. Feedforward: Với 1 giá trị đầu vào $\mathbf{x}$, tính giá trị đầu ra của network, trong quá trình tính toán, lưu lại các activation $\mathbf{a}^{(l)}$ tại mỗi layer.

  $$\begin{aligned}
\mathbf{a}^{(0)} &= \mathbf{x} \\\\ z_{i}^{(l)} &= \mathbf{w}_i^{(l)T}\mathbf{a}^{(l-1)} + b_i^{(l)} \\\\
\mathbf{z}^{(l)} &= \mathbf{W}^{(l)T}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)},~~ l =  1, 2, \dots, L \\\\
\mathbf{a}^{(l)} &= f(\mathbf{z}^{(l)}), ~~ l =  1, 2, \dots, L \\\\
\mathbf{\hat{y}} &= \mathbf{a}^{(L)}
\end{aligned}$$

B2. Tính:

$$\begin{aligned}\frac{\partial J}{\partial \mathbf{W}^{(L)}} &= \left( \frac{\partial J}{\partial \mathbf{w}_1^{(L)}},\frac{\partial J}{\partial \mathbf{w}_2^{(L)}},\dots,\frac{\partial J}{\partial \mathbf{w}\_{d^{(L)}}^{(L)}} \right) = \begin{pmatrix}
\frac{\partial J}{\partial \mathbf{w}\_{11}^{(L)}}&\frac{\partial J}{\partial \mathbf{w}\_{12}^{(L)}}& \frac{\partial J}{\partial \mathbf{w}\_{1d^{(L)}}^{(L)}}\\ 
\frac{\partial J}{\partial \mathbf{w}\_{21}^{(L)}}&\frac{\partial J}{\partial \mathbf{w}\_{22}^{(L)}}& \frac{\partial J}{\partial \mathbf{w}\_{2d^{(L)}}^{(L)}}\\ 
\frac{\partial J}{\partial \mathbf{w}\_{d^{(L-1)}1}^{(L)}}&\frac{\partial J}{\partial \mathbf{w}\_{d^{(L-1)}2}^{(L)}}&\frac{\partial J}{\partial\mathbf{w}\_{d^{(L-1)}d^{(L)}}^{(L)}}
\end{pmatrix}
\end{aligned}$$

$$=\begin{pmatrix}
e_1^{(L)}a_1^{(L-1)}&e_2^{(L)}a_1^{(L-1)}&e\_{d^{(L)}}^{(L)}a_1^{(L-1)}\\ 
e_1^{(L)}a_2^{(L-1)}&e_2^{(L)}a_2^{(L-1)}&e\_{d^{(L)}}^{(L)}a_2^{(L-1)}\\ 
e_1^{(L)}a\_{d^{(L-1)}}^{(L-1)}&e_2^{(L)}a\_{d^{(L-1)}}^{(L-1)}&e\_{d^{(L)}}^{(L)}a\_{d^{(L-1)}}^{(L-1)}\\
\end{pmatrix}= \mathbf{a}^{(L-1)}\mathbf{e}^{(L)T}$$

Tương tự:
$$\frac{\partial J}{\partial \mathbf{b}^{(L)}}=\mathbf{e}^{(L)}$$
Trong đó $\mathbf{e}^{(L)} = \left\(e_1^{(L)}, e_2^{(L)},\dots, e_{d^{(L)}}^{(L)}\right)^T \in \mathbb{R}^{d^{(L)}\times 1}$  

B3. Với $l = L-1,L-2,\dots,1$, tính:

$$\begin{aligned}\frac{\partial J}{\partial \mathbf{W}^{(l)}} &= \left( \frac{\partial J}{\partial \mathbf{w}_1^{(l)}},\frac{\partial J}{\partial \mathbf{w}_2^{(l)}},\dots,\frac{\partial J}{\partial \mathbf{w}\_{d^{(l)}}^{(l)}} \right) = \begin{pmatrix}
\frac{\partial J}{\partial \mathbf{w}\_{11}^{(l)}}&\frac{\partial J}{\partial \mathbf{w}\_{12}^{(l)}}& \frac{\partial J}{\partial \mathbf{w}\_{1d^{(l)}}^{(l)}}\\ 
\frac{\partial J}{\partial \mathbf{w}\_{21}^{(l)}}&\frac{\partial J}{\partial \mathbf{w}\_{22}^{(l)}}& \frac{\partial J}{\partial \mathbf{w}\_{2d^{(l)}}^{(l)}}\\ 
\frac{\partial J}{\partial \mathbf{w}\_{d^{(l-1)}1}^{(l)}}&\frac{\partial J}{\partial \mathbf{w}\_{d^{(l-1)}2}^{(l)}}&\frac{\partial J}{\partial\mathbf{w}\_{d^{(l-1)}d^{(l)}}^{(l)}}
\end{pmatrix}
\end{aligned}$$

$$=\begin{pmatrix}
e_1^{(l)}a_1^{(l-1)}&e_2^{(l)}a_1^{(l-1)}&e_{d^{(l)}}^{(l)}a_1^{(l-1)}\\
e_1^{(l)}a_2^{(l-1)}&e_2^{(l)}a_2^{(l-1)}&e_{d^{(l)}}^{(l)}a_2^{(l-1)}\\
e_1^{(l)}a_{d^{(l-1)}}^{(l-1)}&e_2^{(l)}a_{d^{(l-1)}}^{(l-1)}&e_{d^{(l)}}^{(l)}a_{d^{(l-1)}}^{(l-1)}\\
\end{pmatrix}= \mathbf{a}^{(l-1)}\mathbf{e^{(l)T}}$$

Trong đó:

$$\mathbf{e}^{(l)} = \begin{pmatrix}e_1^{(l)} & e_2^{(l)} \dots e_{d^{(l)}}^{(l)}\end{pmatrix}^T=\begin{pmatrix}
\frac{\partial J}{\partial z_1^{(l)}}&\frac{\partial J}{\partial z_2^{(l)}} \dots \frac{\partial J}{\partial z_{d^{(l)}}^{(l)}}\end{pmatrix}^T=\begin{pmatrix}
\frac{\partial J}{\partial a_1^{(l)}}\cdot \frac{\partial a_1^{(l)}}{\partial z_1^{(l)}}\\ 
\frac{\partial J}{\partial a_2^{(l)}}\cdot \frac{\partial a_2^{(l)}}{\partial z_2^{(l)}}\\ 
\vdots \\ 
\frac{\partial J}{\partial a_{d^{(l)}}^{(l)}}\cdot \frac{\partial a_{d^{(l)}}^{(l)}}{\partial z_{d^{(l)}}^{(l)}}
\end{pmatrix}=\begin{pmatrix}
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(\frac{\partial J}{\partial z_k^{(l+1)}}\cdot \frac{\partial z_k^{(l+1)}}{\partial a_1^{(l)}}\right) {\sigma}'(z_1^{(l)})\\ 
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(\frac{\partial J}{\partial z_k^{(l+1)}}\cdot \frac{\partial z_k^{(l+1)}}{\partial a_2^{(l)}}\right) {\sigma}'(z_2^{(l)})\\
\vdots \\
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(\frac{\partial J}{\partial z_k^{(l+1)}}\cdot \frac{\partial z_k^{(l+1)}}{\partial a_{d^{(l)}}^{(l)}}\right) {\sigma}'(z_{d^{(l)}}^{(l)})
\end{pmatrix}$$

$$=\begin{pmatrix}
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(e_k^{(l+1)} w_{1k}^{(l+1)}\right) {\sigma}'(z_1^{(l)})\\ 
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(e_k^{(l+1)} w_{2k}^{(l+1)}\right) {\sigma}'(z_2^{(l)})\\
\vdots \\
\displaystyle\sum_{k=1}^{d^{(l+1)}}\left(e_k^{(l+1)} w_{d^{(l)}k}^{(l+1)}\right) {\sigma}'(z_{d^{(l)}}^{(l)})
\end{pmatrix}=\begin{pmatrix}
\left(\mathbf{w}\_{1:}^{(l+1)}\mathbf{e^{(l+1)}}\right){\sigma}'(z_1^{(l)})\\
\left(\mathbf{w}\_{2:}^{(l+1)}\mathbf{e^{(l+1)}}\right){\sigma}'(z_2^{(l)})\\
\vdots\\
\left(\mathbf{w}\_{d^{(l)}:}^{(l+1)}\mathbf{e^{(l+1)}}{\sigma}'(z_{d^{(l)}}^{(l)})\right)
\end{pmatrix}=\left(\mathbf{W}^{(l+1)}\mathbf{e^{(l+1)}}\right)\odot{\sigma}'(\mathbf{z}^{(l)})$$

Tương tự:
$$\frac{\partial J}{\partial \mathbf{b}^{(l)}}=\mathbf{e}^{(l)}$$
