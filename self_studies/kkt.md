# 変位距離の確率密度関数

$$
\begin{aligned}
    P(D \mid x) = \frac{x}{2\left( \delta D+ \epsilon^2\right)}\exp \left [- \frac{x^2}{4(\delta D +\epsilon^2)} \right] \tag{1}
\end{aligned}
$$
ここで，$D(\geq0)$は拡散係数，$\delta$はフレームインターバル，$\epsilon^2$は顕微鏡の座標決定誤差である．

# 1.  Maximum likelihood estimation

まず$\ln P$の$D$に関する微分を求めておく．
$$
\begin{aligned}
    \frac{\partial}{\partial D}\ln P &=& \frac{\partial}{\partial D}\left[ \ln x - \ln[2(\delta D + \epsilon^2)] -\frac{x^2}{4(\delta D+\epsilon^2)}
    \right] \\
    &=&
    \frac{\delta x^2}{(\delta D+\epsilon^2)^2}-\frac{\delta}{\delta D + \epsilon^2}
\end{aligned}
$$
式(1)の対数尤度関数は，以下のようになる．
$$
\begin{aligned}
    \ln P = \sum_n^N x_n
\end{aligned}