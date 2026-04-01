# DCPO: Dynamic Clipping Policy Optimization

---

## 1. 总结

DCPO 针对现有方法（如 GRPO、DAPO）在策略优化中存在的 **固定裁剪不合理、优势退化为零导致样本浪费、以及长度归一化破坏优劣排序** 等问题，提出了一套系统性改进方案：

- 通过 **动态自适应裁剪（DAC）** 根据旧策略概率自适应调整更新边界，缓解重要性采样方差与梯度失真；
- 通过 **平滑优势标准化（SAS）** 融合历史与当前奖励，避免零优势与训练震荡；
- 引入 **OTM loss** 去除长度归一化，保留 response 级别的相对优势关系。

整体上，DCPO 在稳定性、样本利用率和优化目标一致性上均显著优于现有方法，同时带来更高训练效率与性能提升。

[不同算法核心对比](https://zhuanlan.zhihu.com/p/1888311680880080185)

[DCPO详解](https://zhuanlan.zhihu.com/p/1949847263393343362)

[arxiv](https://arxiv.org/abs/2509.02333)

---

## 2. 主要贡献

| 模块                             | 核心问题                                                                         | 方法                                          | 带来的改进                                               |
| -------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------- |
| **动态自适应裁剪（DAC）** | 固定 clip 对不同概率 token“一刀切”，低概率 token 学习受限、高概率 token 易震荡 | clip 边界随旧概率 q(x) 动态变化               | 更合理的更新空间，降低裁剪率（↓10×），提升有效梯度利用 |
| **平滑优势标准化（SAS）**  | 同一 batch reward 相同 → advantage=0 → 无梯度；跨 step 波动大                  | 融合当前 step + 历史累计 reward，并做平滑选择 | 消除“死区”，非零梯度占比提升（↑28%），训练更稳定      |
| **OTM Loss（去长度归一）** | token-level 平均导致短答案被过度强化，破坏 response 排序                         | 以 response 为单位建模，不做长度归一          | 保留真实优势关系，避免“次优解偏好”                     |

---

## 3.背景知识

### GRPO

$$
\mathcal{T}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left( r_{i,t}\left(\theta\right)\hat{A}_{i,t}, \text{clip}(r_{i,t}\left(\theta\right),1-\epsilon,1+\epsilon)\hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

GRPO 首先对每个查询采样 G 个response，通过基于规则的奖励函数分配奖励 R，并估计token级别的优势A。

### DAPO

$$
\begin{aligned}
    \mathcal{T}_{\mathrm{DAPO}}\left(\theta\right)&={\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{~clip}\left(r_{i,t}(\theta),1-{\epsilon_{\mathrm{low}}},1+{\epsilon_{\mathrm{high}}}\right)\hat{A}_{i,t}\right)\\
& where,\ 0 <\left|\{o_i\mid{is\_equivalent}(a,o_i)\}\right|<G
\end{aligned}
$$

$0 <\left|\{o_i\mid{is\_equivalent}(a,o_i)\}\right|<G$ 表示 DAPO 将**丢弃**所有Group中reward相同的response，并重新生成response以维持批次大小，显性丢弃采样样本。

### GSPO

$$
\begin{aligned}
    \mathcal{T}_{\mathrm{GSPO}}\left(\theta\right)=&\frac{1}{G}\sum_{i=1}^G\min\left( s_{i}\left(\theta\right)\hat{A}_{i}, \text{clip}(s_{i}\left(\theta\right),1-\epsilon,1+\epsilon)\hat{A}_{i} \right) \\
    &\text{where\ } s_i(\theta) = % \left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\right)^{\frac{1}{|o_i|}}=
    exp(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}log(\frac{\pi_\theta(o_{j,t}|q)}{\pi_{\theta_{old}}(o_{j,t}|q)})) 
\end{aligned}
$$

GSPO 将标记级裁剪方法替换为序列级裁剪，然后丢弃方差较大的非零优势response，同时保留标记级概率比高的标记，也可能导致训练不稳定，并浪费了方差不为零的response中大量的有用token。

### 通常优势A计算方式

$$
\hat{A}_{j,t}^i = \frac{\left(R^i_j-\mu^i\right)}{\sigma^i} \\
$$

在之前的研究中，例如 GRPO 和 DAPO，第i步step中，第j个response 中标记 t 的优势 $\hat{A}_{j,t}^i$ 是通过将奖励 $R^i_j$  通过第 i 步生成的 G 个response的奖励均值 $\mu^i$ 和标准差 $\sigma^i$ 来计算的。当同一提示的奖励相同-> 优势为0 -> loss 为0，这些response不会对模型更新做出贡献，从而造成response浪费。

## 4.核心详情

### 4.1 动态自适应采样（DAC)

对于重要性抽样，函数 $f(x)$ 在新概率 $p(x)$ 下的期望值可以通过重要性采样权重比等价为在旧概率 $q(x)$ 下的期望值。虽然这种估计量是无偏的，但其**方差可能会显著增大**，这是重要性采样中常见的挑战：

$$
\begin{aligned}
    {Var}_{x \sim q}\left[f(x)\frac{p(x)}{q(x)}\right]- {Var}_{x \sim p}\left[f(x)\right] =&\mathbb{E}_{x \sim p}\left[f(x)^2(\frac{p(x)}{q(x)}-1)\right] =\int f(x)^2(\frac{p(x)}{q(x)}-1)p(x)\mathrm{d}x
    \end{aligned}
$$

以往的研究（例如 PPO、GRPO）通常为 $|p(x)q(x)−1|$ 设定固定的边界 $\epsilon$ 以限制该方差的偏差。这种方法没有考虑不同token的概率差异，导致旧概率值 $q(x)$ 越小, 有效梯度的 $p(x)$ 绝对空间越小。这是不合理的，因为模型对每个token的置信度越低（概率越低），后续训练中 $r(x)$ 可能越大，它能提供的信息就越有价值，**一刀切方式**不可取。我们提出了一种更实用的替代方案，即将概率作为约束条件 $|(p(x)q(x)−1)p(x)|≤\epsilon$ 纳入其中，动态自适应的随着旧概率 $q(x)$ 的不同而不同的clip范围。最终，我们得到了动态自适应裁剪边界，它能够根据不同的旧概率自适应地调整 $r(x)$ 的边界。

$$
\begin{aligned}
        0.5+\frac{1}{2}\sqrt{\max\left(1-\frac{4\epsilon_{low}}{q\left(x\right)},\ 0\right)}\leq&r\left(x\right) \leq 0.5+\frac{1}{2}\sqrt{1+\frac{4\epsilon_{high}}{q\left(x\right)}}
    \end{aligned}
$$

<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x5.png" alt="图 1" style="width:48%;">
  <img src="https://arxiv.org/html/2509.02333v2/x6.png" alt="图 2" style="width:48%;">
</div>
<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x7.png" alt="图 1" style="width:48%;">
  <img src="https://arxiv.org/html/2509.02333v2/x8.png" alt="图 2" style="width:48%;">
</div>

|    不同的剪辑方法    |                Clip裁剪阈值                | q(x) |  有效最小 p(x)  | 有效最大 q(x) |
| :-------------------: | :-----------------------------------------: | :--: | :-------------: | :------------: |
| 对称固定边界（GRPO） |              $\epsilon=0.2$              | 0.9 |      0.72      |  min(1.08,1)  |
| 非对称固定边界(DAPO) | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ | 0.9 |      0.72      |  min(1.152,1)  |
| 动态自适应界限（Our） | $\epsilon_{low}=0.16,\epsilon_{high}=0.2$ | 0.9 | **0.69** |  min(1.06,1)  |
| 对称固定边界（GRPO） |              $\epsilon=0.2$              | 0.01 |      0.008      |     0.0012     |
| 非对称固定边界(DAPO) | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ | 0.01 |      0.008      |    0.00128    |
| 动态自适应界限（our） | $\epsilon_{low}=0.16,\epsilon_{high}=0.2$ | 0.01 | **0.005** | **0.05** |

正如我们所看到的，当概率较小时，我们动态自适应裁剪方法的 $q(x)$ 值比固定方法（无论是非对称固定方法还是对称固定方法）要大得多，但又在可控的范围内。

### 4.2 平滑优势标准化(SAS)

以往的研究仅考虑当前步骤中生成response的奖励来计算优势。这种方法可能会导致以下几个问题：

1. 当response采样中的随机性导致给定步骤中的所有奖励都相同时，优势变为零，loss为0，从而导致prompt对参数更新无法做出贡献，但这些不同的推理轨迹可能存在有价值。
2. 高熵采样中的随机性会导致标签计数严重偏斜，从而导致不同步骤间标准化优势值出现大幅波动，甚至符号反转，进而破坏训练的稳定性。

我们考虑同一prompt的累积奖励来计算优势[即，将同一条prompt的已知reward进行total计算]。

$$
\begin{aligned}
        \hat{A}_{total,j}^i=\frac{\left(R^i_j-\mu_{total}^i\right)}{\sigma_{total}^i} \\
    \end{aligned}
$$

为了减轻当前步骤标准化 ${\hat{A}^i_{new,j}}$ 和累积标准化 ${\hat{A}^i_{total,j}}$的波动，我们引入了两个平滑函数${\hat{SA}^i_{new,j}}$ 和${\hat{SA}^i_{total,j}}$，它们表示两种标准化方法之间的加权平均值，权重在步骤 $i$ 中发生变化。

$$
\hat{SA}^i_{new,j} = \frac{i-1}{i}\hat{A}_{new,j}^i + \frac{1}{i}\hat{A}_{total,j}^i,\ \hat{SA}^i_{total,j} = \frac{1}{i}\hat{A}_{new,j}^i + \frac{i-1}{i}\hat{A}_{total,j}^i
$$

为了减少累积标准化和当前步骤标准化的各自波动对训练稳定性的影响，我们将最终优势 $\hat{A_j^i}$ 定义为绝对值较小的平滑优势。

$$
\hat{A}^i_j=\begin{cases} \hat{SA}^i_{new,j} , & \text{when} \ |\hat{SA}^i_{new,j}| < |\hat{SA}^i_{total,j}|\\
        \hat{SA}^i_{total,j} , & \text{otherwise}
    \end{cases}
$$

一旦prompt参与模型优化，该提示的response将在后续步骤中都参与模型更新。当前step中的奖励相同时，它们将以优势$\frac{1}{i}\hat{A_{total,j}^i}$参与loss计算。

# OTM

$$
\begin{aligned}
    \mathcal{T}_{\mathrm{DCPO}}\left(\theta\right)&={\frac{1}{G}\sum_{i=1}^G\sum_{t=1}^{|o_i|}}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{~clip}\left(r_{i,t}(\theta),1-{\epsilon_{\mathrm{low}}},1+{\epsilon_{\mathrm{high}}}\right)\hat{A}_{i,t}\right)
\end{aligned}
$$

| —        | Advantage(优势值) | 长度 | GRPO中每个token的权重 | DCPO中每个token权重 |
| :-------- | ----------------- | ---- | --------------------- | ------------------- |
| Response1 | 1                 | 500  | 1/500                 | 1                   |
| Response2 | 2                 | 2000 | 1/2000                | 2                   |

- 对于DAPO、GRPO中考虑长度可以看出来 response1 的优势只值为1，小于 response2的优势值2， 人类更倾向于response2， 但如果考虑长度，response1 的每个token的权重是1/500 > response2的每个token的权重1/2000. 所以loss的计算使得模型学习的更多是response1，而不是response2. 这会导致模型更看重的次优结果，而不是最优结果。
- 所以**DCPO中为了保证advantage不被破坏，忽略长度对loss的影响。**
