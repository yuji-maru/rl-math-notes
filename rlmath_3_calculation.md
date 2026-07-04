---
tags:
  - 強化学習
  - 演習
  - 手計算
aliases:
  - RL手計算演習
author: Claude Fable 5 (medium)
---
# 強化学習演習 手計算編 — 導出を自分の手で再構成する

> [!note] シリーズ全体の目次
> - [[強化学習の数理_1_基礎編]] — 古典的RL理論(MDP・Bellman方程式・DP・TD・方策勾配)
> - [[強化学習の数理_2_深層学習編]] — 深層強化学習(DQN・A2C・PPO・SAC等)
> - **[[強化学習の数理_3_演習_手計算編]]**(本稿) — 上記2冊の導出を手計算で再構成する演習
> - [[強化学習の数理_4_演習_実践コーディング編]] — 上記2冊のアルゴリズムをコードで実装する演習

> [!abstract] 本演習テキストについて
> 本テキストは、解説テキスト [[強化学習の数理_1_基礎編]](古典理論編)および [[強化学習の数理_2_深層学習編]](深層編)の**導出過程を、誘導小問に従って自分の手で再構成する**ための演習問題集である。
>
> - 各問題は「その計算をすると理論の目的が自然に見える」順序で小問に分割してある。**小問は飛ばさず順番に解くこと。**
> - 導出に必要な定義式・公式は、各問題冒頭の `[!info] 使う道具` に**すべて**掲げた。原則としてこの道具と直前の小問の結果だけで解ける。
> - 解答は折りたたみの `[!success]- 解答` に収めた。まず自力で計算し、答え合わせに使うこと。
> - 記号は [[強化学習の数理_1_基礎編#付録C 記法一覧]] に従う。
> - 計算機で確かめたくなったら、姉妹編 [[強化学習の数理_4_演習_実践コーディング編]] の対応課題へ進むとよい。

---

## 第I部 古典理論編([[強化学習の数理_1_基礎編]] 対応)

---

### 問題1 割引収益と実効ホライズン

> [!info] 使う道具
> - 割引収益の定義:$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$,$\gamma \in [0,1)$
> - 幾何級数の和:$\sum_{k=0}^\infty x^k = \dfrac{1}{1-x}$($|x|<1$)
> - 参照:[[強化学習の数理_1_基礎編#第2章 収益と価値関数 — 「良さ」の定量化]]

**問1.1** すべての時刻で報酬が定数 $R_{t} = r$ のとき、$G_t$ を $r, \gamma$ で表せ。

**問1.2** 報酬が有界 $|R_t| \le R_{\max}$ のとき、$|G_t| \le \dfrac{R_{\max}}{1-\gamma}$ を示せ(三角不等式 → 幾何級数の順に評価する)。この結果は「$\gamma < 1$ なら価値関数が必ず有限に定義できる」ことを意味する。

**問1.3** 定義の無限和を $k=0$ の項とそれ以外に分けて括り直すことで、再帰関係
$$G_t = R_{t+1} + \gamma\, G_{t+1}$$
を導け。**この1行が以後のすべての方程式の源泉である。**

**問1.4** 重み $\gamma^k = e^{k \ln \gamma}$ と、$\gamma \to 1$ での近似 $\ln \gamma \approx -(1-\gamma)$ を使い、重みが $1/e$ に減衰するステップ数(実効ホライズン)が $\tau_{\text{eff}} \approx \dfrac{1}{1-\gamma}$ となることを確かめよ。$\gamma = 0.99$ のとき $\tau_{\text{eff}}$ はいくつか。

> [!success]- 解答
> **1.1** $G_t = r\sum_k \gamma^k = \dfrac{r}{1-\gamma}$。
> **1.2** $|G_t| \le \sum_k \gamma^k |R_{t+k+1}| \le R_{\max}\sum_k \gamma^k = \dfrac{R_{\max}}{1-\gamma}$。
> **1.3** $G_t = R_{t+1} + \sum_{k=1}^\infty \gamma^k R_{t+k+1} = R_{t+1} + \gamma \sum_{j=0}^\infty \gamma^j R_{(t+1)+j+1} = R_{t+1} + \gamma G_{t+1}$($j = k-1$ と置換)。
> **1.4** $e^{k\ln\gamma} = e^{-1}$ ⟺ $k = -1/\ln\gamma \approx 1/(1-\gamma)$。$\gamma=0.99$ で $\tau_{\text{eff}} \approx 100$ ステップ。

---

### 問題2 $V$ と $Q$ の関係、アドバンテージの平均ゼロ性

> [!info] 使う道具
> - 定義:$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$,$Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$
> - 全確率則(タワー性):$\mathbb{E}[X \mid B] = \sum_a \Pr(A = a \mid B)\, \mathbb{E}[X \mid B, A = a]$
> - アドバンテージ:$A^\pi(s,a) := Q^\pi(s,a) - V^\pi(s)$
> - 参照:[[強化学習の数理_1_基礎編#第2章 収益と価値関数 — 「良さ」の定量化]] 2.2–2.3節

**問2.1** タワー性で条件 $B = \{S_t = s\}$、確率変数 $X = G_t$ ととり、
$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a) \tag{2.2}$$
を導け(「$V$ は $Q$ の方策平均」)。

**問2.2** (2.2)を使って $\sum_a \pi(a \mid s)\, A^\pi(s,a) = 0$ を示せ。この「方策自身のもとでアドバンテージの平均はゼロ」という性質は、問題10のベースライン、[[強化学習の数理_2_深層学習編]] のDueling Networkのゲージ固定(問題12)で繰り返し使う。

**問2.3** ある状態 $s$ で $Q^\pi(s, a_1) = 4$, $Q^\pi(s, a_2) = 1$, $\pi(a_1 \mid s) = 0.25$ とする($|\mathcal{A}| = 2$)。$V^\pi(s)$、$A^\pi(s, a_1)$、$A^\pi(s, a_2)$ を求め、問2.2の性質を数値で検算せよ。

> [!success]- 解答
> **2.1** $V^\pi(s) = \mathbb{E}[G_t \mid s] = \sum_a \Pr(A_t = a \mid S_t = s)\,\mathbb{E}[G_t \mid s, a] = \sum_a \pi(a|s)\,Q^\pi(s,a)$。
> **2.2** $\sum_a \pi A^\pi = \sum_a \pi Q^\pi - V^\pi \sum_a \pi(a|s) = V^\pi - V^\pi \cdot 1 = 0$。
> **2.3** $V^\pi = 0.25 \times 4 + 0.75 \times 1 = 1.75$。$A^\pi(s,a_1) = 2.25$,$A^\pi(s,a_2) = -0.75$。検算:$0.25 \times 2.25 + 0.75 \times (-0.75) = 0.5625 - 0.5625 = 0$。✓

---

### 問題3 Bellman期待方程式の導出(全ステップ再構成)

> [!info] 使う道具
> - 収益の再帰(問1.3):$G_t = R_{t+1} + \gamma G_{t+1}$
> - タワー性(問題2)
> - MDPの構成要素:$\pi(a|s)$,$P(s'|s,a)$,$R(s,a) = \mathbb{E}[R_{t+1} \mid s, a]$
> - **マルコフ性**:$S_{t+1} = s'$ が与えられれば、それ以降の軌道の分布は $(S_t, A_t)$ に依存しない
> - 参照:[[強化学習の数理_1_基礎編#第3章 Bellman期待方程式 — 価値の自己無撞着方程式]]

**問3.1** $G_t = R_{t+1} + \gamma G_{t+1}$ の両辺に $\mathbb{E}_\pi[\cdot \mid S_t = s]$ をとり、
$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} \mid S_t = s] + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid S_t = s]$$
と書け(期待値の線形性のみ)。

**問3.2** 第1項を行動 $a$ で周辺化し、$\sum_a \pi(a|s)\, R(s,a)$ になることを示せ。

**問3.3** 第2項を行動 $a$ と次状態 $s'$ で二重に周辺化し、
$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\; \mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$$
まで書き下せ。

**問3.4** 右端の条件付き期待値に**マルコフ性**と**方策の定常性**を適用し、それが $V^\pi(s')$ に等しいことを論証せよ(どこでどちらの仮定を使ったかを明記すること。ここがこの導出の心臓部である)。

**問3.5** 以上を合成して **Bellman期待方程式**
$$V^\pi(s) = \sum_a \pi(a|s)\left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V^\pi(s') \right] \tag{3.1}$$
を完成させよ。

**問3.6** 同じ手順を条件 $\{S_t = s, A_t = a\}$ で繰り返し、
$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V^\pi(s') \tag{3.2}$$
を導け。さらに(2.2)を(3.2)の $V^\pi(s')$ に代入して、$Q$ だけで閉じた式(3.3)を書け。

> [!success]- 解答
> **3.1** 期待値の線形性より直ちに従う。
> **3.2** タワー性:$\mathbb{E}_\pi[R_{t+1}|s] = \sum_a \pi(a|s)\,\mathbb{E}[R_{t+1}|s,a] = \sum_a \pi(a|s) R(s,a)$。
> **3.3** タワー性を $A_t$、次いで $S_{t+1}$ について適用するのみ。
> **3.4** マルコフ性により、$S_{t+1} = s'$ を与えたとき $G_{t+1}$(時刻 $t+1$ 以降の報酬の関数)の分布は $(S_t, A_t)$ によらない。よって条件から $(s, a)$ を落とせて $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$。方策が定常(時刻非依存)なので、この量は時刻 $t+1$ を明示せずとも同じ関数 $V^\pi$ で書け、$= V^\pi(s')$。
> **3.5** 3.2と3.4を3.1に代入し、$\sum_a \pi(a|s)$ で括る。
> **3.6** 条件に $A_t = a$ が入るため方策平均が不要になり、$Q^\pi(s,a) = R(s,a) + \gamma\sum_{s'} P(s'|s,a) V^\pi(s')$。代入して
> $Q^\pi(s,a) = R(s,a) + \gamma\sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s',a')$。

---

### 問題4 線形方程式としての厳密解 — 2状態MRPを解く

> [!info] 使う道具
> - ベクトル形:$\boldsymbol{v}^\pi = \boldsymbol{r}^\pi + \gamma P^\pi \boldsymbol{v}^\pi$,すなわち $(I - \gamma P^\pi)\boldsymbol{v}^\pi = \boldsymbol{r}^\pi$
> - $2\times 2$ 逆行列:$\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \dfrac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$
> - Neumann級数:$(I - \gamma P^\pi)^{-1} = \sum_{k=0}^\infty (\gamma P^\pi)^k$
> - 参照:[[強化学習の数理_1_基礎編#第3章 Bellman期待方程式 — 価値の自己無撞着方程式]] 3.3節

状態 $\{s_1, s_2\}$、$\gamma = 0.5$ のマルコフ報酬過程を考える:
$$P^\pi = \begin{pmatrix} 0.5 & 0.5 \\ 0 & 1 \end{pmatrix}, \qquad \boldsymbol{r}^\pi = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$$
($s_1$ からは半々で自分か $s_2$ へ移り期待報酬2、$s_2$ は報酬0の吸収状態。)

**問4.1** $P^\pi$ が確率行列(各行の和が1、成分非負)であることを確認し、行列 $I - \gamma P^\pi$ を成分で書き下せ。

**問4.2** $2\times2$ 逆行列公式で $(I - \gamma P^\pi)^{-1}$ を計算し、$\boldsymbol{v}^\pi = (I-\gamma P^\pi)^{-1}\boldsymbol{r}^\pi$ を求めよ。

**問4.3** 別解として、$s_2$ が吸収状態(以後報酬0)であることから $V^\pi(s_2) = 0$ を直接論証し、$s_1$ に関するBellman方程式(3.1)を1変数方程式として解いて、問4.2と一致することを確かめよ。

**問4.4** Neumann級数の最初の3項 $\boldsymbol{r}^\pi + \gamma P^\pi \boldsymbol{r}^\pi + \gamma^2 (P^\pi)^2 \boldsymbol{r}^\pi$ を計算し、部分和が問4.2の厳密解にどう近づくかを観察せよ。第 $k$ 項が「$k$ ステップ後の期待報酬の割引」という意味をもつことを、$s_1$ 成分について言葉で説明せよ。

> [!success]- 解答
> **4.1** $I - \gamma P^\pi = \begin{pmatrix} 0.75 & -0.25 \\ 0 & 0.5 \end{pmatrix}$。
> **4.2** 行列式 $= 0.375$。逆行列 $= \dfrac{1}{0.375}\begin{pmatrix} 0.5 & 0.25 \\ 0 & 0.75\end{pmatrix} = \begin{pmatrix} 4/3 & 2/3 \\ 0 & 2 \end{pmatrix}$。$\boldsymbol{v}^\pi = \begin{pmatrix} 4/3 & 2/3 \\ 0 & 2 \end{pmatrix}\begin{pmatrix}2\\0\end{pmatrix} = \begin{pmatrix} 8/3 \\ 0 \end{pmatrix}$。
> **4.3** $s_2$ 以降の報酬はすべて0なので $G_t = 0$、よって $V(s_2) = 0$。$V(s_1) = 2 + 0.5\,(0.5 V(s_1) + 0.5 \cdot 0)$ ⟹ $V(s_1)(1 - 0.25) = 2$ ⟹ $V(s_1) = 8/3$。一致。✓
> **4.4** 第0項 $(2,0)^\top$、第1項 $\gamma P^\pi \boldsymbol r = 0.5\,(1, 0)^\top = (0.5, 0)^\top$、第2項 $\gamma^2 (P^\pi)^2 \boldsymbol r$:$(P^\pi)^2 = \begin{pmatrix}0.25 & 0.75\\ 0 & 1\end{pmatrix}$ より $(0.125, 0)^\top$。部分和 $s_1$ 成分:$2 \to 2.5 \to 2.625 \to \cdots \to 8/3 \approx 2.667$。第 $k$ 項の $s_1$ 成分は「$s_1$ に $k$ ステップ後もまだ留まっている確率 $0.5^k$ × 期待報酬 $2$ × 割引 $\gamma^k$」で、遠い将来の寄与ほど幾何的に小さい。

---

### 問題5 Bellman最適方程式:$\max$ の出どころ

> [!info] 使う道具
> - 最適価値の定義:$V^*(s) = \max_\pi V^\pi(s)$,$Q^*(s,a) = \max_\pi Q^\pi(s,a)$
> - (2.2)の最適方策版:$V^*(s) = \sum_a \pi^*(a|s)\, Q^*(s,a)$
> - (3.2)の導出手順(問3.6)
> - 参照:[[強化学習の数理_1_基礎編#第4章 最適性とBellman最適方程式]]

**問5.1** 「平均は最大値を超えない」($\sum_a p_a x_a \le \max_a x_a$、$p$ は確率分布)を使い、$V^*(s) \le \max_a Q^*(s,a)$ を示せ。

**問5.2** 逆向きの不等号を、背理法で示せ:もし $V^*(s) < \max_a Q^*(s,a)$ なら、状態 $s$ で $\arg\max_a Q^*(s,a)$ を選びその後 $\pi^*$ に従う方策の $s$ からの期待収益はいくらになるか。それは何と矛盾するか。

**問5.3** 問5.1・5.2から $V^*(s) = \max_a Q^*(s,a)$(式(4.1))を結論し、これを $Q^*(s,a) = R(s,a) + \gamma\sum_{s'}P(s'|s,a)V^*(s')$(式(4.2)。導出は問3.6と同一手順)と互いに代入して、**Bellman最適方程式**の2つの形(4.3)(4.4)を書き下せ。

**問5.4** 期待方程式(3.1)と最適方程式(4.3)を並べ、両者の違いが「$\sum_a \pi(a|s)(\cdot)$ ↔ $\max_a(\cdot)$」の置き換え**だけ**であることを確認せよ。この置き換えが方程式の線形性をどう変えるか、一言で述べよ(解法にどんな帰結をもたらすかは問題7で確かめる)。

> [!success]- 解答
> **5.1** $V^*(s) = \sum_a \pi^*(a|s) Q^*(s,a) \le \max_a Q^*(s,a)$。
> **5.2** その方策の期待収益は $\max_a Q^*(s,a) > V^*(s)$ となり、$V^*$ が全方策にわたる最大値であることに矛盾。よって $V^*(s) \ge \max_a Q^*(s,a)$。
> **5.3** $V^*(s) = \max_a\left[R(s,a) + \gamma\sum_{s'}P(s'|s,a)V^*(s')\right]$(4.3)、$Q^*(s,a) = R(s,a) + \gamma\sum_{s'}P(s'|s,a)\max_{a'}Q^*(s',a')$(4.4)。
> **5.4** 期待方程式は未知数について**線形**(逆行列で閉じた解が書ける)、最適方程式は $\max$ により**非線形**(区分線形)。閉形式の解がなく、反復法で解くしかない——その収束保証が問題6の縮小性である。

---

### 問題6 縮小写像:理論の心臓部を証明する

> [!info] 使う道具
> - Bellman作用素:$(T^\pi v)(s) = \sum_a \pi(a|s)[R(s,a) + \gamma\sum_{s'}P(s'|s,a)v(s')]$,$(T^* v)(s) = \max_a[R(s,a) + \gamma\sum_{s'}P(s'|s,a)v(s')]$
> - supノルム:$\|v\|_\infty = \max_s |v(s)|$
> - $\pi(\cdot|s)$ と $P(\cdot|s,a)$ は確率分布(非負・和が1)
> - 参照:[[強化学習の数理_1_基礎編#第5章 Bellman作用素と不動点定理 — 理論の心臓部]]、[[強化学習の数理_1_基礎編#付録A Banachの不動点定理の証明]]

**問6.1** 任意の $u, v$ に対し、$(T^\pi u)(s) - (T^\pi v)(s)$ を計算せよ。報酬項が消えることを確認し、残る式に(i)三角不等式、(ii)$|u(s') - v(s')| \le \|u - v\|_\infty$、(iii)確率の和が1、を順に適用して
$$\|T^\pi u - T^\pi v\|_\infty \le \gamma \|u - v\|_\infty$$
を証明せよ。各行でどの道具を使ったかを明記すること。

**問6.2**(補題)$a^\dagger := \arg\max_a f(a)$ とおいて、
$$\max_a f(a) - \max_a g(a) \le \max_a |f(a) - g(a)|$$
を示せ(ヒント:$\max_a g(a) \ge g(a^\dagger)$)。$f \leftrightarrow g$ を入れ替えることで絶対値付きの主張 $|\max f - \max g| \le \max|f-g|$ を完成させよ。

**問6.3** 問6.2の補題を $f(a) = R(s,a) + \gamma\sum_{s'}P(s'|s,a)u(s')$、$g(a)$ をその $v$ 版として適用し、$T^*$ も $\gamma$-縮小であることを示せ。**なぜ $T^\pi$ の証明がそのままでは通らず補題が必要なのか**($\max$ は線形でないため差し引きで報酬が消せない)を一言で述べよ。

**問6.4**(Banachの不動点定理の核)縮小写像 $T$ の反復列 $x_{k+1} = Tx_k$ について:
(a) $d(x_{k+1}, x_k) \le \gamma^k d(x_1, x_0)$ を縮小性の反復適用で示せ。
(b) $m > n$ に対し三角不等式と幾何級数で $d(x_m, x_n) \le \dfrac{\gamma^n}{1-\gamma} d(x_1, x_0)$ を示し、$\{x_k\}$ がCauchy列であることを結論せよ。
(c) 不動点の一意性:$x^*, y^*$ がともに不動点なら $d(x^*, y^*) \le \gamma\, d(x^*, y^*)$ を導き、$x^* = y^*$ を結論せよ。

**問6.5**(収束レートの実用計算)価値反復の誤差は $\|v_k - V^*\|_\infty \le \gamma^k \|v_0 - V^*\|_\infty$ で減衰する。初期誤差100、許容誤差 $10^{-3}$ のとき、$\gamma = 0.9$ と $\gamma = 0.99$ でそれぞれ必要な反復回数の目安を $k \ge \dfrac{\ln(10^5)}{\ln(1/\gamma)}$ から求めよ($\ln 10 \approx 2.303$、$\ln(1/0.9) \approx 0.105$、$\ln(1/0.99) \approx 0.01$)。「$\gamma \to 1$ で反復回数が $1/(1-\gamma)$ に比例して増える」ことを確認せよ。

> [!success]- 解答
> **6.1** $(T^\pi u - T^\pi v)(s) = \gamma\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)[u(s')-v(s')]$(報酬は差し引きで消える)。絶対値をとり三角不等式、次に各 $|u-v| \le \|u-v\|_\infty$ で押さえ、確率の和が1なので $\le \gamma\|u-v\|_\infty$。全 $s$ で成立するので左辺の $\max_s$ をとって完成。
> **6.2** $\max f - \max g = f(a^\dagger) - \max_a g(a) \le f(a^\dagger) - g(a^\dagger) \le \max_a|f - g|$。入れ替えて逆向きも同じ上界で押さえられるので絶対値の主張を得る。
> **6.3** $|(T^*u)(s) - (T^*v)(s)| \le \max_a|f(a) - g(a)| = \max_a \gamma|\sum_{s'}P(s'|s,a)(u-v)(s')| \le \gamma\|u-v\|_\infty$。$\max$ 同士の差は各 $a$ の差に分配できない(異なる $a$ で最大が達成されうる)ため、補題で「差の最大」に持ち替える必要がある。
> **6.4** (a) $d(x_{k+1},x_k) = d(Tx_k, Tx_{k-1}) \le \gamma d(x_k, x_{k-1}) \le \cdots \le \gamma^k d(x_1,x_0)$。(b) $d(x_m,x_n) \le \sum_{k=n}^{m-1}\gamma^k d(x_1,x_0) \le \frac{\gamma^n}{1-\gamma}d(x_1,x_0) \to 0$。(c) $d(x^*,y^*) = d(Tx^*,Ty^*) \le \gamma d(x^*,y^*)$、$\gamma<1$ より $d=0$。
> **6.5** $\gamma=0.9$:$k \ge 5\ln 10 / 0.105 \approx 110$ 回。$\gamma=0.99$:$k \ge 11.5/0.01 \approx 1152$ 回。$\ln(1/\gamma) \approx 1-\gamma$ なので反復回数は $\propto 1/(1-\gamma)$。

---

### 問題7 方策反復を手で回す — 2状態MDP

> [!info] 使う道具
> - [[強化学習の数理_1_基礎編#第6章 動的計画法 — モデルが既知のときの厳密解法]] 6.6節と同じ2状態MDP(決定論的遷移):
>
> | 状態 | 行動 | 次状態 | 報酬 |
> | ---- | ---- | ---- | ---- |
> | $s_1$ | stay | $s_1$ | $+1$ |
> | $s_1$ | move | $s_2$ | $0$ |
> | $s_2$ | stay | $s_2$ | $+2$ |
> | $s_2$ | move | $s_1$ | $0$ |
>
> - ただし本問では割引率を **$\gamma = 0.3$** に変える(解説テキストの $\gamma = 0.9$ と結論が変わる。それが狙いである)。
> - 方策評価:決定論的方策なら $V^\pi(s) = R(s, \pi(s)) + \gamma V^\pi(s_{\text{next}})$
> - 方策改善:$\pi'(s) = \arg\max_a Q^\pi(s,a)$,$Q^\pi(s,a) = R(s,a) + \gamma V^\pi(s')$

**問7.1** 方策 $\pi_0$ =「常に stay」を評価せよ($V^{\pi_0}(s_1), V^{\pi_0}(s_2)$ を1変数方程式として解く)。

**問7.2** $Q^{\pi_0}(s, a)$ を4つすべて計算し、貪欲化した $\pi_1$ を求めよ。$\gamma = 0.9$ のとき(解説テキスト6.6節)と結論がどう違うか。

**問7.3** $\pi_1$ を評価し、再度貪欲化して方策が変化しないこと(=停止)を確認せよ。得られた $V$ がBellman最適方程式(4.3)を満たすことを、両状態で代入検算せよ。

**問7.4** 「$\gamma = 0.9$ では $s_1$ から move が最適、$\gamma = 0.3$ では stay が最適」となった理由を、問1.4の実効ホライズン $\tau_{\text{eff}} \approx 1/(1-\gamma)$ の言葉で説明せよ(計算問題:それぞれの $\tau_{\text{eff}}$ を求め、moveの「1ステップの投資」を回収できるか比較する)。

> [!success]- 解答
> **7.1** $V^{\pi_0}(s_1) = 1 + 0.3 V^{\pi_0}(s_1) \Rightarrow 1/0.7 = 10/7 \approx 1.429$。$V^{\pi_0}(s_2) = 2/0.7 = 20/7 \approx 2.857$。
> **7.2** $Q(s_1,\text{stay}) = 1 + 0.3 \times 10/7 = 10/7 \approx 1.429$;$Q(s_1,\text{move}) = 0 + 0.3 \times 20/7 = 6/7 \approx 0.857$;$Q(s_2,\text{stay}) = 20/7 \approx 2.857$;$Q(s_2,\text{move}) = 0.3 \times 10/7 = 3/7 \approx 0.429$。貪欲化:$\pi_1(s_1) = \text{stay}$,$\pi_1(s_2) = \text{stay}$。$\gamma=0.9$ では $s_1$ で move が選ばれたが、$\gamma=0.3$ では stay のまま。
> **7.3** $\pi_1 = \pi_0$ なので価値も同じで、改善は既に停止している。検算:$V(s_1) = \max(1 + 0.3 \cdot \frac{10}{7},\ 0.3 \cdot \frac{20}{7}) = \max(\frac{10}{7}, \frac{6}{7}) = \frac{10}{7}$ ✓。$V(s_2) = \max(2 + 0.3\cdot\frac{20}{7},\ 0.3\cdot\frac{10}{7}) = \max(\frac{20}{7}, \frac{3}{7}) = \frac{20}{7}$ ✓。
> **7.4** $\gamma = 0.9$:$\tau_{\text{eff}} \approx 10$ ステップ先まで見る。move の初期投資(報酬0の1ステップ)後、約10ステップ分の差額 $+1$/ステップを回収できるので得($0.9 \times 20 = 18 > 10$)。$\gamma = 0.3$:$\tau_{\text{eff}} \approx 1.4$ ステップしか見ない。投資回収前に割引で価値が消えるので、目先の $+1$ をとる stay が得($0.3 \times \frac{20}{7} = \frac{6}{7} < \frac{10}{7}$)。割引率は「どれだけ未来のために我慢できるか」を決めるパラメータである。

---

### 問題8 方策改善定理の展開を追う

> [!info] 使う道具
> - 仮定:すべての $s$ で $Q^\pi(s, \pi'(s)) \ge V^\pi(s)$($\pi, \pi'$ は決定論的方策)
> - (3.2):$Q^\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a]$
> - 参照:[[強化学習の数理_1_基礎編#第6章 動的計画法 — モデルが既知のときの厳密解法]] 6.2節

**問8.1** 展開の最初の2段
$$V^\pi(s) \le Q^\pi(s, \pi'(s)) = \mathbb{E}\big[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = \pi'(s)\big]$$
を書き、次に**期待値の中の $V^\pi(S_{t+1})$ に仮定をもう一度適用**して
$$\le \mathbb{E}_{\pi'}\big[R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) \mid S_t = s\big]$$
まで自分の手で導け(途中で(3.2)を $S_{t+1}$ に使う)。

**問8.2** この操作を $k$ 回繰り返した式の一般形を書き、残差項 $\gamma^k \mathbb{E}_{\pi'}[V^\pi(S_{t+k})]$ が $k \to \infty$ で消える理由を、問1.2の有界性を根拠に述べよ。極限が $V^{\pi'}(s)$ に一致することを確認し、$V^{\pi'}(s) \ge V^\pi(s)$ を結論せよ。

**問8.3** 貪欲方策 $\pi'(s) = \arg\max_a Q^\pi(s,a)$ が定理の仮定を自動的に満たすことを、(2.2)を使う1行の不等式で示せ(=**貪欲化は決して方策を悪化させない**)。

**問8.4** 改善が止まった($V^{\pi'} = V^\pi$)とき、成り立つ等式がBellman最適方程式(4.3)そのものであることを示し、「改善停止 ⟺ 最適到達」を結論せよ。問7.3の停止判定は、まさにこの事実を使っていたことを確認せよ。

> [!success]- 解答
> **8.1** 期待値内の $V^\pi(S_{t+1}) \le Q^\pi(S_{t+1}, \pi'(S_{t+1}))$(仮定)とし、これに(3.2)を適用すると $Q^\pi(S_{t+1}, \pi'(S_{t+1})) = \mathbb{E}[R_{t+2} + \gamma V^\pi(S_{t+2}) \mid S_{t+1}, A_{t+1} = \pi'(S_{t+1})]$。外側の期待値と合成すれば与式(行動が2ステップとも $\pi'$ で選ばれているので $\mathbb{E}_{\pi'}$ と書ける)。
> **8.2** $V^\pi(s) \le \mathbb{E}_{\pi'}[\sum_{j=1}^{k}\gamma^{j-1}R_{t+j} + \gamma^k V^\pi(S_{t+k}) \mid S_t = s]$。$|V^\pi| \le R_{\max}/(1-\gamma)$ で有界だから残差は $\gamma^k \cdot \text{有界} \to 0$。残るのは $\pi'$ のもとでの割引報酬和の期待値 $= V^{\pi'}(s)$。
> **8.3** $Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s,a) \ge \sum_a \pi(a|s) Q^\pi(s,a) = V^\pi(s)$。
> **8.4** 改善停止時、$V^\pi(s) = \max_a Q^\pi(s,a) = \max_a[R(s,a) + \gamma\sum_{s'}P V^\pi(s')]$。これは(4.3)なので $V^\pi = V^*$(問題6の一意性より)。問7.3では「貪欲化しても方策が変わらない」=この等式の成立を確認していた。

---

### 問題9 サンプルによる学習:MC・IS・TD・Q学習

> [!info] 使う道具
> - 逐次平均:$N$ 個目のサンプル $G$ を得たときの平均の更新
> - 重点サンプリング比:$\rho_{t:T-1} = \dfrac{\Pr_\pi(\tau)}{\Pr_b(\tau)}$,軌道の生起確率 $\Pr_\pi(\tau) = \prod_k \pi(a_k|s_k) P(s_{k+1}|s_k,a_k)$
> - TD誤差:$\delta_t = R_{t+1} + \gamma\hat V(S_{t+1}) - \hat V(S_t)$
> - SARSA目標:$R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$;Q学習目標:$R_{t+1} + \gamma\max_{a'}Q(S_{t+1},a')$
> - 参照:[[強化学習の数理_1_基礎編#第7章 モンテカルロ法 — モデルフリー学習への第一歩]]、[[強化学習の数理_1_基礎編#第8章 時間差分 (TD) 学習 — SARSAとQ学習]]

**問9.1**(逐次平均の導出)$\hat V_N = \frac{1}{N}\sum_{i=1}^N G^{(i)}$ とする。$\hat V_N = \hat V_{N-1} + \frac{1}{N}(G^{(N)} - \hat V_{N-1})$ を代数計算で示せ。「(目標 − 現在値)× ステップサイズ」というこの形が、以後のすべての学習則の原型である。

**問9.2**(ISで $P$ が消える)重点サンプリング比の定義に軌道の生起確率の積の式を代入し、遷移確率 $P$ が分子分母で相殺して
$$\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(a_k|s_k)}{b(a_k|s_k)}$$
となることを確かめよ。この相殺がなければモデルフリーの方策オフ学習は成立しない。

**問9.3**(ISの分散爆発)各ステップで $\pi(a_k|s_k)/b(a_k|s_k)$ が確率 $1/2$ で $2$、確率 $1/2$ で $0$ の値をとる(挙動方策が半分の確率で目標方策の選ばない行動をとる)としよう。$T - t = n$ ステップの比 $\rho$ について $\mathbb{E}[\rho]$ と $\mathbb{E}[\rho^2]$ を計算し、分散が $n$ とともに指数的に増大することを示せ。

**問9.4**(TD誤差の期待値)$\hat V = V^\pi$(真値)のとき、Bellman期待方程式(3.1)を使って $\mathbb{E}_\pi[\delta_t \mid S_t = s] = 0$ を示せ。「TD(0)はBellman方程式の残差がゼロになるまで推定値を調整する」ことの意味を確認せよ。

**問9.5**(数値でSARSAとQ学習を比べる)現在の推定が $Q(s', a_1) = 3$, $Q(s', a_2) = 5$。遷移 $(s, a, r{=}1, s')$ を観測し、$\varepsilon$-貪欲方策が探索して $A_{t+1} = a_1$ を選んだとする。$\gamma = 0.9$、$\alpha = 0.1$、更新前 $Q(s,a) = 2$ とするとき、SARSA更新後とQ学習更新後の $Q(s,a)$ をそれぞれ計算せよ。両者の目標値の差が「実際にとった行動 vs 最良の行動」の差であることを確認せよ。

**問9.6**(過大評価バイアスの定量化)状態 $s'$ の2行動の真のQ値がともに $0$、推定には独立なノイズが乗り $Q(s', a_i) = \varepsilon_i$、$\varepsilon_i$ は確率 $1/2$ ずつで $\pm 1$ とする。$\mathbb{E}[\max(\varepsilon_1, \varepsilon_2)]$ を4通りの場合を列挙して計算し、$\max_a \mathbb{E}[Q] = 0$ と比較せよ。次に「選択」を $\varepsilon$ で、「評価」を独立なもう1組のノイズ $\varepsilon'$ で行うDouble推定 $\mathbb{E}[\varepsilon'_{\arg\max_i \varepsilon_i}]$ を計算し、バイアスが消えることを確かめよ。

> [!success]- 解答
> **9.1** $\hat V_N = \frac{1}{N}(G^{(N)} + (N-1)\hat V_{N-1}) = \hat V_{N-1} + \frac{1}{N}(G^{(N)} - \hat V_{N-1})$。
> **9.2** $\rho = \dfrac{\prod \pi(a_k|s_k) P(s_{k+1}|s_k,a_k)}{\prod b(a_k|s_k) P(s_{k+1}|s_k,a_k)}$。同じ軌道なので同じ $P$ 因子が両方に現れ、約分して方策比の積のみ残る。
> **9.3** 各因子は $\mathbb{E} = \frac12\cdot 2 = 1$(不偏)、$\mathbb{E}[(\cdot)^2] = \frac12 \cdot 4 = 2$。独立性より $\mathbb{E}[\rho] = 1$、$\mathbb{E}[\rho^2] = 2^n$、分散 $= 2^n - 1$。ステップ数に対して指数爆発。
> **9.4** $\mathbb{E}_\pi[\delta_t|s] = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1})|s] - V^\pi(s) = \sum_a\pi[R + \gamma\sum_{s'}P V^\pi(s')] - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0$(第2の等号が(3.1))。
> **9.5** SARSA目標 $= 1 + 0.9 \times Q(s',a_1) = 1 + 2.7 = 3.7$。更新:$2 + 0.1(3.7 - 2) = 2.17$。Q学習目標 $= 1 + 0.9\max(3,5) = 5.5$。更新:$2 + 0.1(5.5-2) = 2.35$。差は $\gamma\,(\max_a Q - Q(\cdot, A_{t+1})) = 0.9 \times 2 = 1.8$、すなわち探索行動をとった影響を受ける(SARSA)か受けない(Q学習)かの違い。
> **9.6** $(\varepsilon_1,\varepsilon_2) \in \{(1,1),(1,-1),(-1,1),(-1,-1)\}$ 各確率 $1/4$。$\max$ は $1,1,1,-1$ なので $\mathbb{E}[\max] = \frac{3-1}{4} = 0.5 > 0 = \max_a\mathbb{E}[Q]$。Double推定:選択された行動が何であれ、その行動の $\varepsilon'$ は選択と独立で平均0。よって $\mathbb{E} = 0$、バイアス消滅。✓

---

### 問題10 方策勾配定理とベースライン

> [!info] 使う道具
> - (2.2)と(3.2):$V^\pi(s) = \sum_a \pi(a|s)Q^\pi(s,a)$,$Q^\pi(s,a) = R(s,a) + \gamma\sum_{s'}P(s'|s,a)V^\pi(s')$
> - 積の微分則、規格化条件 $\sum_a \pi_{\boldsymbol\theta}(a|s) = 1$
> - log-derivative:$\nabla\pi = \pi\,\nabla\ln\pi$
> - softmax方策:$\pi_{\boldsymbol\theta}(a|s) = \dfrac{e^{h(s,a)}}{\sum_b e^{h(s,b)}}$,選好 $h(s,a) = \boldsymbol\theta^\top\boldsymbol\phi(s,a)$(線形)
> - 参照:[[強化学習の数理_1_基礎編#第10章 方策勾配法 — 方策を直接最適化する]]

**問10.1**($\nabla V^\pi$ の再帰式)$V^\pi(s) = \sum_a \pi(a|s)Q^\pi(s,a)$ の両辺を $\boldsymbol\theta$ で微分し(積の微分)、$\nabla Q^\pi$ に(3.2)を代入して($R, P$ は $\boldsymbol\theta$ 非依存であることに注意)、
$$\nabla V^\pi(s) = \boldsymbol g(s) + \gamma\sum_{s'}P^\pi(s'|s)\,\nabla V^\pi(s'),\qquad \boldsymbol g(s) := \sum_a \nabla\pi(a|s)\,Q^\pi(s,a)$$
を導け。この式が問題4のBellman期待方程式と同じ「ソース項+割引伝播」の構造をもつこと(だからNeumann級数で展開できること)を指摘せよ。

**問10.2**(log-derivative trick)恒等式 $\nabla\pi = \pi\nabla\ln\pi$ を確認し、$\boldsymbol g(s) = \sum_a \pi(a|s)\,\nabla\ln\pi(a|s)\,Q^\pi(s,a) = \mathbb{E}_{a\sim\pi}[\nabla\ln\pi(a|s)\,Q^\pi(s,a)]$ と期待値の形に書き換えよ。この書き換えの目的(サンプルで推定可能にする)を述べよ。

**問10.3**(ベースラインの不偏性)状態のみの関数 $b(s)$ について
$$\sum_a \pi(a|s)\,\nabla\ln\pi(a|s)\;b(s) = \boldsymbol 0$$
を、$b$ を和の外に出す → log-derivativeを逆に使う → 規格化条件を微分する、の3ステップで証明せよ。

**問10.4**(softmaxのスコア関数)線形softmax方策について
$$\nabla_{\boldsymbol\theta}\ln\pi_{\boldsymbol\theta}(a|s) = \boldsymbol\phi(s,a) - \sum_b \pi_{\boldsymbol\theta}(b|s)\,\boldsymbol\phi(s,b)$$
を導け($\ln\pi = h(s,a) - \ln\sum_b e^{h(s,b)}$ と分けて微分する)。「選んだ行動の特徴 − 方策平均の特徴」という形の意味を一言で述べよ。

**問10.5**(REINFORCEを1ステップ手計算)2本腕バンディット(状態1つ、$\gamma^t$ 因子なし)、特徴は one-hot($\boldsymbol\phi(a_1) = (1,0)^\top$, $\boldsymbol\phi(a_2) = (0,1)^\top$)、初期 $\boldsymbol\theta = (0,0)^\top$(⟹ $\pi = (0.5, 0.5)$)。行動 $a_1$ をとって収益 $G = 2$ を得た。$\alpha = 0.1$ として REINFORCE 更新 $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha\, G\,\nabla\ln\pi(a_1)$ を計算し、更新後の方策 $\pi(a_1)$ を求めよ($e^{0.1} \approx 1.105$)。良い結果につながった行動の確率が上がることを数値で確認せよ。

> [!success]- 解答
> **10.1** $\nabla V^\pi = \sum_a[\nabla\pi\,Q^\pi + \pi\nabla Q^\pi]$。$\nabla Q^\pi(s,a) = \gamma\sum_{s'}P(s'|s,a)\nabla V^\pi(s')$ を代入し、$\sum_a \pi(a|s)P(s'|s,a) = P^\pi(s'|s)$ でまとめれば与式。(3.4)の $\boldsymbol r^\pi$ が $\boldsymbol g$ に置き換わっただけの同型の線形方程式であり、解は $\nabla V^\pi = \sum_k \gamma^k (P^\pi)^k \boldsymbol g$。
> **10.2** $\nabla\ln\pi = \nabla\pi/\pi$ より明らか。$\pi$ が係数として現れる和は「$\pi$ からサンプルした $a$ での値の平均」で近似できる——実際に方策を走らせて得るデータがそのままモンテカルロ推定になる。
> **10.3** $b(s)\sum_a\pi\nabla\ln\pi = b(s)\sum_a\nabla\pi = b(s)\nabla\sum_a\pi(a|s) = b(s)\,\nabla 1 = \boldsymbol 0$。
> **10.4** $\nabla\ln\pi(a|s) = \nabla h(s,a) - \dfrac{\sum_b e^{h(s,b)}\nabla h(s,b)}{\sum_b e^{h(s,b)}} = \boldsymbol\phi(s,a) - \sum_b\pi(b|s)\boldsymbol\phi(s,b)$。更新は「実際に選んだ行動」を「平均的な行動」に対して相対的に押し上げる方向を向く。
> **10.5** $\nabla\ln\pi(a_1) = (1,0)^\top - (0.5, 0.5)^\top = (0.5, -0.5)^\top$。更新:$\boldsymbol\theta = 0.1 \times 2 \times (0.5,-0.5)^\top = (0.1, -0.1)^\top$。新方策:$\pi(a_1) = \dfrac{e^{0.1}}{e^{0.1}+e^{-0.1}} = \dfrac{1.105}{1.105 + 0.905} \approx 0.550$。$0.5 \to 0.55$ に上昇。✓

---

## 第II部 深層強化学習編([[強化学習の数理_2_深層学習編]] 対応)

---

### 問題11 semi-gradientとstop-gradient、DQNの目標

> [!info] 使う道具
> - 二乗損失:$\mathcal L(\boldsymbol\theta) = \frac12(y - \hat v(s;\boldsymbol\theta))^2$
> - TD目標:$y = r + \gamma\hat v(s';\boldsymbol\theta)$(素朴版)/ $y = r + \gamma\hat v(s';\bar{\boldsymbol\theta})$(stop-gradient版、$\bar{\boldsymbol\theta}$ には勾配を流さない)
> - 参照:[[強化学習の数理_1_基礎編#第9章 関数近似への入り口 — 表形式の限界を超える]] 9.2節、[[強化学習の数理_2_深層学習編#第2章 DQN — 深層Q学習と不安定性への最初の処方箋]] 1.3節・2.2節

**問11.1** $y$ を**定数として**扱ったときの $-\nabla_{\boldsymbol\theta}\mathcal L$ を計算し、semi-gradient TD(0)の更新則
$$\boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha\,[y - \hat v(s;\boldsymbol\theta)]\,\nabla_{\boldsymbol\theta}\hat v(s;\boldsymbol\theta)$$
と一致することを確かめよ。

**問11.2** 今度は $y = r + \gamma\hat v(s';\boldsymbol\theta)$ の $\boldsymbol\theta$ 依存性も**込めて**真の勾配 $-\nabla_{\boldsymbol\theta}\mathcal L$ を計算せよ。semi-gradientと比べて余分に現れる項を特定し、「semi-gradient は目標側の勾配 $\nabla\hat v(s';\boldsymbol\theta)$ を意図的に落としたもの」であることを式で確認せよ。

**問11.3** 線形近似 $\hat v(s;\boldsymbol w) = \boldsymbol w^\top\boldsymbol\phi(s)$、特徴 $\boldsymbol\phi(s) = (1, 0)^\top$、$\boldsymbol\phi(s') = (0.8, 0.2)^\top$、$\boldsymbol w = (1, 2)^\top$、遷移 $(s, r{=}0.5, s')$、$\gamma = 0.9$、$\alpha = 0.1$ とする。TD誤差 $\delta$ と semi-gradient 更新後の $\boldsymbol w$ を計算せよ。one-hot特徴なら表形式TD(0)に帰着することも、$\boldsymbol\phi(s) = \boldsymbol e_s$ を代入して確認せよ。

**問11.4** DQNのターゲットネットワークは「$\bar{\boldsymbol\theta} = \boldsymbol\theta^-$ を $C$ ステップ凍結する」stop-gradientの強化版である。凍結期間中に解いている問題が「固定した目標 $T^*Q_{\boldsymbol\theta^-}$ への回帰」、すなわち [[強化学習の数理_1_基礎編]] 第6章の反復的方策評価(価値反復)の外側ループの復元であることを、更新式を書き並べて対応づけよ。

> [!success]- 解答
> **11.1** $-\nabla\mathcal L = (y - \hat v(s;\boldsymbol\theta))\nabla_{\boldsymbol\theta}\hat v(s;\boldsymbol\theta)$。一致。
> **11.2** $-\nabla\mathcal L = (y - \hat v(s))[\nabla\hat v(s) - \gamma\nabla\hat v(s')]$。余分な項は $-\gamma(y-\hat v(s))\nabla\hat v(s';\boldsymbol\theta)$。semi-gradient はこれをゼロと置いた(目標を教師信号として固定した)ものである。
> **11.3** $\hat v(s) = 1$,$\hat v(s') = 0.8 + 0.4 = 1.2$。$\delta = 0.5 + 0.9 \times 1.2 - 1 = 0.58$。更新:$\boldsymbol w \leftarrow (1,2)^\top + 0.1 \times 0.58 \times (1,0)^\top = (1.058, 2)^\top$。one-hotなら $\nabla\hat v(s) = \boldsymbol e_s$ で更新は成分 $s$ のみに働き、表のセル更新 $\hat V(s) \leftarrow \hat V(s) + \alpha\delta$ と同一。
> **11.4** DP:$v_{k+1} = T^* v_k$(古い $v_k$ を固定して計算)。DQN:凍結期間 $k$ の間、目標 $y = (T^*Q_{\boldsymbol\theta^-_k})(s,a)$ のサンプル版へ $Q_{\boldsymbol\theta}$ を回帰 ⟹ 期間終了時 $Q_{\boldsymbol\theta^-_{k+1}} \approx T^* Q_{\boldsymbol\theta^-_k}$。凍結インデックスが反復インデックス $k$ の役割を果たす(Fitted Q-Iteration描像)。

---

### 問題12 Double DQNとDueling:バイアスの解剖とゲージ固定

> [!info] 使う道具
> - Jensen型不等式:$\max$ は凸関数なので $\mathbb{E}[\max_i X_i] \ge \max_i \mathbb{E}[X_i]$
> - Double DQN目標:$y = r + \gamma\,Q_{\boldsymbol\theta^-}(s', \arg\max_{a'}Q_{\boldsymbol\theta}(s',a'))$
> - Dueling合成:$Q(s,a) = V(s) + \big(A(s,a) - \frac{1}{|\mathcal A|}\sum_{a'}A(s,a')\big)$
> - 問2.2の結果(アドバンテージの平均ゼロ性)
> - 参照:[[強化学習の数理_2_深層学習編#第3章 DQNの系譜 — Rainbowに至る改良の解剖]] 3.1–3.2節

**問12.1** 真値がすべて0、推定ノイズ $\varepsilon_{a'}$ が独立に標準正規分布 $\mathcal N(0, \sigma^2)$ に従う2行動の場合、$\mathbb{E}[\max(\varepsilon_1, \varepsilon_2)] = \dfrac{\sigma}{\sqrt\pi} \approx 0.564\sigma$ が知られている(導出は不要、値として使ってよい)。ブートストラップでこの正のバイアスが**次の目標値に伝播して累積する**機構を、$Q$ 学習の更新式の目標値の中に $\max$ が入っていることから説明せよ。

**問12.2** $\max_{a'}Q(s',a') = Q(s', \arg\max_{a'}Q(s',a'))$ と「選択」「評価」の合成に分解し、DQN(同一ネットで両方)とDouble DQN(選択=オンライン $\boldsymbol\theta$、評価=ターゲット $\boldsymbol\theta^-$)の目標値の式を並べて書け。問9.6の計算結果を引用して、なぜ分離でバイアスが消える(減る)のかを述べよ。

**問12.3**(Duelingの同定不能性)分解 $Q = V + A$ には、任意の定数 $c$ について $(V, A) \to (V + c,\ A - c)$ としても $Q$ が不変という自由度(ゲージ自由度)がある。これを式で確認し、「学習で $V$ ヘッドと $A$ ヘッドの役割が定まらない」という問題になる理由を述べよ。

**問12.4** Dueling合成式の括弧内の平均差し引きが、このゲージを「$\sum_{a'} A(s,a') = 0$」に固定する操作であることを示せ:合成後の $Q$ から逆算される実効的な $A$ の平均がゼロになることを計算で確かめよ。問2.2で示した $\sum_a \pi(a|s)A^\pi(s,a) = 0$ の一様方策版がここで規約として再利用されていることを指摘せよ。

> [!success]- 解答
> **12.1** 更新目標 $r + \gamma\max_{a'}Q(s',a')$ の第2項が平均 $+0.564\gamma\sigma$ だけ過大 → $Q(s,a)$ が過大に学習される → その $Q(s,a)$ がさらに前の状態の目標値の $\max$ の中に入り、過大分が $\gamma$ 割引されつつ何段も遡って積み上がる。
> **12.2** DQN:$y = r + \gamma Q_{\boldsymbol\theta^-}(s', \arg\max_{a'}Q_{\boldsymbol\theta^-}(s',a'))$(選択も評価も $\boldsymbol\theta^-$)。DDQN:$y = r + \gamma Q_{\boldsymbol\theta^-}(s', \arg\max_{a'}Q_{\boldsymbol\theta}(s',a'))$。問9.6の通り、選択に使うノイズと評価に使うノイズが独立(または相関が弱い)なら、選ばれた行動の評価側ノイズは条件付きでも平均0に近く、$\mathbb E[\max]$ 型の正の偏りが消える。
> **12.3** $(V+c) + (A-c) = V + A = Q$。$Q$ への教師信号だけでは $c$ が決まらないので、$V$ ヘッドが状態価値を、$A$ ヘッドが相対的な差を表す、という意図した分業が学習では強制されない。
> **12.4** 合成後 $Q(s,a) - V(s) = A(s,a) - \bar A(s)$($\bar A$ は行動平均)。この量の行動平均は $\bar A - \bar A = 0$。すなわち出力側では常に「平均ゼロのアドバンテージ+状態価値」という一意な分解が実現され、ゲージが固定される。$\pi$ を一様分布に置き換えた平均ゼロ規約である。

---

### 問題13 優先度付き経験再生のIS補正

> [!info] 使う道具
> - 重点サンプリングの原理(問9.2):分布 $q$ でサンプルして分布 $p$ の期待値を求めるには重み $p/q$ を掛ける
> - PERのサンプリング確率:$p_i \propto (|\delta_i| + \epsilon)^\alpha$、補正重み $w_i = \left(\frac{1}{N}\cdot\frac{1}{p_i}\right)^\beta$
> - 参照:[[強化学習の数理_2_深層学習編#第3章 DQNの系譜 — Rainbowに至る改良の解剖]] 3.3節

**問13.1** 意図した学習は「バッファの一様分布 $p_{\text{unif}}(i) = 1/N$ での期待損失の最小化」である。実際のサンプリング分布が $p_i$ のとき、不偏性を回復する重みが $w_i \propto \dfrac{1/N}{p_i}$ となることを、重点サンプリングの原理から導け($\beta = 1$ の場合)。

**問13.2** バッファサイズ $N = 4$、TD誤差絶対値 $(|\delta_1|, \dots, |\delta_4|) = (3, 1, 1, 1)$、$\epsilon = 0$、$\alpha = 1$ とする。各サンプルの選択確率 $p_i$ と、$\beta = 1$ での補正重み $w_i$(最大値で正規化:$w_i / \max_j w_j$)を計算せよ。「よく選ばれるサンプルほど1回あたりの更新を弱める」ことを数値で確認せよ。

**問13.3** $\alpha = 0$ と $\beta$ の役割の違いを整理せよ:$\alpha = 0$ のとき $p_i$ と $w_i$ はどうなるか。学習初期に $\beta < 1$(不完全補正)を許し終盤に $\beta \to 1$ とアニールする設計の意図を、「バイアス」と「学習速度」のトレードオフとして述べよ。

> [!success]- 解答
> **13.1** $\mathbb{E}_{i\sim p}\left[\frac{p_{\text{unif}}(i)}{p_i}\ell_i\right] = \sum_i p_i \frac{1/N}{p_i}\ell_i = \frac1N\sum_i \ell_i = \mathbb{E}_{\text{unif}}[\ell]$。よって重み $\frac{1/N}{p_i}$ で不偏。
> **13.2** $\sum|\delta| = 6$ より $p = (1/2, 1/6, 1/6, 1/6)$。$w_i \propto 1/p_i = (2, 6, 6, 6)$(× $1/N$ は共通因子)。正規化して $w = (1/3, 1, 1, 1)$。優先度3倍で選ばれるサンプル1は、1回の更新の重みが $1/3$ に抑えられる。
> **13.3** $\alpha = 0$ なら $p_i = 1/N$(一様)で $w_i = 1$(補正不要)、通常のDQNに帰着。初期は推定がどうせ粗いのでバイアスを許して「学べるサンプル」を集中的に使い速度を稼ぎ、収束の質が問題になる終盤に $\beta \to 1$ で不偏性を回復する。

---

### 問題14 GAE:望遠鏡和と幾何混合を計算し切る

> [!info] 使う道具
> - TD誤差:$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$
> - $k$ステップアドバンテージ:$\hat A_t^{(k)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{k-1}r_{t+k} + \gamma^k V(s_{t+k}) - V(s_t)$
> - GAE定義:$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{k=1}^\infty \lambda^{k-1}\hat A_t^{(k)}$
> - 幾何級数:$\sum_{k=l+1}^\infty \lambda^{k-1} = \dfrac{\lambda^l}{1-\lambda}$
> - 参照:[[強化学習の数理_2_深層学習編#第4章 深層方策勾配 (I) — A2C/A3C と一般化アドバンテージ推定 (GAE)]] 4.3節

**問14.1**(望遠鏡和)$\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$ に $\delta$ の定義を代入して展開し、$V$ を含む項が隣接項間で打ち消し合って $\gamma^k V(s_{t+k}) - V(s_t)$ だけが残ることを、$k = 2$ の場合について全項を書き下して確認せよ。その上で一般の $k$ について
$$\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\,\delta_{t+l}$$
を結論せよ。

**問14.2**(和の順序交換)GAEの定義に問14.1を代入し、$l$ を固定して $k \ge l+1$ の寄与を集める順序交換を実行し、幾何級数の公式を使って閉形式
$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l\,\delta_{t+l}$$
を導け。

**問14.3**(両極限)閉形式に $\lambda = 0$ を代入すると何になるか。$\lambda = 1$ のときは望遠鏡和(問14.1の $k \to \infty$ 版)を使って $G_t - V(s_t)$(MCベースライン付き)に一致することを示せ。それぞれのバイアス・分散の性格を一言で述べよ。

**問14.4**(数値計算)長さ3のエピソードで $\delta_0 = 1$, $\delta_1 = -0.5$, $\delta_2 = 2$(それ以降は0)、$\gamma = 0.9$, $\lambda = 0.5$ とする。
(a) 閉形式で $\hat A_0, \hat A_1, \hat A_2$ を計算せよ。
(b) 後ろ向き再帰 $\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}$($\hat A_3 = 0$)で同じ値が出ることを確認せよ。この再帰が $O(T)$ 実装の根拠である。

> [!success]- 解答
> **14.1** $k=2$:$\delta_t + \gamma\delta_{t+1} = [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)] + \gamma[r_{t+2} + \gamma V(s_{t+2}) - V(s_{t+1})] = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) - V(s_t) = \hat A_t^{(2)}$($\gamma V(s_{t+1})$ が相殺)。一般には $V$ 項が $\sum_l[\gamma^{l+1}V(s_{t+l+1}) - \gamma^l V(s_{t+l})]$ の望遠鏡和になり端点のみ残る。
> **14.2** $(1-\lambda)\sum_{k\ge1}\lambda^{k-1}\sum_{l=0}^{k-1}\gamma^l\delta_{t+l} = \sum_{l\ge0}\gamma^l\delta_{t+l}\,(1-\lambda)\sum_{k\ge l+1}\lambda^{k-1} = \sum_l \gamma^l\delta_{t+l}\cdot\lambda^l$。
> **14.3** $\lambda=0$:$\hat A_t = \delta_t$(1ステップTD、低分散・高バイアス)。$\lambda=1$:$\sum_l\gamma^l\delta_{t+l} = \lim_k \hat A_t^{(k)} = G_t - V(s_t)$($\gamma^kV \to 0$)、不偏・高分散。
> **14.4** (a) $\gamma\lambda = 0.45$。$\hat A_0 = 1 + 0.45(-0.5) + 0.45^2(2) = 1 - 0.225 + 0.405 = 1.18$。$\hat A_1 = -0.5 + 0.45 \times 2 = 0.4$。$\hat A_2 = 2$。(b) $\hat A_2 = 2$;$\hat A_1 = -0.5 + 0.45 \times 2 = 0.4$;$\hat A_0 = 1 + 0.45 \times 0.4 = 1.18$。一致。✓

---

### 問題15 性能差分補題と代理目的関数

> [!info] 使う道具
> - $A^\pi(s_t,a_t) = \mathbb{E}_{s_{t+1}}[r_{t+1} + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)]$((3.2)とアドバンテージ定義から;遷移は方策によらず $P$ に従う)
> - $J(\pi) = \mathbb{E}_{s_0\sim\rho_0}[V^\pi(s_0)]$
> - 望遠鏡和のテクニック(問14.1で習得済み)
> - log-derivative trick、方策勾配定理(問10)
> - 参照:[[強化学習の数理_2_深層学習編#第5章 深層方策勾配 (II) — 性能差分補題・自然勾配・TRPO・PPO]] 5.2–5.3節

**問15.1** $\mathbb{E}_{\tau\sim\pi'}\left[\sum_{t=0}^\infty \gamma^t A^\pi(s_t, a_t)\right]$ に上の $A^\pi$ の表式を代入し、和を「報酬の和」と「$V^\pi$ の望遠鏡和」に分けよ。後者が $-V^\pi(s_0)$ に潰れることを示し(残差消滅の根拠も述べる)、**性能差分補題**
$$J(\pi') - J(\pi) = \mathbb{E}_{\tau\sim\pi'}\left[\sum_t \gamma^t A^\pi(s_t,a_t)\right]$$
を完成させよ。

**問15.2** この補題から、[[強化学習の数理_1_基礎編]] 第6章の方策改善定理(すべての $s$ で $A^\pi(s, \pi'(s)) \ge 0$ なら $\pi'$ は改善)が**系として**従うことを説明せよ(右辺の被積分関数の符号を見る)。

**問15.3** 補題の右辺は新方策 $\pi'$ の訪問分布 $\bar d^{\pi'}$ を要するため直接使えない。状態分布を旧方策のもの $\bar d^\pi$ で置き換え、行動分布のずれを確率比 $\frac{\pi'(a|s)}{\pi(a|s)}$ で補正した**代理目的関数** $L_\pi(\pi')$ の定義を書け。行動側の補正が厳密であるのに状態側が近似である理由を、問9.2(ISでは分布のずれを比で補正する)を引きつつ述べよ。

**問15.4**(1次の一致)$L_\pi(\pi_{\boldsymbol\theta'})$ を $\boldsymbol\theta'$ で微分して $\boldsymbol\theta' = \boldsymbol\theta$ で評価すると、log-derivative trick により方策勾配定理の表式(問10.2)に一致することを示せ。すなわち $L_\pi$ は $J$ の1次近似として正しく、「$\pi'$ が $\pi$ の近くにいる限り信用できる」——これがTRPO・PPOの「歩幅制限」の根拠である。

> [!success]- 解答
> **15.1** 代入すると $\mathbb{E}_{\tau\sim\pi'}[\sum_t\gamma^t r_{t+1}] + \mathbb{E}[\sum_t(\gamma^{t+1}V^\pi(s_{t+1}) - \gamma^t V^\pi(s_t))]$。第1項は $J(\pi')$。第2項は望遠鏡和で $\lim_T \gamma^T V^\pi(s_T) - V^\pi(s_0) = -V^\pi(s_0)$($V^\pi$ 有界、問1.2)。$s_0\sim\rho_0$ で平均して $-J(\pi)$。
> **15.2** 各 $(s_t,a_t)$ で被積分関数 $A^\pi \ge 0$ なら右辺 $\ge 0$、よって $J(\pi') \ge J(\pi)$。補題は改善定理の**等式版・定量版**である。
> **15.3** $L_\pi(\pi') = J(\pi) + \frac{1}{1-\gamma}\mathbb{E}_{s\sim\bar d^\pi, a\sim\pi}\left[\frac{\pi'(a|s)}{\pi(a|s)}A^\pi(s,a)\right]$。行動は状態を固定すれば分布が既知($\pi$ と $\pi'$)なのでISの比で厳密補正できるが、状態分布 $\bar d^{\pi'}$ は $\pi'$ で軌道を生成しないと得られない(遷移カーネルの多段合成)ため、旧分布で代用する近似が入る。
> **15.4** $\nabla_{\boldsymbol\theta'}L\big|_{\boldsymbol\theta'=\boldsymbol\theta} = \frac{1}{1-\gamma}\mathbb{E}\left[\frac{\nabla\pi_{\boldsymbol\theta}(a|s)}{\pi_{\boldsymbol\theta}(a|s)}A^\pi\right] = \frac{1}{1-\gamma}\mathbb{E}_{s\sim\bar d^\pi, a\sim\pi}[\nabla\ln\pi\; A^\pi]$。これは方策勾配定理($\Psi = A^\pi$ 版)に一致。

---

### 問題16 PPOのクリップ目的関数を場合分けで読み切る

> [!info] 使う道具
> - $L^{\mathrm{CLIP}} = \hat{\mathbb E}_t\left[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\right]$,$\rho_t = \dfrac{\pi_{\boldsymbol\theta'}(a_t|s_t)}{\pi_{\boldsymbol\theta}(a_t|s_t)}$
> - $\mathrm{clip}(x, l, u)$:$x$ を $[l, u]$ に切り詰める
> - 参照:[[強化学習の数理_2_深層学習編#第5章 深層方策勾配 (II) — 性能差分補題・自然勾配・TRPO・PPO]] 5.6節

$\epsilon = 0.2$ とする。以下の4ケースそれぞれについて、(i) $\min$ の中の2項の値、(ii) $L^{\mathrm{CLIP}}$ の値、(iii) $\rho$ に関する勾配が生きているか消えているか、を求めよ。

**問16.1** $\hat A = +1$, $\rho = 1.5$(良い行動の確率を上げすぎた)
**問16.2** $\hat A = +1$, $\rho = 0.7$(良い行動なのに確率が下がってしまった)
**問16.3** $\hat A = -1$, $\rho = 0.7$(悪い行動の確率を下げすぎた)
**問16.4** $\hat A = -1$, $\rho = 1.5$(悪い行動なのに確率が上がってしまった)

**問16.5** 4ケースの結果を表にまとめ、次の2点を結論せよ:(a) クリップは「目的関数を**改善する方向**の更新だけを頭打ちにする」非対称な仕掛けである。(b) したがって $L^{\mathrm{CLIP}} \le \rho\hat A$(素の代理目的の**下界**=悲観的近似)である。信頼領域の思想([[強化学習の数理_2_深層学習編]] 5.3節のKL下界)が、制約なしの1次最適化にどう翻訳されたかを一言で述べよ。

> [!success]- 解答
> **16.1** 非クリップ項 $1.5$、クリップ項 $1.2 \times 1 = 1.2$。$\min = 1.2$(クリップ側)。$\rho$ を動かしても値が変わらない → **勾配消滅**。「これ以上確率を上げるインセンティブを与えない」。
> **16.2** 非クリップ $0.7$、クリップ $0.8$。$\min = 0.7$(非クリップ側)→ **勾配は生きている**。良い行動の確率を戻す(上げる)方向の学習は妨げない。
> **16.3** 非クリップ $-0.7$、クリップ $-0.8$。$\min = -0.8$(クリップ側)→ **勾配消滅**。「悪い行動でも一度に潰しすぎない」。
> **16.4** 非クリップ $-1.5$、クリップ $-1.2$。$\min = -1.5$(非クリップ側)→ **勾配は生きている**。悪い行動の確率が上がった失敗はいつでも訂正できる。
> **16.5**
>
> | ケース | 更新の向き | クリップ |
> | ---- | ---- | ---- |
> | 16.1 $A>0, \rho$ 大 | 改善方向に行きすぎ | 発動(頭打ち) |
> | 16.2 $A>0, \rho$ 小 | 悪化を戻す | 非発動 |
> | 16.3 $A<0, \rho$ 小 | 改善方向に行きすぎ | 発動(頭打ち) |
> | 16.4 $A<0, \rho$ 大 | 悪化を戻す | 非発動 |
>
> (a) 発動するのは改善方向の行きすぎのみ。(b) $\min$ をとる以上つねに素の項以下、すなわち下界。「KL制約で歩幅を縛る」代わりに「歩幅を伸ばしても得をしない目的関数を作る」ことで、同じ保守性を通常のSGDだけで実現した。

---

### 問題17 Fisher情報行列と自然勾配

> [!info] 使う道具
> - スコアの平均ゼロ:$\mathbb{E}_{p_{\boldsymbol\theta}}[\nabla\ln p_{\boldsymbol\theta}] = \boldsymbol 0$(問10.3と同じ規格化条件の微分)
> - Fisher情報行列:$F(\boldsymbol\theta) = \mathbb{E}[\nabla\ln p\,\nabla\ln p^\top] = -\mathbb{E}[\nabla^2\ln p]$
> - KLの2次展開:$D_{\mathrm{KL}}(p_{\boldsymbol\theta}\|p_{\boldsymbol\theta+\Delta}) = \frac12\Delta^\top F\Delta + O(\Delta^3)$
> - Lagrange未定乗数法
> - 参照:[[強化学習の数理_2_深層学習編#付録A Fisher情報行列と自然勾配の導出]]

**問17.1**(1次元で全部やる)$p_\theta = $ Bernoulli分布:$p_\theta(1) = \theta$, $p_\theta(0) = 1-\theta$。
(a) $\ln p_\theta(x)$ と $\partial_\theta \ln p_\theta(x)$ を $x = 0, 1$ それぞれについて書け。
(b) スコアの平均がゼロであることを直接計算で確認せよ。
(c) $F(\theta) = \mathbb{E}[(\partial_\theta\ln p)^2] = \dfrac{1}{\theta(1-\theta)}$ を計算せよ。$\theta$ が0や1に近いほど $F$ が大きい(=同じ $\Delta\theta$ でも分布が激しく動く)ことの直観を述べよ。

**問17.2** 平均 $\mu$ のみをパラメータとするガウス分布 $p_\mu = \mathcal N(x; \mu, \sigma^2)$($\sigma$ 固定)について $F(\mu) = 1/\sigma^2$ を計算せよ。自然勾配 $\Delta \propto F^{-1}\nabla J = \sigma^2\nabla J$ が「分散が小さい(=分布が尖っている)ほど歩幅を小さくする」補正であることを解釈せよ。ガウス方策の学習後期(分散が縮む)に通常勾配が危険になる、という [[強化学習の数理_2_深層学習編]] 5.4節の議論との対応を述べよ。

**問17.3** 制約付き最大化 $\max_\Delta \nabla J^\top\Delta$ s.t. $\frac12\Delta^\top F\Delta \le \epsilon$ をLagrange未定乗数法で解き、最適方向が $\Delta \propto F^{-1}\nabla J$ であることを導け(停留条件 $\nabla J = \lambda F\Delta$ を書くだけでよい)。

> [!success]- 解答
> **17.1** (a) $\ln p = x\ln\theta + (1-x)\ln(1-\theta)$、$\partial_\theta\ln p = \frac{x}{\theta} - \frac{1-x}{1-\theta}$。$x=1$ で $1/\theta$、$x=0$ で $-1/(1-\theta)$。(b) $\theta\cdot\frac1\theta + (1-\theta)\cdot(-\frac{1}{1-\theta}) = 1 - 1 = 0$。✓ (c) $F = \theta\cdot\frac{1}{\theta^2} + (1-\theta)\cdot\frac{1}{(1-\theta)^2} = \frac1\theta + \frac{1}{1-\theta} = \frac{1}{\theta(1-\theta)}$。端に近い分布はパラメータの微小変化で(相対的に)大きく変形する:確率0.01→0.02は「倍増」だが0.50→0.51はほぼ不変。
> **17.2** $\ln p = -\frac{(x-\mu)^2}{2\sigma^2} + \text{const}$、$\partial_\mu\ln p = \frac{x-\mu}{\sigma^2}$。$F = \mathbb{E}[(x-\mu)^2]/\sigma^4 = 1/\sigma^2$。自然勾配は歩幅を $\sigma^2$ 倍する:分散が小さいほどKLの意味で同じ距離を保つためにパラメータの歩幅を縮める。方策の分散が縮んだ学習後期に固定学習率の通常勾配を使うと、分布としては巨大な一歩になり性能崩壊(5.1節の自己増悪ループ)を招く——その処方箋が計量 $F$ での補正である。
> **17.3** $\mathcal L = \nabla J^\top\Delta - \lambda(\frac12\Delta^\top F\Delta - \epsilon)$、$\partial_\Delta\mathcal L = \nabla J - \lambda F\Delta = 0$ ⟹ $\Delta = \frac1\lambda F^{-1}\nabla J \propto F^{-1}\nabla J$。

---

### 問題18 決定論的方策勾配(DPG)の連鎖律

> [!info] 使う道具
> - 決定論的方策:$a = \mu_{\boldsymbol\theta}(s)$、このとき $V^\mu(s) = Q^\mu(s, \mu_{\boldsymbol\theta}(s))$
> - 多変数の連鎖律:$\nabla_{\boldsymbol\theta}\,Q(s, \mu_{\boldsymbol\theta}(s))$ は「$\mu$ を通じた依存」と「$Q^\mu$ 自身のパラメータ依存」の2経路をもつ
> - 参照:[[強化学習の数理_2_深層学習編#第6章 連続行動制御 — 決定論的方策勾配・DDPG・TD3]] 6.2節

**問18.1** $V^\mu(s) = Q^\mu(s, \mu_{\boldsymbol\theta}(s))$ を $\boldsymbol\theta$ で微分し、2つの項
$$\nabla_{\boldsymbol\theta}V^\mu(s) = \underbrace{\nabla_{\boldsymbol\theta}\mu_{\boldsymbol\theta}(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu(s)}}_{\text{(A)}} + \underbrace{\nabla_{\boldsymbol\theta}Q^\mu(s,a)\big|_{a=\mu(s)}}_{\text{(B)}}$$
に分解せよ。(A)(B)それぞれが「何の変化による寄与」かを言葉で述べよ。

**問18.2** (B)に $Q^\mu(s,a) = R(s,a) + \gamma\int P(s'|s,a)V^\mu(s')ds'$ を代入し($R, P$ は $\boldsymbol\theta$ 非依存)、問10.1と同型の再帰式「ソース項(A)+割引伝播」が得られることを確認せよ。これを展開した結果が**DPG定理**
$$\nabla_{\boldsymbol\theta}J = \mathbb{E}_{s\sim d^\mu}\left[\nabla_{\boldsymbol\theta}\mu_{\boldsymbol\theta}(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu_{\boldsymbol\theta}(s)}\right]$$
である(展開の詳細は問10.1と同じなので省略してよい)。

**問18.3** 確率的方策勾配(問10.2)と見比べて、(i) 行動空間上の期待値が消えている理由、(ii) スコア関数 $\nabla\ln\pi$ の代わりに $Q$ の**行動微分**が現れている意味(=クリティックの値でなく形状・勾配を使う)、(iii) その代償としてなぜ探索を外生ノイズで足す必要があるか、の3点を述べよ。

**問18.4**(1次元の手計算)$s$ 固定、$a = \mu_\theta = \theta$(方策パラメータがそのまま行動)、クリティックが $Q(a) = -(a - 3)^2$ と近似されているとする。DPG更新 $\theta \leftarrow \theta + \alpha\,\partial_a Q|_{a=\theta}$ を $\theta_0 = 0$, $\alpha = 0.25$ から2回実行し、$\theta$ が $Q$ の頂点 $a = 3$ に向かうことを確かめよ。もしクリティックの誤差で偽の頂点(たとえば $Q$ が $a = 10$ 付近に誤った山をもつ)があれば、アクターはそこへも同じ機構で登ってしまう——これがDDPGの「クリティック搾取」問題([[強化学習の数理_2_深層学習編]] 6.3節)の骨格であることを確認せよ。

> [!success]- 解答
> **18.1** (A) 方策が変わって**選ぶ行動そのものが変わる**直接効果。(B) 方策が変わると**将来の軌道が変わり $Q^\mu$ という関数自体が変わる**間接効果。
> **18.2** (B) $= \gamma\int P(s'|s,\mu(s))\nabla_{\boldsymbol\theta}V^\mu(s')ds'$。よって $\nabla V^\mu(s) = (\text{A項}) + \gamma\,\mathbb{E}_{s'}[\nabla V^\mu(s')]$、問10.1と同じ構造。展開すれば割引訪問分布 $d^\mu$ 上の(A)の平均、すなわちDPG定理。
> **18.3** (i) 方策が1点分布なので $a$ について平均する必要がない(サンプリングノイズが消え低分散)。(ii) 「その行動の確率を上げろ」ではなく「$Q$ が増える方向へ行動を連続的にずらせ」という勾配情報を直接使う。(iii) 方策自身がランダム性をもたないため、放っておくと同じ行動しか試さず探索が死ぬ。ゆえに $a = \mu(s) + \text{ノイズ}$ とする(=挙動方策と目標方策が分離し、本質的に方策オフ)。
> **18.4** $\partial_a Q = -2(a-3)$。$\theta_1 = 0 + 0.25 \times 6 = 1.5$。$\theta_2 = 1.5 + 0.25 \times 3 = 2.25$。頂点3へ単調接近。アクターは「$Q$ の勾配が指す山」を登るだけなので、山が近似誤差による偽物でも区別できず登る。TD3の $\min$ 二重クリティック・目標平滑化はこの搾取への処方箋である。

---

### 問題19 最大エントロピーRL:Boltzmann方策とソフトBellman方程式

> [!info] 使う道具
> - ソフト価値の関係式:$V_{\mathrm{soft}}(s) = \mathbb{E}_{a\sim\pi}[Q_{\mathrm{soft}}(s,a) - \alpha\ln\pi(a|s)]$
> - 制約付き変分問題のLagrange法(制約:$\sum_a\pi_a = 1$)
> - log-sum-exp:$\mathrm{lse}_\alpha(Q) := \alpha\ln\sum_a e^{Q_a/\alpha}$
> - 参照:[[強化学習の数理_2_深層学習編#第7章 最大エントロピー強化学習 — ソフトBellman方程式とSAC]] 7.1–7.2節

**問19.1** $Q_{\mathrm{soft}}$ を固定し、$\max_{\pi(\cdot|s)}\sum_a\pi_a(Q_a - \alpha\ln\pi_a)$ を制約 $\sum_a\pi_a = 1$ のもとで解く。Lagrange関数を書き、$\pi_a$ で微分してゼロと置き、
$$\pi^*_a = \frac{e^{Q_a/\alpha}}{Z},\qquad Z = \sum_a e^{Q_a/\alpha}$$
を導け(未定乗数 $\lambda$ は規格化で消えることを確認する)。

**問19.2** $\pi^*$ を目的関数に代入し($\ln\pi^*_a = Q_a/\alpha - \ln Z$ を使う)、最大値が
$$V_{\mathrm{soft}}(s) = \alpha\ln Z = \alpha\ln\sum_a e^{Q_a/\alpha}$$
であることを計算で示せ。

**問19.3**(数値例と零温度極限)2行動、$Q = (2, 0)$ とする。
(a) $\alpha = 1$ のとき $\pi^*$ と $V_{\mathrm{soft}}$ を計算せよ($e^2 \approx 7.389$)。$V_{\mathrm{soft}} > \max_a Q_a = 2$ となること、超過分がエントロピーボーナスであることを確認せよ。
(b) $\alpha = 0.1$ のとき $V_{\mathrm{soft}}$ はほぼいくらか($e^{20}$ は巨大なので $\alpha\ln(e^{20} + 1) \approx \alpha \times 20$ と評価してよい)。$\alpha \to 0$ で $\mathrm{lse}_\alpha \to \max$、方策が貪欲(基底状態)に凝縮することを確認せよ。

**問19.4**(縮小性の存続)log-sum-expの非拡大性
$$\big|\mathrm{lse}_\alpha(x) - \mathrm{lse}_\alpha(y)\big| \le \max_a|x_a - y_a|$$
を認めると、ソフトBellman最適作用素の $\gamma$-縮小性の証明は、[[強化学習の数理_1_基礎編]] 第5章の $T^*$ の証明(問6.3)の補題5.3をこの非拡大性に差し替えるだけで完了する。証明のどのステップがどう置き換わるかを対応表として書け(前作の理論が「$\max$ → lse」の置換で丸ごと再利用できることを実感するのが目的)。

**問19.5**(統計力学対応)対応表 $Q \leftrightarrow -E$、$\alpha \leftrightarrow k_BT$、$V_{\mathrm{soft}} = \alpha\ln Z \leftrightarrow -F = k_BT\ln Z$ のもとで、問19.2で使った関係式 $V = \mathbb{E}_\pi[Q] + \alpha\mathcal H(\pi)$ が熱力学関係式 $F = \langle E\rangle - TS$ に対応することを、符号に注意して確認せよ。

> [!success]- 解答
> **19.1** $\mathcal F = \sum_a\pi_a(Q_a - \alpha\ln\pi_a) + \lambda(\sum_a\pi_a - 1)$。$\partial_{\pi_a}\mathcal F = Q_a - \alpha\ln\pi_a - \alpha + \lambda = 0$ ⟹ $\pi_a = e^{Q_a/\alpha}\,e^{(\lambda-\alpha)/\alpha}$。第2因子は $a$ によらない定数なので規格化 $\sum\pi_a = 1$ で $1/Z$ に定まる。
> **19.2** $\sum_a\pi^*_a(Q_a - \alpha(Q_a/\alpha - \ln Z)) = \sum_a\pi^*_a\,\alpha\ln Z = \alpha\ln Z$。
> **19.3** (a) $Z = e^2 + 1 \approx 8.389$。$\pi^* \approx (0.881, 0.119)$。$V_{\mathrm{soft}} = \ln 8.389 \approx 2.13 > 2 = \max_a Q_a$。エントロピーボーナスの確認:$V - \mathbb E_{\pi^*}[Q] = 2.13 - 0.881 \times 2 = 0.365$、一方 $\alpha\mathcal H(\pi^*) = 1 \times [-(0.881\ln 0.881 + 0.119\ln 0.119)] \approx 0.365$ で一致。✓(b) $V_{\mathrm{soft}} \approx 0.1 \times \ln(e^{20}+1) \approx 0.1 \times 20.000 = 2.000$。$\pi^*(a_1) = \frac{1}{1+e^{-20}} \approx 1$:貪欲方策へ凝縮。
> **19.4**
>
> | 前作 $T^*$ の証明 | ソフト版 |
> | ---- | ---- |
> | 補題5.3:$\lvert\max f - \max g\rvert \le \max\lvert f-g\rvert$ | lseの非拡大性:$\lvert\mathrm{lse}(x)-\mathrm{lse}(y)\rvert \le \max\lvert x-y\rvert$ |
> | $f(a) = R + \gamma\sum P u$、$g$ は $v$ 版 | 同一(中身は変わらない) |
> | $\lvert(T^*u - T^*v)(s)\rvert \le \max_a \gamma\lvert\sum P(u-v)\rvert \le \gamma\lVert u-v\rVert_\infty$ | $\max$ を lse に読み替えて同一の評価 |
>
> よってソフトBellman作用素も $\gamma$-縮小で、不動点の存在・一意性・幾何収束がすべて継承される。
> **19.5** $V = \mathbb E_\pi[Q] + \alpha\mathcal H$ に $Q = -E$, $V = -F$, $\alpha = k_BT$, $\mathcal H = S$ を代入:$-F = -\langle E\rangle + TS$、すなわち $F = \langle E\rangle - TS$。✓

---

### 問題20 RLHF:Bradley–TerryとKL正則化つき最適方策

> [!info] 使う道具
> - Bradley–Terryモデル:$\Pr(y_1 \succ y_2) = \sigma(r_\phi(x,y_1) - r_\phi(x,y_2))$,$\sigma(z) = \dfrac{1}{1+e^{-z}}$,$\sigma'(z) = \sigma(z)(1-\sigma(z))$
> - KL正則化つき目的:$\max_\pi\ \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,D_{\mathrm{KL}}(\pi(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x))$
> - 問19.1のLagrange計算(そのまま流用できる)
> - 参照:[[強化学習の数理_2_深層学習編#第9章 現代的フロンティア — オフラインRL・探索・RLHF]] 9.3節

**問20.1** 選好ペア $(y_w \succ y_l)$ に対する負の対数尤度 $\mathcal L = -\ln\sigma(r_w - r_l)$($r_w = r_\phi(x, y_w)$ 等)について、$\partial\mathcal L/\partial r_w$ と $\partial\mathcal L/\partial r_l$ を計算せよ($\Delta := r_w - r_l$ と置くと便利)。勾配が「モデルが選好を当てられていないとき($\sigma(\Delta)$ が小さいとき)ほど大きい」ことを確認せよ。

**問20.2** Bradley–Terryモデルでは両方の報酬に同じ定数 $c$ を足しても尤度が不変であることを示せ(報酬の原点の不定性)。この事実は、報酬モデルが決めるのは「差」だけだという構造的制約を意味する。

**問20.3** KL正則化つき目的を書き下すと
$$\mathbb{E}_{y\sim\pi}\left[r(x,y) - \beta\ln\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]$$
となる。問19.1と同じLagrange計算(規格化制約つき変分)を実行し、最適方策の閉形式
$$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)$$
を導け。問19.1との違いが「基準測度が一様分布から $\pi_{\mathrm{ref}}$ に変わっただけ」であること、$\beta$ が温度 $\alpha$ の役割を果たすことを指摘せよ。

**問20.4**(数値例)2つの応答候補、$\pi_{\mathrm{ref}} = (0.5, 0.5)$、報酬 $r = (1, 0)$、$\beta = 1$ とする。$\pi^*$ を計算せよ($e \approx 2.718$)。次に $\beta = 0.25$ で再計算し($e^4 \approx 54.6$)、$\beta$ を下げる(=KL罰則を弱める)と方策が報酬の高い応答へ強く凝縮すること——そして報酬モデルの誤差もそれだけ強く搾取されるようになること——を数値で確認せよ。

> [!success]- 解答
> **20.1** $\mathcal L = -\ln\sigma(\Delta)$。$\frac{\partial\mathcal L}{\partial\Delta} = -\frac{\sigma'(\Delta)}{\sigma(\Delta)} = -(1-\sigma(\Delta)) = \sigma(\Delta) - 1$。よって $\partial\mathcal L/\partial r_w = \sigma(\Delta) - 1 \le 0$(勝者の報酬を上げる)、$\partial\mathcal L/\partial r_l = 1 - \sigma(\Delta) \ge 0$(敗者を下げる)。大きさはどちらも $1 - \sigma(\Delta)$:予測が外れているほど強く更新。
> **20.2** $\sigma((r_w + c) - (r_l + c)) = \sigma(r_w - r_l)$。差しか現れないため。
> **20.3** $\mathcal F = \sum_y \pi_y(r_y - \beta\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}}) + \lambda(\sum\pi_y - 1)$。$\partial_{\pi_y}\mathcal F = r_y - \beta\ln\frac{\pi_y}{\pi_{\mathrm{ref},y}} - \beta + \lambda = 0$ ⟹ $\pi_y \propto \pi_{\mathrm{ref},y}\,e^{r_y/\beta}$。問19.1で $\pi_{\mathrm{ref}}$ を一様分布にとれば(定数として $Z$ に吸収)Boltzmann方策に帰着し、$\beta \leftrightarrow \alpha$。
> **20.4** $\beta = 1$:重み $(0.5e, 0.5) = (1.359, 0.5)$、$\pi^* \approx (0.731, 0.269)$。$\beta = 0.25$:重み $(0.5e^4, 0.5) = (27.3, 0.5)$、$\pi^* \approx (0.982, 0.018)$。KL罰則を弱めると分布が報酬最大の応答にほぼ全確率を置く。報酬 $r$ が真の選好の不完全な代理である以上、この凝縮は「代理の穴」への凝縮になりうる——$\beta$ は搾取の抑え(参照分布への繋留の強さ)である。

---

> [!quote] 結び
> 本編で手を動かした計算——望遠鏡和、確率の規格化条件の微分、Lagrange法、縮小性の評価——は、いずれも数種類の技法の繰り返しであったことに気づいただろうか。Bellman方程式の導出(問題3)と性能差分補題(問題15)、ベースラインの不偏性(問題10)とFisher行列の1次項消滅(問題17)、Boltzmann方策(問題19)とRLHFの閉形式(問題20)は、それぞれ同じ計算の再演である。解説テキスト2冊が主張した「理論の背骨は一本」という見方を、指先の感覚として持ち帰ってほしい。数値実験による確認は [[強化学習の数理_4_演習_実践コーディング編]] へ。
