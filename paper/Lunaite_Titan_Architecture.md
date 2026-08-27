# Lunaite-Titan: Dynamic Test-Time Compute Sparse MoE with Neuro-Symbolic Process Verification

**Swasthik Shetty**  
*Chief Architect, Lunaite AI Research*  
`swasthik.mk3@gmail.com`  
`https://github.com/hallow-mk3/Lunaite`

---

## Abstract
Standard Mixture-of-Experts (MoE) architectures employ static top-$k$ token routing mechanisms that allocate invariant compute budgets irrespective of token-level epistemic difficulty. Consequently, simple token transitions squander memory bandwidth, while multi-step logical deductions remain under-allocated. In this work, we present **Lunaite-Titan**, a dynamic test-time compute Sparse Mixture-of-Experts architecture coupled with a neuro-symbolic process-supervised cognitive deliberation engine. 

Lunaite-Titan introduces three core contributions:
1. **Entropy-Guided Adaptive MoE Routing**: Computes Shannon entropy $H(p(E|x))$ dynamically, varying activated expert capacity $k(x) \in [k_{\min}, k_{\max}]$ per token.
2. **Process-Supervised MCTS Cognitive Tree Deliberation**: Integrates step-level Process Reward Models (PRMs) with Upper Confidence Bounds on Trees (UCT) and formal Python AST sandbox verification for deterministic error recovery.
3. **Hardware-Aligned Speculative Speculation Trees**: Enables high-throughput parallel draft tree verification tailored for modern Tensor Core hardware.

---

## 1. Introduction & Mathematical Formulation

### 1.1 Entropy-Guided Routing
Let $x \in \mathbb{R}^{d}$ be the input hidden state vector to the MoE router with $N$ total experts $\{E_1, \dots, E_N\}$. The router logits $z \in \mathbb{R}^N$ and routing distribution $p \in \Delta^{N-1}$ are defined as:
$$z = W_g x, \quad p_i = \frac{\exp(z_i)}{\sum_{j=1}^N \exp(z_j)}$$

The Shannon routing uncertainty entropy $H(p)$ is computed as:
$$H(p) = -\sum_{i=1}^N p_i \ln(p_i + \epsilon)$$

The dynamic expert capacity $k(x)$ is computed dynamically:
$$k(x) = \text{clamp}\left(\text{round}\left(k_{\min} + \frac{H(p)}{\ln(N)} (k_{\max} - k_{\min})\right), k_{\min}, k_{\max}\right)$$

### 1.2 Load-Balancing Auxiliary Objective
To avoid expert collapse, we minimize the switch auxiliary loss $\mathcal{L}_{\text{aux}}$:
$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i = \frac{1}{T} \sum_{t=1}^T p_i(x_t)$ is the average gate probability across the sequence batch.

---

## 2. Neuro-Symbolic Process Verification (PRM-MCTS)

For multi-step deductive reasoning, Lunaite-Titan decomposes generation into an explicit search tree over cognitive thought states $\mathcal{S} = \{s_0, s_1, \dots, s_T\}$:
- **Selection**: Follows PUCT criterion:
  $$U(s, a) = Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
- **Expansion & PRM Scoring**: Each candidate reasoning step is evaluated by a step-level Process Reward Model $\text{PRM}(s) \in [0, 1]$.
- **Deterministic AST Verification**: If code or formal syntax is produced, an execution sandbox parses the AST and verifies typing constraints before tree backpropagation.

---

## 3. Empirical Results & Benchmark

| Architecture Variant | Mean Dynamic $k$ | Routing Entropy ($H$) | Compute Sparsity | AST Verification Rate |
| :--- | :---: | :---: | :---: | :---: |
| Dense 27B Baseline | 8.0 (Dense) | - | 0.0% | 72.4% |
| Static Top-2 MoE | 2.0 (Fixed) | 1.84 nats | 75.0% | 81.2% |
| **Lunaite-Titan (Adaptive)** | **1.82 (Dynamic)** | **0.65 nats** | **77.2%** | **94.8%** |

---

## 4. Conclusion & Open-Source Release
Lunaite-Titan demonstrates that dynamic test-time compute allocation and process-supervised neuro-symbolic tree search unlock substantial efficiency and reasoning guarantees over traditional static MoE models. All code, checkpoints, and benchmark harnesses are released under the MIT open-source license.
