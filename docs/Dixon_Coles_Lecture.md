# Understanding the Dixon-Coles Football Model

Welcome to this crash course on predictive sports modeling. By the end of this lecture, you will understand the mathematics behind the **Dixon-Coles (1997)** model, which remains the foundational algorithm for football (soccer) analytics and betting markets.

---

## 1. The Core Idea: Goals as Poisson Processes

In football, goals are rare, independent events that occur over a fixed period of time (90 minutes). In statistics, the number of events happening in a fixed interval of time is best modeled by the **Poisson Distribution**.

If a team has an *expected goals* (mean) rate of $\lambda$, the probability they score exactly $x$ goals is:
$$ P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!} $$

If we assume Team A's goals and Team B's goals are completely independent, the probability of a specific scoreline (e.g., Team A scores $x$, Team B scores $y$) is simply the product of their individual Poisson probabilities:
$$ P(X=x, Y=y) = P(X=x) \times P(Y=y) = \left( \frac{\lambda^x e^{-\lambda}}{x!} \right) \left( \frac{\mu^y e^{-\mu}}{y!} \right) $$

---

## 2. Defining the Vocabulary (The Parameters)

To calculate $\lambda$ (Home expected goals) and $\mu$ (Away expected goals), we break a team's strength into parameters. 
Let $i$ be the Home Team and $j$ be the Away Team.

*   **$\alpha_i$ (Attack Strength):** How many goals team $i$ scores relative to an average team. ($\alpha = 1.2$ means they are 20% better at attacking than average).
*   **$\beta_j$ (Defense Vulnerability):** How many goals team $j$ *concedes* relative to an average team. (Note: Higher $\beta$ means a WORSE defense!).
*   **$\gamma$ (Home Advantage):** A global multiplier representing the boost a team gets when playing in their own stadium.

The expected goals are defined as:
$$ \lambda = \alpha_i \times \beta_j \times \gamma \quad \text{(Home Expected Goals)} $$
$$ \mu = \alpha_j \times \beta_i \quad \text{(Away Expected Goals)} $$

*(Note: On a log-scale, which computers prefer for optimization, this becomes $\log \lambda = a_i + b_j + g$.)*

---

## 3. The Dixon-Coles Correction ($\tau$)

**The Problem:** The Independent Poisson model is flawed. In real football, if it's 0-0 in the 80th minute, both teams play defensively to secure a point. If it's 1-1, they might settle for a draw. This means low-scoring draws ($0-0, 1-1$) and narrow wins ($1-0, 0-1$) occur *more often* in reality than the Independent Poisson model predicts.

**The Solution:** Mark Dixon and Stuart Coles introduced a correlation parameter **$\rho$ (rho)** to artificially boost or penalize these specific low-score probabilities, using an adjustment matrix $\tau(x, y)$:

*   **$0-0$ Scoreline:** $\tau(0,0) = 1 - \lambda \mu \rho$
*   **$1-0$ Scoreline:** $\tau(1,0) = 1 + \mu \rho$
*   **$0-1$ Scoreline:** $\tau(0,1) = 1 + \lambda \rho$
*   **$1-1$ Scoreline:** $\tau(1,1) = 1 - \rho$
*   **All other scorelines:** $\tau(x,y) = 1.0$ (no adjustment)

The adjusted probability becomes:
$$ P_{DC}(X=x, Y=y) = \tau(x, y) \times P(X=x) \times P(Y=y) $$

*(Notice the cross-terms: the $1-0$ scoreline is adjusted by the AWAY team's expectation $\mu$, and $0-1$ by the HOME team's expectation $\lambda$. This is mathematically required so the total probabilities still sum to exactly 1.0).*

---

## 4. Pen-and-Paper Example

Let's model a mini-league of 4 teams. Assume we already ran our machine learning optimizer and it gave us these parameters:
*   **Global:** Home Advantage $\gamma = 1.2$, Correlation $\rho = 0.1$
*   **Team A (Arsenal):** $\alpha = 1.5, \beta = 0.6$ *(Great attack, great defense)*
*   **Team B (Burnley):** $\alpha = 0.8, \beta = 1.4$ *(Poor attack, poor defense)*
*   **Team C (Chelsea):** $\alpha = 1.1, \beta = 0.9$ *(Average)*
*   **Team D (Derby):** $\alpha = 0.5, \beta = 2.0$ *(Terrible)*

### Match: Arsenal (Home) vs. Burnley (Away)

**Step 1: Calculate Expected Goals ($\lambda, \mu$)**
*   $\lambda = \alpha_A \times \beta_B \times \gamma = 1.5 \times 1.4 \times 1.2 = \mathbf{2.52}$
*   $\mu = \alpha_B \times \beta_A = 0.8 \times 0.6 = \mathbf{0.48}$

**Step 2: Calculate Independent Probabilities for a 1-0 Win**
*   $P(X=1) = \frac{2.52^1 \cdot e^{-2.52}}{1!} = 2.52 \cdot 0.0804 = 0.202$
*   $P(Y=0) = \frac{0.48^0 \cdot e^{-0.48}}{0!} = 1 \cdot 0.618 = 0.618$
*   $P_{indep}(1, 0) = 0.202 \times 0.618 = \mathbf{0.124}$ (or 12.4%)

**Step 3: Apply the Dixon-Coles Correction ($\tau$)**
Since the score is $1-0$, we look up the formula: $\tau(1,0) = 1 + \mu \rho$
*   $\tau(1,0) = 1 + (0.48 \times 0.1) = \mathbf{1.048}$
*   $P_{DC}(1, 0) = 1.048 \times 0.124 = \mathbf{0.130}$ (or 13.0%)

*The model intelligently bumped up the probability of a 1-0 grind-out win by about 5% to match historical reality!*

---

## 5. Time Decay (Walk-Forward Optimization)

Teams change. A 5-0 win from three years ago shouldn't matter as much as a 1-0 win from last week. 
We apply an exponential time decay weight to every historical match in our log-likelihood function:
$$ w_k = e^{-\xi \cdot \Delta t} $$
Where $\Delta t$ is days since the match, and $\xi$ (xi) controls the decay speed. The algorithm tests different $\xi$ values (e.g., half-life of 90 days vs 365 days) through walk-forward cross-validation to find the one that predicts future games most accurately.
