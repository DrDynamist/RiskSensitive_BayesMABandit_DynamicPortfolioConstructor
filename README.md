# RiskSensitive_BayesMABandit_DynamicPortfolioConstructor
Multi Armed Bandit-based SNP500 Portfolio Optimizer. Regular Thompson Sampling and Risk Aware (Conditional ValueAtRisk) Thompson Bandit.
Arms include pre-set portfolios of the following constructions: 
(1. Equal weight over active selected portfolios 2. Equal weight over top 5 most volatile stocks 3. Momentum-based, 4. Markowitz (min variance)
Tested against buy-and-hold strategy

Posterior predictive checked performed, under Normal-Inverse-Gamma conjugacy assumptions
