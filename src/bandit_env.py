# src/bandit_env.py

from dataclasses import dataclass, field
from typing import Dict, Callable, List

import numpy as np
import pandas as pd

from .config import ASSETS, PRESAMPLE_END, BANDIT_START
from .bayes_model import BayesianArm

PolicyFn = Callable[[Dict[str, BayesianArm]], str]


@dataclass
class BanditEnvironment:
    returns_df: pd.DataFrame
    assets: List[str] = field(default_factory=lambda: ASSETS)
    presample_end: str = PRESAMPLE_END
    bandit_start: str = BANDIT_START
    initial_wealth: float = 1.0

    def __post_init__(self):
        df = self.returns_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.returns_df = df

        # Split presample / bandit periods
        self.presample = df[df["date"] <= pd.to_datetime(self.presample_end)]
        self.bandit = df[df["date"] >= pd.to_datetime(self.bandit_start)]

        # Restrict to selected assets
        self.presample = self.presample[self.presample["ticker"].isin(self.assets)]
        self.bandit = self.bandit[self.bandit["ticker"].isin(self.assets)]

        # Create arms and pre-train on presample
        self.arms: Dict[str, BayesianArm] = {
            name: BayesianArm(name) for name in self.assets
        }
        self._pretrain()

    def _pretrain(self):
        for name in self.assets:
            arm = self.arms[name]
            xs = self.presample.loc[self.presample["ticker"] == name, "ret"].values
            for x in xs:
                arm.update(float(x))

    def simulate(self, policy_fn: PolicyFn) -> pd.DataFrame:
        """
        Run bandit from BANDIT_START to end of available data.
        """
        wealth = self.initial_wealth
        dates = sorted(self.bandit["date"].unique())
        history = []

        for d in dates:
            day_slice = self.bandit[self.bandit["date"] == d]

            # choose arm
            chosen = policy_fn(self.arms)

            r_row = day_slice[day_slice["ticker"] == chosen]
            if r_row.empty:
                continue

            r = float(r_row["ret"].iloc[0])
            wealth *= np.exp(r)

            # bandit update: only chosen arm
            self.arms[chosen].update(r)

            history.append(
                {
                    "date": d,
                    "chosen": chosen,
                    "ret": r,
                    "wealth": wealth,
                }
            )

        return pd.DataFrame(history)
