"""Gaussian Hidden Markov Model with real-time discipline.

THE THREE THINGS ALMOST EVERY HMM REGIME PROJECT GETS WRONG
-----------------------------------------------------------

**1. Using ``predict()`` or ``predict_proba()`` as a trading signal.**
``hmmlearn.predict()`` runs Viterbi over the *entire* sequence, so the state it
assigns to March 2008 is informed by what happened in 2009. ``predict_proba()``
returns *smoothed* posteriors ``P(s_t | ALL data)`` - same problem. Both look
completely reasonable in a plot and produce spectacular, entirely fictitious
backtests. What a real-time decision needs is the **filtered** probability
``P(s_t | data up to t)``, implemented here in :func:`filtered_probabilities`.

**2. Fitting once on all history.**
Even with filtered probabilities, a model whose parameters were estimated on
1970-2026 knew about 2008 when it labelled 1985. :func:`expanding_window_states`
refits on data up to ``t`` only.

**3. Label switching.**
HMM states are unidentified: "state 0" from a 1995 refit has no relationship to
"state 0" from a 2010 refit. Without deterministic relabelling after every fit the
regime timeline is noise. :func:`relabel_by_growth` fixes an economic ordering.

Everything below exists to get those three right.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..config import model_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    X: np.ndarray,
    n_states: int,
    covariance_type: str = "diag",
    n_init: int = 20,
    seed: int = 42,
    max_iter: int = 500,
):
    """Fit a Gaussian HMM, taking the best of ``n_init`` random starts.

    EM is non-convex: different initializations converge to materially different
    regime assignments. A single unseeded fit is not reproducible and not
    defensible, so we run many seeds and keep the best log-likelihood.

    ``covariance_type='diag'`` by default: a 4-state full-covariance model on 5
    features carries ~95 free parameters, and the effective sample here is a
    handful of business cycles, not several hundred months.
    """
    from hmmlearn.hmm import GaussianHMM

    best, best_ll = None, -np.inf

    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=max_iter,
            random_state=seed + i,
            tol=1e-4,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
                ll = model.score(X)
        except Exception as exc:  # noqa: BLE001 - degenerate starts are expected
            log.debug("HMM init %d failed: %s", i, exc)
            continue

        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = model, ll

    if best is None:
        raise RuntimeError(f"All {n_init} HMM initializations failed")

    return best, best_ll


def relabel_by_growth(model, growth_col: int = 0):
    """Impose a deterministic, economically meaningful state ordering.

    States are sorted ascending by their mean on the growth feature, so state 0 is
    always the weakest-growth state and state K-1 the strongest, in every refit.

    Without this the expanding-window timeline is meaningless: consecutive refits
    would permute the labels arbitrarily and the "regime history" would be an
    artefact of EM's initialization rather than of the economy.
    """
    order = np.argsort(model.means_[:, growth_col])

    model.means_ = model.means_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]

    if hasattr(model, "covars_"):
        covars = model.covars_
        cov_type = model.covariance_type
        if cov_type == "diag":
            model._covars_ = np.array([np.diag(covars[i]) for i in order])
        elif cov_type == "full":
            model._covars_ = covars[order]
        elif cov_type == "spherical":
            model._covars_ = np.asarray(covars)[order]
        elif cov_type == "tied":
            pass  # shared across states; nothing to permute

    return model, order


# ---------------------------------------------------------------------------
# THE CRITICAL PIECE - filtered (forward-only) probabilities
# ---------------------------------------------------------------------------

def _log_emission(model, X: np.ndarray) -> np.ndarray:
    """log P(x_t | state=j), shape (T, K)."""
    return model._compute_log_likelihood(X)


def filtered_probabilities(model, X: np.ndarray) -> np.ndarray:
    """``P(s_t = j | x_1..x_t)`` - the forward filter, computed in log space.

    This is the ONLY probability that may drive a backtest, because it conditions
    solely on information available at ``t``.

    Implemented directly rather than via hmmlearn because its forward-pass API is
    private and changes between versions; the recursion is short and this keeps the
    guarantee explicit and testable::

        alpha_1(j) = pi_j b_j(x_1)
        alpha_t(j) = [sum_i alpha_{t-1}(i) A_ij] b_j(x_t)

    normalising at each step. Log space avoids underflow over long samples.
    """
    log_b = _log_emission(model, X)
    log_A = np.log(np.maximum(model.transmat_, 1e-300))
    log_pi = np.log(np.maximum(model.startprob_, 1e-300))

    T, K = log_b.shape
    log_alpha = np.zeros((T, K))

    log_alpha[0] = log_pi + log_b[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        # marginalise the previous state, then apply this step's emission
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def smoothed_probabilities(model, X: np.ndarray) -> np.ndarray:
    """``P(s_t = j | ALL data)`` - full-sample posterior.

    Provided for the explicit filtered-vs-smoothed comparison in the report, which
    demonstrates the size of the bias. **Never** use this as a trading signal.
    """
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_n_states(
    X: np.ndarray,
    grid: list[int],
    covariance_type: str = "diag",
    n_init: int = 10,
    seed: int = 42,
    holdout_frac: float = 0.25,
) -> pd.DataFrame:
    """Choose the number of states by held-out likelihood, plus BIC and AIC.

    The brief assumed four states because the economic story has four. That is a
    hypothesis, not a fact - the data may support two (expansion / contraction) or
    three. Selecting on a held-out tail rather than in-sample fit avoids simply
    rewarding the most heavily parameterised model.
    """
    n_obs, n_features = X.shape
    split = int(n_obs * (1 - holdout_frac))
    X_train, X_test = X[:split], X[split:]

    rows = []
    for k in grid:
        try:
            model, ll_train = fit_hmm(
                X_train, k, covariance_type, n_init=n_init, seed=seed
            )
            ll_test = model.score(X_test)

            if covariance_type == "diag":
                n_cov = k * n_features
            elif covariance_type == "full":
                n_cov = k * n_features * (n_features + 1) // 2
            else:
                n_cov = k
            n_params = (k - 1) + k * (k - 1) + k * n_features + n_cov

            rows.append(
                {
                    "n_states": k,
                    "loglik_train": round(ll_train, 1),
                    "loglik_holdout": round(ll_test, 1),
                    "loglik_holdout_per_obs": round(ll_test / max(len(X_test), 1), 4),
                    "n_params": n_params,
                    "bic": round(-2 * ll_train + n_params * np.log(len(X_train)), 1),
                    "aic": round(-2 * ll_train + 2 * n_params, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("n_states=%d failed: %s", k, exc)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expanding-window (walk-forward) state inference
# ---------------------------------------------------------------------------

def expanding_window_states(
    features: pd.DataFrame,
    n_states: int = 3,
    initial_train_months: int = 180,
    refit_every_months: int = 12,
    covariance_type: str = "diag",
    n_init: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Walk-forward regime probabilities with no future information.

    At each step the model is fitted on ``features[:t]``, states are relabelled
    deterministically, and the filtered probability at ``t`` is recorded. The model
    is refitted every ``refit_every_months``; between refits the existing
    parameters are applied to the extended history (still filtered, still causal).

    This is the series a backtest may legitimately trade on.
    """
    X_all = features.to_numpy(dtype="float64")
    n_obs = len(features)

    if n_obs <= initial_train_months:
        raise ValueError(
            f"Need more than {initial_train_months} observations, got {n_obs}"
        )

    prob_cols = [f"p_state_{i}" for i in range(n_states)]
    out = pd.DataFrame(index=features.index, columns=prob_cols, dtype="float64")
    out["state"] = pd.NA
    out["refit_date"] = pd.NaT

    model = None
    last_refit = -np.inf
    n_refits = 0

    for t in range(initial_train_months, n_obs):
        if (t - last_refit) >= refit_every_months or model is None:
            X_train = X_all[:t]  # strictly data up to (not including) t
            try:
                model, _ = fit_hmm(
                    X_train, n_states, covariance_type, n_init=n_init, seed=seed
                )
                model, _ = relabel_by_growth(model, growth_col=0)
                last_refit = t
                n_refits += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("refit at %s failed: %s", features.index[t], exc)
                if model is None:
                    continue

        # filter over history through t, and take only the final step
        probs = filtered_probabilities(model, X_all[: t + 1])
        out.iloc[t, out.columns.get_indexer(prob_cols)] = probs[-1]
        out.iloc[t, out.columns.get_loc("state")] = int(np.argmax(probs[-1]))
        out.iloc[t, out.columns.get_loc("refit_date")] = features.index[int(last_refit)]

    log.info(
        "expanding-window HMM: %d refits, live signal from %s",
        n_refits,
        features.index[initial_train_months].date(),
    )
    return out


def full_sample_states(
    features: pd.DataFrame,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_init: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Full-sample fit, reporting filtered AND smoothed probabilities.

    Used for the in-sample description of the regimes and, importantly, for the
    filtered-vs-smoothed comparison that quantifies how much look-ahead the
    conventional approach would have injected.
    """
    X = features.to_numpy(dtype="float64")
    model, ll = fit_hmm(X, n_states, covariance_type, n_init=n_init, seed=seed)
    model, order = relabel_by_growth(model, growth_col=0)

    filt = filtered_probabilities(model, X)
    smooth = smoothed_probabilities(model, X)

    out = pd.DataFrame(index=features.index)
    for i in range(n_states):
        out[f"p_filtered_{i}"] = filt[:, i]
    for i in range(n_states):
        out[f"p_smoothed_{i}"] = smooth[:, i]

    out["state_filtered"] = filt.argmax(axis=1)
    out["state_smoothed"] = smooth.argmax(axis=1)
    out.attrs["loglik"] = ll
    out.attrs["model"] = model
    out.attrs["state_order"] = order

    disagree = (out["state_filtered"] != out["state_smoothed"]).mean()
    log.info(
        "full-sample HMM: loglik=%.1f | filtered and smoothed disagree on %.1f%% of months",
        ll, 100 * disagree,
    )
    return out


def describe_states(model, feature_names: list[str], n_states: int) -> pd.DataFrame:
    """Per-state feature means and expected durations - how to read the states.

    An HMM state is just an index until it is characterised. This turns "state 2"
    into "growth -1.4 sigma, inflation +0.3 sigma, expected duration 11 months",
    which is what makes the model interpretable rather than a black box.
    """
    rows = []
    for i in range(n_states):
        row = {"state": i}
        row |= {name: round(float(model.means_[i, j]), 3)
                for j, name in enumerate(feature_names)}
        p_self = float(model.transmat_[i, i])
        row["self_transition"] = round(p_self, 3)
        row["expected_duration_months"] = round(1.0 / max(1e-9, 1 - p_self), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def default_hmm_config() -> dict:
    return model_config()["hmm"]


# ---------------------------------------------------------------------------
# Robust model selection
# ---------------------------------------------------------------------------

def select_n_states_robust(
    features: pd.DataFrame,
    grid: list[int],
    covariance_type: str = "diag",
    n_init: int = 6,
    seeds: tuple[int, ...] = (100, 200, 300, 400, 500),
    min_episodes_per_state: int = 6,
    stability_threshold: float = 0.70,
) -> pd.DataFrame:
    """Select the number of states on STABILITY and episode adequacy, not likelihood.

    Empirically (see reports/model_selection.md) the usual criteria fail on this
    problem:

    * **BIC falls monotonically** across the whole grid tested (2 -> 8 states). It
      never turns, so it identifies no optimum - it simply rewards parameters.
    * **Held-out likelihood is non-monotonic and unstable**, because the held-out
      tail contains COVID, an observation unlike anything in training. Whichever K
      happens to accommodate it "wins" for reasons that have nothing to do with
      regime structure.

    Two criteria that do discriminate:

    * **Stability** - refit under different random seeds and measure label agreement
      (adjusted Rand index). If seeds disagree, the "regimes" are artefacts of EM
      initialization rather than features of the economy.
    * **Episode adequacy** - the effective sample is the number of regime EPISODES,
      not months. A state entered four times cannot support a conditional return
      estimate, however many months it spans.
    """
    from sklearn.metrics import adjusted_rand_score

    from .hysteresis import episode_summary

    X = features.to_numpy(dtype="float64")
    rows = []

    for k in grid:
        labelings = []
        for seed in seeds:
            try:
                model, _ = fit_hmm(X, k, covariance_type, n_init=n_init, seed=seed)
                model, _ = relabel_by_growth(model, growth_col=0)
                labelings.append(filtered_probabilities(model, X).argmax(axis=1))
            except Exception as exc:  # noqa: BLE001
                log.warning("K=%d seed=%d failed: %s", k, seed, exc)

        if len(labelings) < 2:
            continue

        aris = [
            adjusted_rand_score(labelings[0], other) for other in labelings[1:]
        ]
        stability = float(np.mean(aris))

        states = pd.Series(labelings[0], index=features.index)
        es = episode_summary(states)
        min_eps = int(es["episodes"].min()) if not es.empty else 0
        n_populated = len(es)

        rows.append(
            {
                "n_states": k,
                "stability_ari": round(stability, 3),
                "states_populated": n_populated,
                "min_episodes": min_eps,
                "min_months": int(es["total_months"].min()) if not es.empty else 0,
                "stable_enough": stability >= stability_threshold,
                "episodes_adequate": min_eps >= min_episodes_per_state,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["acceptable"] = df["stable_enough"] & df["episodes_adequate"]
    # among acceptable configurations prefer the richest description of the economy
    acceptable = df[df["acceptable"]]
    df.attrs["recommended"] = (
        int(acceptable["n_states"].max()) if not acceptable.empty
        else int(df.loc[df["stability_ari"].idxmax(), "n_states"])
    )
    return df
