"""How much a research result is worth once the search that found it is counted, and how long a forward test must run.

* Deflated Sharpe (Bailey and Lopez de Prado, 2014): the best of N tried variants looks good by luck alone. Under the null
  (no variant has an edge) the expected maximum of N independent Sharpe estimates is about
  sd * ((1 - g) * z(1 - 1/N) + g * z(1 - 1/(N e))), g the Euler-Mascheroni constant, sd the spread of the estimates. The
  deflated Sharpe is the probability that the observed Sharpe beats that, with the standard error corrected for skew and
  fat tails. Per-period (per-trade) Sharpe throughout, never annualised: annualising shrinks the standard error and
  saturates the answer at 0 or 1.
* In t units the same correction needs no Sharpe at all: a t-statistic is a per-trade Sharpe times sqrt(n), and under the
  null each variant's t is about standard normal, so the best of N has the expected maximum above with sd = 1. That is what
  `deflated_t` uses; it is how the research logs (date-clustered t) can be read against the number of variants tried.
* Sample size: the number of independent units (trading dates) a forward test needs to see a given mean at a one-sided
  level alpha with the given power, from the spread of the per-date means.
Trials that resemble each other are not independent, so N overstates the correction; pass the number of genuinely
different ideas when that is known, and report both."""

import math
from dataclasses import dataclass
from statistics import NormalDist

_Z = NormalDist()
EULER_GAMMA = 0.5772156649015329


def expected_max_null(n_trials: int, sd: float = 1.0) -> float:
    """Expected largest of `n_trials` independent estimates whose true value is 0 and spread is `sd`."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    return sd * (
        (1 - EULER_GAMMA) * _Z.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _Z.inv_cdf(1 - 1 / (n_trials * math.e))
    )


@dataclass(frozen=True)
class Deflated:
    observed: float  # the selected variant's statistic (per-trade Sharpe, or t)
    expected_max: float  # what the best of n_trials reaches by luck
    probability: float  # probability the true value beats the luck benchmark; > 0.95 is strong evidence
    n_trials: int


def deflated_sharpe(
    sharpe: float, n_obs: int, n_trials: int, sharpe_sd: float, skew: float = 0.0, kurtosis: float = 3.0
) -> Deflated:
    """Deflated Sharpe ratio of a per-period `sharpe` measured on `n_obs` returns, picked as the best of `n_trials`
    variants whose Sharpe estimates spread by `sharpe_sd`. `kurtosis` is the plain kurtosis (3 for a normal)."""
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    benchmark = expected_max_null(n_trials, sharpe_sd)
    variance = (1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / (n_obs - 1)
    if variance <= 0:
        raise ValueError("skew/kurtosis give a non-positive variance for this Sharpe")
    return Deflated(sharpe, benchmark, _Z.cdf((sharpe - benchmark) / math.sqrt(variance)), n_trials)


def deflated_t(t: float, n_trials: int) -> Deflated:
    """A t-statistic (e.g. date-clustered) read against the best of `n_trials` variants: the probability its true value is
    above what luck alone gives the best of that many."""
    benchmark = expected_max_null(n_trials)
    return Deflated(t, benchmark, _Z.cdf(t - benchmark), n_trials)


def units_needed(effect: float, sd: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Independent units (e.g. trading dates) needed to detect a mean of `effect` when single units spread by `sd`,
    one-sided test at `alpha` with `power`."""
    if effect <= 0 or sd <= 0:
        raise ValueError("effect and sd must be positive")
    z = _Z.inv_cdf(1 - alpha) + _Z.inv_cdf(power)
    return math.ceil((z * sd / effect) ** 2)


def detectable_effect(n_units: int, sd: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest mean `n_units` independent units with spread `sd` can detect (one-sided, `alpha`, `power`)."""
    if n_units < 1 or sd <= 0:
        raise ValueError("n_units must be >= 1 and sd positive")
    return (_Z.inv_cdf(1 - alpha) + _Z.inv_cdf(power)) * sd / math.sqrt(n_units)
