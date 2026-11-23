import math
import numpy as np
import scipy as sp
import mpmath as mp
import warnings

from .truncmix_helpers import (
    _update_I_Beta_curr,
    _update_zeta_array_row,
    _update_phi_array_row,
    _create_recursive_Beta_Binomial_array,
    _update_Beta_Binomial_array,
    _calculate_RIP_null,
    _calculate_truncated_dirichlet_mean_Gauss_Legendre,
    _precompute_log_I_Beta,
    _update_row_array_general,
    _get_sequential_truncation_window,
)


def welford_mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Welford's online algorithm for the running mean and variance
    """
    m = [x[0]]
    v = [0]
    for i, xi in enumerate(x[1:]):
        m.append(m[-1] + (xi - m[-1]) / (i + 2))
        v.append(v[-1] + (xi - m[-2]) * (xi - m[-1]))
    v = v / np.arange(1, len(x) + 1)
    return np.array(m), v

##########################################################################################


class NonnegMean:
    """
    Tests of the hypothesis that the mean of a population of values in [0, u] is less than or equal to t.
        Several tests are implemented, all ultimately based on martingales or supermartingales:
            Kaplan-Kolmogorov (with and without replacement)
            Kaplan-Markov (without replacement)
            Kaplan-Wald (without replacement)
            Wald SPRT (with and with replacement)
            ALPHA supermartingale test (with and without replacement, using a variety of estimators)
            Betting martingale tests (with and without replacement, using a variety of betting rules)
    Some tests work for all nonnegative populations; others require a finite upper bound `u`.
    Many of the tests have versions for sampling with replacement (`N=np.inf`) and for sampling
    without replacement (`N` finite).
    Betting martingales and ALPHA martingales are different parametrizations of the same tests, but
      lead to different heuristics for selecting the parameters.
    """

    TESTS = (
        ALPHA_MART := "ALPHA_MART",
        BETTING_MART := "BETTING_MART",
        KAPLAN_KOLMOGOROV := "KAPLAN_KOLMOGOROV",
        KAPLAN_MARKOV := "KAPLAN_MARKOV",
        KAPLAN_WALD := "KAPLAN_WALD",
        WALD_SPRT := "WALD_SPRT",
    )

    def __init__(
        self,
        test: callable = None,
        estim: callable = None,
        bet: callable = None,
        u: float = 1,
        N: int = np.inf,
        t: float = 1/2,
        random_order: bool = True,
        **kwargs,
    ):
        """
        kwargs can be used to set attributes such as `betting` and parameters later used by
        `test`, `estim`, or `bet`, for instance, `eta` and to pass `c`, `d`, `f`, and `minsd` to
        `shrink_trunc()` or other estimators or betting strategies.
        """
        if test is None:  # default to alpha_mart
            test = self.alpha_mart
        if estim is None:
            estim = self.fixed_alternative_mean
            self.eta = kwargs.get(
                "eta", t + (u - t) / 2
            )  # initial estimate of population mean
        if bet is None:
            bet = self.fixed_bet
            self.lam = kwargs.get("lam", 0.5)  # initial fraction of fortune to bet
        self.test = test.__get__(self)
        self.estim = estim.__get__(self)
        self.bet = bet.__get__(self)
        self.u = u
        self.N = N
        self.t = t
        self.random_order = random_order
        self.kwargs = kwargs  # preserving these for __str__()
        self.__dict__.update(kwargs)

    def __str__(self):
        return (
            f"test: {str(self.test)} estim: {str(self.estim)} bet: {str(self.bet)} upper bound u: {self.u} "
            f"N: {self.N} null mean t: {self.t} kwargs: {self.kwargs}"
        )

    def alpha_mart(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds the ALPHA martingale for the hypothesis that the population
        mean is less than or equal to t using a martingale method,
        for a population of size N, based on a series of draws x.

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        attributes used:
            keyword arguments for estim() and for this function
            u: float > 0 (default 1)
                upper bound on the population
            eta: float in (t,u] (default u*(1-eps))
                value parametrizing the bet. Use alternative hypothesized population mean for polling audit
                or a value nearer the upper bound for comparison audits


        Returns
        -------
        p: float
            sequentially valid p-value of the hypothesis that the population mean is less than or equal to t
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        N = self.N
        t = self.t
        u = self.u
        atol = kwargs.get("atol", 2 * np.finfo(float).eps)
        rtol = kwargs.get("rtol", 10**-6)
        _S, Stot, _j, m = self.sjm(N, t, x)
        x = np.array(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            etaj = self.estim(x)
            terms = np.cumprod((x * etaj / m + (u - x) * (u - etaj) / (u - m)) / u)
        terms[m > u] = 0  # true mean is certainly less than hypothesized
        terms[np.isclose(0, m, atol=atol)] = 1  # ignore
        terms[np.isclose(u, m, atol=atol, rtol=rtol)] = 1  # ignore
        terms[np.isclose(0, terms, atol=atol)] = (
            1  # martingale effectively vanishes; p-value 1
        )
        terms[m < 0] = np.inf  # true mean certainly greater than hypothesized
        terms[-1] = (
            np.inf if Stot > N * t else terms[-1]
        )  # final sample makes the total greater than the null
        return min(1, 1 / np.max(terms)), np.minimum(1, 1 / terms)

    def sjm(self, N: int, t: float, x: np.array) -> tuple[np.array, float, np.array, np.array]:
        """
        This method calculates the cumulative sum of the input array `x`, the total sum of `x`,
        an array of indices, and the mean of the population after each draw if the null hypothesis is true.

        Parameters
        ----------
        N : int or float
            The size of the population. If N is np.inf, it means the sampling is with replacement.
        t : float
            The hypothesized population mean under the null hypothesis.
        x : np.array
            The input data array.

        Returns
        -------
        S : np.array
            The cumulative sum of the input array `x`, excluding the last element.
        Stot : float
            The total sum of the input array `x`.
        j : np.array
            An array of indices from 1 to the length of `x`.
        m : np.array
            The mean of the population after each draw if the null hypothesis is true.
        """
        assert isinstance(N, int) or (math.isinf(N) and N > 0), "Population size is not an integer!"
        S = np.insert(np.cumsum(x), 0, 0)  # 0, x_1, x_1+x_2, ...,
        Stot = S[-1]  # sample total
        S = S[0:-1]  # same length as the data
        j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
        assert j[-1] <= N, "Sample size is larger than the population!"
        m = (
            (N * t - S) / (N - j + 1) if np.isfinite(N) else t
        )  # mean of population after (j-1)st draw, if null is true (t=eta is the mean)
        return S, Stot, j, m

    def betting_mart(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds the betting martingale for the hypothesis that the population
        mean is less than or equal to t using a martingale method,
        for a population of size N, based on a series of draws x.

        If kwargs includes `lam`, uses `lam` as a real-valued bet or a vector of real-valued bets.
        Throws an error if `lam`

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        attributes used:
            keyword arguments for bet() and for this function
            u: float > 0 (default 1)
                upper bound on the population
            eta: float in (t,u] (default u*(1-eps))
                value parametrizing the bet. Use alternative hypothesized population mean for polling audit
                or a value nearer the upper bound for card-level comparison audits


        Returns
        -------
        if kwargs.out == 'p-value' (default):
            p: float or vector of floats
                sequentially valid p-value of the hypothesis that the population mean is less than or equal to t,
                for each bet `lam`
            p_history: numpy array
                sample by sample history of p-values. Not meaningful unless the sample is in random order.
        if kwargs.out == 'mart':
                numpy.array, the martingale / E-value sequence
        """
        N = self.N
        t = self.t
        u = self.u
        atol = kwargs.get("atol", 2 * np.finfo(float).eps)
        rtol = kwargs.get("rtol", 10**-6)
        out = kwargs.get("out", "p-values")
        _S, Stot, _j, m = self.sjm(N, t, x)
        x = np.array(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lam = self.bet(x)
            terms = np.cumprod(1 + lam * (x - m))
        terms[m > u] = 0  # true mean is certainly less than hypothesized
        terms[np.isclose(0, m, atol=atol)] = 1  # ignore
        terms[np.isclose(u, m, atol=atol, rtol=rtol)] = 1  # ignore
        terms[np.isclose(0, terms, atol=atol)] = (
            1  # martingale effectively vanishes; p-value 1
        )
        terms[m < 0] = np.inf  # true mean certainly greater than hypothesized
        terms[-1] = (
            np.inf if Stot > N * t else terms[-1]
        )  # final sample makes the total greater than the null
        if out == "p-values":
            output = min(1, 1 / np.max(terms)), np.minimum(1, 1 / terms)
        elif out == "mart":
            output = terms
        else:
            raise ValueError("keyword argument \'out\' is misspecified, needs to be in [\'p-values', \'mart\']")
        return output

    def mix_betting_mart(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds a simple discrete mixture martingale as a (flat) average of D TSMs each with fixed bet 'lam'

        If kwargs includes `lam`, uses `lam` as a real-valued bet or a vector of real-valued bets.
        Throws an error if `lam`

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        lam: length-D np.array of bets
        attributes used:
            keyword arguments for bet() and for this function
            u: float > 0 (default 1)
                upper bound on the population
            lams: a numpy array of bets that will be mixed over; defaults to an equi-spaced length 100 grid on [0,1/t]

        Returns
        -------
        p: float or vector of floats
            sequentially valid p-value of the hypothesis that the population mean is less than or equal to t,
            for each bet `lam`
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        N = self.N
        t = self.t
        u = self.u
        lams = getattr(self, "lams", np.linspace(0,1/t,100))
        n = len(x)
        D = len(lams)
        marts = np.zeros((D, n))
        for d in range(D):
            test = NonnegMean(N=N, u=u, bet=NonnegMean.fixed_bet, lam=lams[d])
            marts[d,:] = test.betting_mart(x, out='mart')
        mart = np.mean(marts, 0)
        return min(1, 1 / np.max(mart)), np.minimum(1, 1 / mart)



    def fixed_alternative_mean(self, x: np.array, **kwargs) -> np.array:
        """
        Compute the alternative mean just before the jth draw, for a fixed alternative that the original population
        mean is eta.
        Throws a warning if the sample implies that the fixed alternative is false (because the population would
        have negative values or values greater than u).

        S_1 := 0
        S_j := \sum_{i=1}^{j-1} x_i, j >= 1
        eta_j := (N*eta-S_j)/(N-j+1) if np.isfinite(N) else t

        Parameters
        ----------
        x: np.array
            input data
        kwargs:
            eta: float in (t, u) (default u*(1-eps))
                alternative hypothethesized value for the population mean
            u: float > 0 (default 1)
                upper bound on the population values
        """
        u = self.u
        N = self.N
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        _S, _Stot, _j, m = self.sjm(N, eta, x)
        if (negs := np.sum(m < 0)) > 0:
            warnings.warn(
                f"Implied population mean is negative in {negs} of {len(x)} terms"
            )
        if (pos := np.sum(m > u)) > 0:
            warnings.warn(
                f"Implied population mean is greater than {u=} in {pos} of {len(x)} terms"
            )
        return m

    def shrink_trunc(self, x: np.array, **kwargs) -> np.array:
        """
        apply shrinkage/truncation estimator to an array to construct a sequence of "alternative" values

        sample mean is shrunk towards eta, with relative weight d compared to a single observation,
        then that combination is shrunk towards u, with relative weight f/(stdev(x)).

        The result is truncated above at u*(1-eps) and below at m_j+e_j(c,j)

        Shrinking towards eta stabilizes the sample mean as an estimate of the population mean.
        Shrinking towards u takes advantage of low-variance samples to grow the test statistic more rapidly.

        The running standard deviation is calculated using Welford's method.

        S_1 := 0
        S_j := \sum_{i=1}^{j-1} x_i, j >= 1
        m_j := (N*t-S_j)/(N-j+1) if np.isfinite(N) else t
        e_j := c/sqrt(d+j-1)
        sd_1 := sd_2 = 1
        sd_j := sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
        eta_j :=  ( [(d*eta + S_j)/(d+j-1) + f*u/sd_j]/(1+f/sd_j) \vee (m_j+e_j) ) \wedge (u*(1-eps)-e_j)

        Parameters
        ----------
        x: np.array
            input data
        attributes used:
            eta: float in (t, u) (default u*(1-eps))
                initial alternative hypothethesized value for the population mean
            c: positive float
                scale factor in constraints to keep the estimator of the mean from getting too close to t or u before
                the empirical mean is stable
            d: positive float
                relative weight of eta compared to an observation, in updating the alternative for each term
            f: positive float
                relative weight of the upper bound u (normalized by the sample standard deviation)
            minsd: positive float
                lower threshold for the standard deviation of the sample, to avoid divide-by-zero errors and
                to limit the weight of u
        """
        # set the parameters
        u = self.u
        N = self.N
        t = self.t
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        c = getattr(self, "c", (eta-t)/2-np.finfo(float).eps)
        if u-c < t+c: # constraints could be inconsistent
            new_c = (u-c)/2
            warnings.warn(f'{c=} is too large: resetting to {new_c}')
            c = new_c
        d = getattr(self, "d", 100)
        f = getattr(self, "f", 0)
        minsd = getattr(self, "minsd", 10**-6)
        S, _, j, m = self.sjm(N, t, x)
        _, v = welford_mean_var(x)
        sdj = np.sqrt(v)
        # threshold the sd, set first two sds to 1
        sdj = np.insert(np.maximum(sdj, minsd), 0, 1)[0:-1]
        sdj[1] = 1
        weighted = ((d * eta + S) / (d + j - 1) + u * f / sdj) / (1 + f / sdj)
        tol = c / np.sqrt(d + j - 1)
        return np.minimum(
            u * (1 - np.finfo(float).eps) - tol,
            np.maximum(weighted, m * (1 + np.finfo(float).eps) + tol)
        )

    def optimal_comparison(self, x: np.array, **kwargs) -> np.array:
        """
        The value of eta corresponding to the "bet" that is optimal for ballot-level comparison audits,
        for which overstatement assorters take a small number of possible values and are concentrated
        on a single value when the CVRs have no errors.

        Let p0 be the rate of error-free CVRs, p1=0 the rate of 1-vote overstatements,
        and p2= 1-p0-p1 = 1-p0 the rate of 2-vote overstatements. Then

        eta = (1-u*p0)/(2-2*u) + u*p0 - 1/2, where p0 is the rate of error-free CVRs.

        Translating to p2=1-p0 gives:

        eta = (1-u*(1-p2))/(2-2*u) + u*(1-p2) - 1/2.

        Parameters
        ----------
        x: np.array
            input data
        error_rate_2: float
            hypothesized rate of two-vote overstatements

        Returns
        -------
        eta: float
            estimated alternative mean to use in alpha
        """
        # set the parameters
        # TO DO: double check where error_rate_2 is set
        p2 = getattr(self, "error_rate_2", 1e-5)  # rate of 2-vote overstatement errors
        return (1 - self.u * (1 - p2)) / (2 - 2 * self.u) + self.u * (1 - p2) - 1 / 2

    def deriv(lam, x, eta):
        return np.sum((x - eta) / (1 + lam * (x - eta)))

    def kelly_optimal(self, x: np.array, pop = None, **kwargs):
        """
        return the Kelly-optimal bet

        Parameters
        ----------
        x: np.array
            input data
        pop: optional np.array
            the population (order does not matter) that will be used to compute the optimal bet
        Takes x to be the population unless pop is provided
        """
        t = self.t # the null mean
        pop = getattr(self, "pop", None) # attempts to inherit pop from the class
        if pop is None:
            pop = x
        min_slope = NonnegMean.deriv(0, pop, t)
        max_slope = NonnegMean.deriv(1/t, pop, t)
        # if the return is always growing, set lambda to the maximum allowed
        if (min_slope > 0) & (max_slope > 0):
            out = 1/t
        # if the return is always shrinking, set lambda to 0
        elif (min_slope < 0) & (max_slope < 0):
            out = 0
        # otherwise, optimize on the interval [0, 1/eta]
        else:
            lam_star = sp.optimize.root_scalar(lambda lam: NonnegMean.deriv(lam, pop, t), bracket = [0, 1/t], method = 'bisect')
            assert lam_star.converged, "Could not find Kelly optimal bet, the optimization may be poorly conditioned"
            out = lam_star['root']
        return out * np.ones_like(x)


    def fixed_bet(self, x: np.array, lam = None, **kwargs) -> np.array:
        """
        Return a fixed value of lambda, the fraction of the current fortune to bet.

        Parameters
        ----------
        x: np.array
            input data

        Uses the bet `lam` if it is passed; otherwise uses the instance property `lam`
        """
        lam = lam if lam else self.lam
        return lam * np.ones_like(x)

    def best_fixed_bet(self, x: np.array, lam: float=0.5, tol: float=1.e-4, **kwargs) -> float:
        """
        Finds the best fixed bet `lambda` for the data `x`.

        Intended to be used to find a good bet from simulated data or CVRs on the assumption that the CVRs are
        accurate.

        It is NOT legitimate to use this retrospectively, only prospectively.

        Parameters
        ----------
        x: np.array
            data
        lam: float
            initial guess
        tol: float
            tolerance to terminate bisection search

        Returns
        -------
        lam: float
            estimated "best bet" for data x
        """
        return lam


    def agrapa(self, x: np.array, **kwargs) -> np.array:
        """
        maximize approximate growth rate adapted to the particular alternative (aGRAPA) bet of Waudby-Smith & Ramdas (WSR)

        This implementation alters the method from support \mu \in [0, 1] to \mu \in [0, u], and to constrain
        the bets to be positive (for one-sided tests against the alternative that the true mean is larger than
        hypothesized)

        lam_j := 0 \vee (\hat{\mu}_{j-1}-t)/(\hat{\sigma}_{j-1}^2 + (t-\hat{\mu})^2) \wedge c_grapa/t

        \hat{\sigma} is the standard deviation of the sample.

        \hat{\mu} is the mean of the sample

        The value of c_grapa \in (0, 1) is passed as an instance variable of Class NonnegMean

        The running standard deviation is calculated using Welford's method.

        S_0 := 0
        S_j := \sum_{i=0}^{j-1} x_i, j >= 1
        t_adj := (N*t-S_j)/(N-j+1) if np.isfinite(N) else t
        sd_0 := 0
        sd_j := sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
        lam_1 := self.lam
        lam_j :=  0 \vee (\hat{m_{j-1}-t)/(sd_{j-1}^2 + (t-m_{j-1})^2) \wedge c_grapa/t

        Parameters
        ----------
        x: np.array
            input data
        attributes used:
            c_grapa_0: float in (0, 1)
                initial scale factor c_j in WSR's agrapa bet
            c_grapa_max: float in (1, 1-np.finfo(float).eps]
                asymptotic limit of the value of c_j
            c_grapa_grow: float in [0, np.infty)
                rate at which to allow c to grow towards c_grapa_max.
                c_j := c_grapa_0 + (c_grapa_max-c_grapa_0)*(1-1/(1+c_grapa_grow*np.sqrt(j)))
                A value of 0 keeps c_j equal to c_grapa for all j.

        """
        # set the parameters
        u = self.u  # population upper bound
        N = self.N  # population size
        t = self.t  # hypothesized population mean
        lam = getattr(self, "lam", 0.5)  # initial bet
        c_g_0 = getattr(
            self, "c_grapa_0", (1 - np.finfo(float).eps)
        )  # initial truncation value c for agrapa
        c_g_m = getattr(
            self, "c_grapa_max", (1 - np.finfo(float).eps)
        )  # asymptotic limit of c
        c_g_g = getattr(self, "c_grapa_grow", 0)  # rate to let c grow towards c_g_m
        mj, sdj2 = welford_mean_var(x)
        t_adj = (
            (N * t - np.insert(np.cumsum(x), 0, 0)[0:-1]) / (N - np.arange(len(x)))
            if np.isfinite(N)
            else t * np.ones(len(x))
        )
        lamj = (mj - t_adj) / (sdj2 + (t_adj - mj) ** 2)  # agrapa bet
        #  shift and set first bet to self.lam
        lamj = np.insert(lamj, 0, lam)[0:-1]
        c = c_g_0 + (c_g_m - c_g_0) * (1 - 1 / (1 + c_g_g * np.sqrt(np.arange(len(x)))))
        lamj = np.maximum(0, np.minimum(c / t_adj, lamj))
        return lamj




    def trunc_mix_beta(self, x: np.array, **kwargs) -> np.array:
        """
        TruncMix(Beta) estimator for With-Replacement (WR) and Without-Replacement (WoR) ternary data suitable for ballot-polling audits.
        
        Extracts parameters from `self`:
        - a1, a2: Beta prior parameters
        - matrix_trunc_dim: Truncation dimension for WoR
        - zeta_vector, I_Beta_array, ell: Pre-initialized arrays for WoR computations
        """
        u = self.u
        N = self.N
        t = self.t

        # Extract hyperparameters or set defaults
        a1 = getattr(self, "a1", 0.1)
        a2 = getattr(self, "a2", 17.9)

        lamj = np.zeros(len(x))
        if len(x) == 0:
            return lamj

        is_ternary = np.all(np.isin(x, [0.0, 0.5, 1.0]))

        if is_ternary and not np.isfinite(N):

            # With Replacement (WR) ternary data
            scale_factor = 1 / (1 - sp.stats.beta.cdf(0.5, a1, a2))
            Beta_a1_a2 = sp.special.beta(a1, a2)
            B_05 = Beta_a1_a2 / scale_factor
            C_0 = (0.5**(a1 + a2)) / B_05
            
            j_t, k_t = 0, 0
            W_prev = 1.0

            for j, x_val in enumerate(x):
                eta_t = (j_t + a1 + C_0 / W_prev) / (j_t + k_t + a1 + a2)
                lam_j = 0.0
                
                if eta_t > t:
                    lam_j = (eta_t - t) / (t * (1 - t))
                
                lamj[j] = lam_j
                
                W_prev *= (1 + lam_j * (x_val - t))
                j_t += (x_val == 1)
                k_t += (x_val == 0)

        elif is_ternary and np.isfinite(N):
            # Without Replacement (WoR) ternary data
            get_trunc_dim = lambda N: next((v for t, v in [(10000, 200), (50000, 250), (100000, 300), (250000, 500), (1000000, 1000)] if N <= t), 10000)
            matrix_trunc_dim = getattr(self, "matrix_trunc_dim", get_trunc_dim(N))
            
            audit_cutoff = len(x)
            if audit_cutoff == 0:
                return lamj
                
            # initialise coefficient arrays and arrays of incomplete beta functions
            zeta_vector = np.zeros(audit_cutoff)
            zeta_vector[0] = 1.0
            phi_vector = np.zeros(audit_cutoff)
            phi_vector[0] = 1.0
            
            I_Beta_array = np.zeros((matrix_trunc_dim + 1, matrix_trunc_dim + 1))
            I_Beta_array[0, 0] = sp.special.beta(a1, a2) * (1 - sp.stats.beta.cdf(t, a1, a2))
            
            ell_orig = np.arange(audit_cutoff)
            ell = ell_orig[:, None]
            emm = ell_orig[None, :]

            log2 = np.log(2)
            j_t, k_t, h_t = 0, 0, 0
            h_t_J, h_t_K = 0, 0
            mu_prod_log = 0.0
            S_j = 0.0  # Added for running total

            eta_t = (a1 + 0.5**(a1 + a2) / I_Beta_array[0, 0]) / (a1 + a2)

            for j, x_val in enumerate(x):
                N_remaining = self.N - j
                t_adj = (self.N * t - S_j) / N_remaining if N_remaining > 0 else t

                if t_adj <= 0.0 or t_adj >= 1.0:
                 break # default lam_j to 0 for future times to avoid division by 0


                lam_j = 0.0
                if eta_t > t_adj:
                    lam_j = (eta_t - t_adj) / (t_adj * (1 - t_adj))
                
                lamj[j] = lam_j
                S_j += x_val
                
                if not ((x_val == 0.5) and (t_adj == 0.5)):
                    I_Beta_array = _update_I_Beta_curr(
                        I_Beta_array, j_t, k_t, h_t_J, h_t_K, x_val, a1, a2, 
                        ell, emm, t_adj, mu_prod_log, log2, matrix_trunc_dim
                    )

                if x_val == 0.5:
                    if t_adj < 0.5:
                        zeta_vector = _update_zeta_array_row(zeta_vector, t_adj, h_t_J, use_scale=1)
                        h_t_J += 1
                    elif t_adj > 0.5:
                        phi_vector = _update_phi_array_row(phi_vector, t_adj, h_t_K, use_scale=1)
                        h_t_K += 1

                j_t += (x_val == 1)
                k_t += (x_val == 0)
                h_t += (x_val == 0.5)

                h_J = min(h_t_J, matrix_trunc_dim)
                h_K = min(h_t_K, matrix_trunc_dim)
                
                tmp = I_Beta_array[:h_J+1, :h_K+1] @ phi_vector[:h_K+1]
                Numerator_curr = zeta_vector[:h_J+1] @ tmp

                mu_prod_log += np.log(1 / (t_adj**(x_val == 1) * (1 - t_adj)**(x_val == 0)))
                
                I_Beta_array_assuming_1 = I_Beta_array.copy()
                I_Beta_array_assuming_1 = _update_I_Beta_curr(
                    I_Beta_array_assuming_1, j_t, k_t, h_t_J, h_t_K, 1, a1, a2, 
                    ell, emm, 1, mu_prod_log, log2, matrix_trunc_dim
                )
                
                tmp_1 = I_Beta_array_assuming_1[:h_J+1, :h_K+1] @ phi_vector[:h_K+1]
                Numerator_assuming_1 = zeta_vector[:h_J+1] @ tmp_1
                
                eta_t = Numerator_assuming_1 / Numerator_curr

        else:

            # General [0,U] TruncMix(Beta)

            scaled_t = t / u # rescale to [0,1]
            scaled_x = x / u

            dps_accuracy = getattr(self, "dps_accuracy", 8) 
            max_safe_dim = getattr(self, "max_safe_dim", 20000)
            audit_cutoff = getattr(self, "matrix_trunc_dim", len(scaled_x))

            log_I_Beta_array = _precompute_log_I_Beta(audit_cutoff, scaled_t, a1, a2, max_safe_dim) # generate matrix of log incomplete beta functions

            zeta_vector = np.zeros(audit_cutoff + 1)
            zeta_vector[0] = 1.0
            phi_vector = np.zeros(audit_cutoff + 1)
            phi_vector[0] = 1.0

            eta_t = np.exp(log_I_Beta_array[1, 0] - log_I_Beta_array[0, 0])
            h_t_1, h_t_2 = 0, 0
            S_j = 0.0
            prev_window = None

            for j, x_val in enumerate(scaled_x):
                
                if np.isfinite(self.N):
                    N_remaining = self.N - j
                    t_adj = (self.N * scaled_t - S_j) / N_remaining if N_remaining > 0 else scaled_t
                else:
                    t_adj = scaled_t

                if t_adj <= 0.0 or t_adj >= 1.0:
                    break # default lam_j to 0 for future times to avoid division by 0

                
                lam_j = 0.0
                if eta_t > t_adj:
                    lam_j = (eta_t - t_adj) / (t_adj * (1 - t_adj)) # calculate the bet based on current posterior mean (eta_t)
                
                lamj[j] = lam_j
                S_j += x_val

                # Prevent index out-of-bounds if audit_cutoff was restricted by matrix_trunc_dim
                if h_t_1 >= audit_cutoff or h_t_2 >= audit_cutoff:
                    continue

                # Update coefficient arrays
                if x_val != t_adj:
                    if t_adj < x_val:
                        zeta_vector, _ = _update_row_array_general(zeta_vector, t_adj, x_val, h_t_1, type='zeta')
                        h_t_1 += 1
                    elif t_adj > x_val:
                        phi_vector, _ = _update_row_array_general(phi_vector, t_adj, x_val, h_t_2, type='phi')
                        h_t_2 += 1

                # Get truncation window
                with np.errstate(divide='ignore'):
                    log_zeta = np.log(zeta_vector)
                    log_phi = np.log(phi_vector)

                L1, U1, L2, U2 = _get_sequential_truncation_window(log_zeta, log_phi, log_I_Beta_array, dps_accuracy=dps_accuracy, prev_window=prev_window)
                prev_window = L1, U1, L2, U2

                # Calculate next posterior mean (eta_{t+1})
                log_zeta_active = log_zeta[L1 : U1 + 1]
                log_phi_active  = log_phi[L2 : U2 + 1]
                
                matrix_chunk = log_I_Beta_array[L1 : U1 + 1, L2 : U2 + 1]
                combined_grid = log_zeta_active[:, None] + log_phi_active[None, :] + matrix_chunk

                shifted_chunk = log_I_Beta_array[L1 + 1 : U1 + 2, L2 : U2 + 1]
                combined_grid_shifted = log_zeta_active[:, None] + log_phi_active[None, :] + shifted_chunk
                
                log_numerator_sum = sp.special.logsumexp(combined_grid_shifted)
                eta_t = np.exp(log_numerator_sum - sp.special.logsumexp(combined_grid))

        return lamj

    
    def trunc_mix_beta_binomial(self, x: np.array, **kwargs) -> np.array:
        """
        TruncMix Beta-Binomial estimator for binary Without-Replacement (WoR) data.
        Requires finite N and strictly {0, 1} data. The upper bound u is ignored.
        """
        if np.isinf(self.N):
            raise ValueError("trunc_mix_beta_binomial requires a finite population size N (WoR).")
            
        # Data must be strictly binary {0, 1}
        if not np.all(np.isin(x, [0.0, 1.0])):
            raise ValueError("trunc_mix_beta_binomial requires strictly binary {0, 1} data.")
            
        t = self.t
        lamj = np.zeros(len(x))
        if len(x) == 0:
            return lamj
            
        a1 = getattr(self, "BB_prior_a1", 1.0)
        a2 = getattr(self, "BB_prior_a2", 1.0)
        
        # Setup PMF dynamically
        BB_pmf_array = _create_recursive_Beta_Binomial_array(self.N, (a1, a2))
        indices = np.arange(self.N + 1)
        C_BB = BB_pmf_array.sum()
        
        j_t, k_t = 0, 0
        S_j = 0.0
        
        # Initial bet before data
        eta_t = np.dot(indices / self.N, BB_pmf_array) / C_BB

        for j, x_val in enumerate(x):
            N_remaining = self.N - j
            t_adj = (self.N * t - S_j) / N_remaining if N_remaining > 0 else t

            if t_adj <= 0.0 or t_adj >= 1.0:
                break # default lam_j to 0 for future times to avoid division by 0

            # Calculate and store the bet
            lam_j = 0.0
            if eta_t > t_adj:
                lam_j = (eta_t - t_adj) / (t_adj * (1 - t_adj))
            lamj[j] = lam_j
            
            # Update state variables
            S_j += x_val
            j_t += (x_val == 1.0)
            k_t += (x_val == 0.0)
            
            BB_pmf_array, C_BB = _update_Beta_Binomial_array(BB_pmf_array, self.N, indices, a1, a2, j_t, k_t, x_val)
            
            # Calculate next step's expected mean
            if C_BB > 0 and (self.N - (j + 1)) > 0:
                eta_t = np.dot((indices - j_t) / (self.N - (j + 1)), BB_pmf_array) / C_BB
            else:
                eta_t = t_adj

        return lamj

    def trunc_mix_dirichlet(self, x: np.array, **kwargs) -> np.array:
        """
        TruncMix Dirichlet estimator for ternary Without-Replacement (WoR) data.
        Requires finite N and strictly {0, 0.5, 1} data. The upper bound u is ignored.
        """
        if np.isinf(self.N):
            raise ValueError("trunc_mix_dirichlet requires a finite population size N (WoR).")

        # Data must be strictly ternary {0, 0.5, 1}
        if not np.all(np.isin(x, [0.0, 0.5, 1.0])):
            raise ValueError("trunc_mix_dirichlet requires strictly ternary {0, 0.5, 1} data.")

        t = self.t
        lamj = np.zeros(len(x))
        if len(x) == 0:
            return lamj
            
        a_ones = getattr(self, "DT_prior_ones", 1.0)
        a_halves = getattr(self, "DT_prior_halves", 1.0)
        a_zeros = getattr(self, "DT_prior_zeros", 1.0)
        n_nodes = getattr(self, "DT_n_nodes", 15)

        j_t, k_t, u_t = 0, 0, 0
        S_j = 0.0

        for j, x_val in enumerate(x):
            N_remaining = self.N - j
            t_adj = (self.N * t - S_j) / N_remaining if N_remaining > 0 else t

            if t_adj <= 0.0 or t_adj >= 1.0:
                break # default lam_j to 0 for future times to avoid division by 0
            
            b_posterior = _calculate_truncated_dirichlet_mean_Gauss_Legendre(
                a_ones + j_t, a_zeros + k_t, a_halves + u_t, t_adj, n_nodes=n_nodes
            )
            
            mu_null = _calculate_RIP_null(b_posterior, j_t, k_t, N_remaining)
            
            lam_j = 0.0
            if not np.any(mu_null <= 0):
                lam_j = max(0.0, (b_posterior[0] / mu_null[0]) - (b_posterior[1] / mu_null[1]))
            lamj[j] = lam_j
            
            S_j += x_val
            j_t += (x_val == 1.0)
            k_t += (x_val == 0.0)
            u_t += (x_val == 0.5)

        return lamj


    def lam_to_eta(self, lam: np.array, mu: np.array) -> np.array:
        """
        Convert bets (lam) for betting martingale to their implied estimates of the mean, eta, for ALPHA

        Parameters
        ----------
        lam: float or numpy array
            the value(s) of lam (the fraction of the current fortune to bet on the next draw)
        mu: float or numpy array
            sequence of population mean(s) if the null is true, adjusted for values already seen

        Returns
        -------
        eta: float or numpy array
            the corresponding value(s) of the mean
        """
        return mu * (1 + lam * (self.u - mu))

    def eta_to_lam(self, eta: np.array, mu: np.array) -> np.array:
        """
        Convert eta for ALPHA to corresponding bet lam for the betting martingale parametrization

        Parameters
        ----------
        eta: float or numpy array
            the value(s) of lam (the fraction of the current fortune to bet on the next draw)
        mu: float or numpy array
            sequence of population mean(s) if the null is true, adjusted for values already seen

        Returns
        -------
        lam: float or numpy array
            the corresponding betting fractions
        """
        return (eta / mu - 1) / (self.u - mu)

    def kaplan_kolmogorov(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        p-value for the hypothesis that the mean of a nonnegative population with N
        elements is t. The alternative is that the mean is less than t.
        If the random sample x is in the order in which the sample was drawn, it is
        legitimate to set random_order = True.
        If not, set random_order = False.

        g is a tuning parameter to protect against data values equal to zero.
        g should be in [0, 1)

        Parameters
        ----------
        x: list
            observations
        N: int
            population size
        t: float
            null value of the population mean
        attributes used:
            g: float
                "padding" in case there any values in the sample are zero
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which can yield a
                more powerful test. See above.

        Returns
        -------
        p: float
           the p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        N = self.N
        t = self.t
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        x = np.array(x)
        assert all(x >= 0), "Negative value in a nonnegative population!"
        assert len(x) <= N, "Sample size is larger than the population!"
        assert N > 0, "Population size not positive!"
        assert N == int(N), "Non-integer population size!"

        _S, _Stot, _j, m = self.sjm(N, t+g, x+g)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms = np.cumprod((x + g) / m)
        terms[m < 0] = np.inf
        p = min((1 / np.max(terms) if random_order else 1 / terms[-1]), 1)
        return p, np.minimum(1 / terms, 1)

    def kaplan_markov(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Kaplan-Markov p-value for the hypothesis that the sample x is drawn IID from a population
        with mean t against the alternative that the mean is less than t.

        If there is a possibility that x has elements equal to zero, set g>0; otherwise, the p-value
        will be 1.

        If the order of the values in the sample is random, you can set random_order = True to use
        optional stopping to increase the power. If the values are not in random order or if you want
        to use all the data, set random_order = False

        Parameters:
        -----------
        x: array-like
            the sample
        attributes used:
            g: float
                "padding" so that if there are any zeros in the sample, the martingale doesn't vanish forever
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which can yield a
                more powerful test. See above.

        Returns:
        --------
        p: float
           the p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        t = self.t
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        if negs := sum(xx < 0 for xx in x) > 0:
            raise ValueError(
                "{negs} negative values in sample from a nonnegative population."
            )
        p_history = np.cumprod((t + g) / (x + g))
        return np.min(
            [1, np.min(p_history) if random_order else p_history[-1]]
        ), np.minimum(p_history, 1)

    def kaplan_wald(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Kaplan-Wald p-value for the hypothesis that the sample x is drawn IID from a population
        with mean t against the alternative that the mean is less than t.

        If there is a possibility that x has elements equal to zero, set g \in (0, 1);
        otherwise, the p-value will be 1.

        If the order of the values in the sample is random, you can set random_order = True to use
        optional stopping to increase the power. If the values are not in random order or if you want
        to use all the data, set random_order = False

        Parameters:
        -----------
        x: array-like
            the sample
        attributes used:
            g: float
                "padding" in case there any values in the sample are zero
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which
                can yield a more powerful test. See above.

        Returns:
        --------
        p: float
            p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.

        """
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        t = self.t
        if g < 0 or g > 1:
            raise ValueError(f"{g=}, but g must be between 0 and 1. ")
        if any(xx < 0 for xx in x):
            raise ValueError("Negative value in sample from a nonnegative population.")
        p_history = np.cumprod((1 - g) * x / t + g)
        return np.min(
            [1, 1 / np.max(p_history) if random_order else 1 / p_history[-1]]
        ), np.minimum(1 / p_history, 1)

    def wald_sprt(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds the p value for the hypothesis that the population
        mean is less than or equal to t against the alternative that it is eta,
        for a population of size N of values in the interval [0, u].

        Generalizes Wald's SPRT for the Bernoulli to sampling without replacement and to
        nonnegative bounded values rather than binary values (for which it is a valid test, but not the SPRT).
        See Stark, 2022. ALPHA: Audit that Learns from Previous Hand-Audited Ballots

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        If the sample is drawn without replacement, the data must be in random order

        Parameters
        ----------
        x: list
            values between 0 and u
        kwargs:
            eta: float in (0,u)
                alternative hypothesized population mean
            random_order: Boolean
                if the data are in random order, setting this to True can improve the power.
                If the data are not in random order, set to False

        Returns
        -------
        tuple:
            p: float
                p-value
            p_history: numpy array
                sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        u = self.u
        N = self.N
        t = self.t
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        random_order = getattr(self, "random_order", True)
        if any((xx < 0 or xx > u) for xx in x):
            raise ValueError(f"Data out of range [0,{u}]")
        if np.isfinite(N):
            if not random_order:
                raise ValueError(
                    "data must be in random order for samples without replacement"
                )
            S = np.insert(np.cumsum(x), 0, 0)[0:-1]  # 0, x_1, x_1+x_2, ...,
            j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
            m = (N * t - S) / (
                N - j + 1
            )  # mean of population after (j-1)st draw, if null is true
            etas = (N * eta - S) / (N - j + 1)
        else:
            m = t
            etas = eta
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms = np.cumprod(
                (x * etas / m + (u - x) * (u - etas) / (u - m)) / u
            )  # generalization of Bernoulli SPRT
        terms[m < 0] = np.inf  # the null is surely false
        terms = np.cumprod(terms)
        return 1 / np.max(terms) if random_order else 1 / terms[-1], np.minimum(
            1, 1 / terms
        )

    def sample_size(self, x: list = None, alpha: float = 0.05, reps: int = None, prefix: bool = False,
                    quantile: float = 0.5, **kwargs) -> int:
        """
        Estimate the sample size to reject the null hypothesis that the population mean of a population of size
        `N` is `<=t` at significance level `alpha`, using pilot data `x`.

        If `reps is None`, tiles copies of `x` to produce a list of length `N`.

        If `reps is not None`, samples at random from `x` to produce `reps` lists of length `N` and reports
        the `quantile` quantile of the resulting sample sizes.

        If `prefix == True`, starts every list with `x` as given, then samples from `x` to produce the
        remaining `N-len(x)` entries.

        Parameters
        ----------
        x: list or np.array
            the data for the simulation or calculation
        alpha: float in (0, 1/2)
            the significance level for the test
        reps: int
            the number of replications to use to simulate the sample size. If `reps is None`, estimates deterministically
        prefix: bool
            whether to use `x` as the prefix of the data sequence. Not used if `reps is None`
        quantile: float in (0, 1)
            desired quantile of the sample sizes from simulations. Not used if `reps is None`
        kwargs:
            keyword args passed to self.test()

        Returns
        -------
        sam_size: int
            estimated sample size
        """
        N = self.N
        if reps is None:
            pop = np.repeat(np.array(x), math.ceil(N / len(x)))[
                0:N
            ]  # tile data to make the population
            p = self.test(pop, **kwargs)[1]
            crossed = p <= alpha
            sam_size = int(N if np.sum(crossed) == 0 else (np.argmax(crossed) + 1))
        else:  # estimate the quantile by a bootstrap-like simulation
            seed = kwargs.get("seed", 1234567890)
            prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
            sams = np.zeros(int(reps))
            pfx = np.array(x) if prefix else []
            ran_len = (N - len(x)) if prefix else N
            for r in range(reps):
                pop = np.append(pfx, prng.choice(x, size=ran_len, replace=True))
                p = self.test(pop, **kwargs)[1]
                crossed = p <= alpha
                sams[r] = N if np.sum(crossed) == 0 else (np.argmax(crossed) + 1)
            sam_size = int(np.quantile(sams, quantile))
        return sam_size
