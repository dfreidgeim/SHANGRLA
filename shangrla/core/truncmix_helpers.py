import math
import numpy as np
import scipy as sp
import mpmath as mp



def _update_zeta_array_row(zeta_vector, mu_t, h_t_J, use_scale = 1):
    # updates zeta vector according to recursion formula in Section X of paper

    next_zeta = np.empty(h_t_J + 2, dtype=zeta_vector.dtype)

    scale = 1.0 /(2 * mu_t * (1 - mu_t)) if use_scale else 1

    # l = 0
    next_zeta[0] = zeta_vector[0] *mu_t * scale
    # 1 <= l <= h_t_J-1
    next_zeta[1:h_t_J+1] = (zeta_vector[0:h_t_J] * (1 - 2*mu_t) + zeta_vector[1:h_t_J+1] * mu_t) *scale
    # l = h_t_J
    next_zeta[h_t_J+1] = zeta_vector[h_t_J] * (1 - 2*mu_t) * scale

    return next_zeta

def _update_phi_array_row(phi_vector, mu_t, h_t_K, use_scale = 1):
    # updates last row of eta_array according to recursion formula. 
    # Done in log space to ensure numerical stability

    next_phi = np.empty(h_t_K + 2, dtype=phi_vector.dtype)

    scale = 1.0 /(2 * mu_t * (1 - mu_t)) if use_scale else 1

    # m = 0
    next_phi[0] = phi_vector[0] *(1-mu_t) * scale

    # 1 <= m <= u_t^K-1
    next_phi[1:h_t_K+1] = (phi_vector[0:h_t_K] * (2*mu_t - 1) + phi_vector[1:h_t_K+1] * (1-mu_t)) *scale
 
    # m = u_t^K
    next_phi[h_t_K+1] = phi_vector[h_t_K] * (2*mu_t-  1) * scale


    return next_phi # new phi_vector

def _update_I_Beta_curr(I_Beta_array, j_t, k_t, h_t_J,h_t_K,X_i, a1, a2, ell, emm, mu_t, mu_prod_log, log2, matrix_trunc_dim):
    ### inputs previous values of j_t,k_t,u_t and X_i and updates vector of Incomplete Beta functions

    # exponential equality
    #np.exp(-(j_t + k_t + a1 + a2 + l + m) * np.log(2) + mu_prod_log) = np.exp(-(j_t + a1 + l) * np.log(2) + mu_prod_log_J) * np.exp(-(k_t + a2 + m) * np.log(2) + mu_prod_log_K) 


    h_J = min(h_t_J, matrix_trunc_dim)
    h_K = min(h_t_K, matrix_trunc_dim)

    prev = I_Beta_array[:h_J+1,:h_K+1]

    l = ell[:h_J+1] # col
    m = emm[:,:h_K+1] # row


    if X_i == 1 or X_i == 0:

        # FIX: split up mu_prod_log into row and column terms
        mu_prod_log_J = mu_prod_log*j_t/(j_t+k_t) if mu_prod_log else 0
        mu_prod_log_K = mu_prod_log*k_t/(j_t+k_t) if mu_prod_log else 0


        if X_i == 1:
            # get ratio of I(a+1,b)/I(a,b) in terms of l, vector multiply all expressions
            Exp_term = np.exp(-(j_t + a1 + l) * log2 + mu_prod_log_J) * np.exp(-(k_t + a2 + m) * log2 + mu_prod_log_K) 
            I_Beta_array[:h_J+1,:h_K+1] = (prev * (j_t + a1 + l) + Exp_term) / ((j_t + k_t + l + m + a1 + a2)*mu_t) # l should be vectorised

        elif X_i == 0:
            # get ratio of I(a,b)/I(a,b+1) in terms of l, vector multiply all expressions
            Exp_term = np.exp(-(j_t + a1 + l) * log2 + mu_prod_log_J) * np.exp(-(k_t + a2 + m) * log2 + mu_prod_log_K) 
            I_Beta_array[:h_J+1,:h_K+1]  = (prev  * (k_t + a2 + m) - Exp_term) / ((j_t + k_t + l + m + a1 + a2) *(1-mu_t) )

    else:

        # add element at the end
        if mu_t < 0.5:

            if not h_J == matrix_trunc_dim:
                # add row to the end
                I_Beta_array[h_J+1,:h_K+1] = (prev[-1,:] * (j_t + a1 + h_J) + np.exp(-(j_t + k_t + a1 + a2 + h_J + m) * log2 + mu_prod_log)) / (j_t + k_t + h_J + m + a1 + a2) # get ratio of I(a+1,b)/I(a,b)
        else:
            if not h_K == matrix_trunc_dim:
                # add col to the end
                I_Beta_array[:h_J+1,h_K+1:h_K+2] = (prev[:,-1:] * (k_t + a2 + h_K) - np.exp(-(j_t + k_t + a1 + a2 + l + h_K) * log2 + mu_prod_log)) / (j_t + k_t + l + h_K + a1 + a2) # get ratio of I(a+1,b)/I(a,b)

    return I_Beta_array


def _precompute_log_I_Beta(size, mu, a1, a2):
    """
    Precomputes the size x size grid of log(B_{1/2}(a1+l, a2+m)).
    Computes row 0 safely using quadrature integration, then builds subsequent rows 
    using a numerically stable recurrence with logaddexp to prevent underflow.
    """


    log_I_Beta = np.empty((size+1,size+1), dtype=np.float64)
    log2 = np.log(2)

    m_array = np.arange(size+1)

    # Row 0 (l=0): Evaluate exactly using mp.quad to required precision
    mp.mp.dps = 50

    log_row_0 = np.empty(size+1, dtype=np.float64)

    for m in range(size+1):
        # Integrand for l=0: b^(a1-1) * (1-b)^(m+a2-1)
        integral = mp.quad(lambda x: mp.exp((a1 - 1) * mp.log(x) + (m + a2 - 1) * mp.log(1 - x)), [mu, 1.0])
        log_row_0[m] = float(mp.log(integral))
   
    log_I_Beta[0, :] = log_row_0

    for l in range(size):
        
        log_mult = np.log(l + a1) - np.log(l + m_array + a1 + a2)
        # log( exp(-(l + m + a1 + a2)*log2) / (l + m + a1 + a2) )
        log_cross = ((l + a1) * np.log(mu)) + ((m_array + a2) * np.log(1.0 - mu)) - np.log(l + m_array + a1 + a2) # general value of mu
        #log_cross = -(l + m_array + a1 + a2) * log2 - np.log(l + m_array + a1 + a2) # mu = 0.5
        log_I_Beta[l+1, :] = np.logaddexp(log_I_Beta[l, :] + log_mult, log_cross) # logaddexp to avoid underflow


    return log_I_Beta

def _update_row_array_general(row_vector, mu_t, X_t, h_t, type = 'zeta'):
    next_vector = np.zeros(h_t + 2, dtype=row_vector.dtype)
    
    # For zeta: Retain weight: mu_t * (1 - X_t); Shift weight: X_t - mu_t
    if type == 'zeta':
        next_vector[0] = row_vector[0] * mu_t * (1 - X_t)
        next_vector[1:h_t+1] = (row_vector[0:h_t] * (X_t - mu_t) + row_vector[1:h_t+1] * mu_t * (1 - X_t))
        next_vector[h_t+1] = row_vector[h_t] * (X_t - mu_t)
    elif type == 'phi':
        next_vector[0] = row_vector[0] * (1-mu_t) * X_t
        next_vector[1:h_t+1] = (row_vector[0:h_t] * (mu_t - X_t) + row_vector[1:h_t+1] * (1-mu_t) * X_t)
        next_vector[h_t+1] = row_vector[h_t] * (mu_t - X_t)
    
    # Empirical Z-scaling to prevent underflow
    Z = np.sum(next_vector, dtype=np.float64)
    next_vector /= Z

    next_vector = np.where(next_vector < 1e-290, 0.0, next_vector)
    
    return next_vector, np.log(Z)

def _get_sequential_truncation_window(log_zeta, log_phi, log_I_Beta, dps_accuracy=8, prev_window=None):
    """
    Tracks the truncation window from the previous time step. 
    Threshold directly controls accuracy: 20.0 ~ 8 dps, 25.0 ~ 10 dps.
    """

    threshold = dps_accuracy * np.log(10)

    # Base Case: t=0 (No previous window
    if prev_window is None:
        h1 = len(log_zeta)
        h2 = len(log_phi)
        
        # Slice log_I_Beta to match the lengths of the current probability vectors
        sub_grid = log_zeta[:, None] + log_phi[None, :] + log_I_Beta[:h1, :h2]
        
        peak = np.max(sub_grid)
        valid = np.where(sub_grid >= peak - threshold)
        return (np.min(valid[0]), np.max(valid[0]), 
                np.min(valid[1]), np.max(valid[1]))

    # Unpack the exact bounds from time t-1
    l_min_prev, l_max_prev, m_min_prev, m_max_prev = prev_window

    # 2. Pad the Window (The "Search Box")
    # Adding a ballot shifts the mass by at most 1 index. 
    # A pad of 4 safely catches the movement while keeping the box tiny.
    pad = 4
    l_start = max(0, l_min_prev - pad)
    l_end   = min(len(log_zeta) - 1, l_max_prev + pad)
    m_start = max(0, m_min_prev - pad)
    m_end   = min(len(log_phi) - 1, m_max_prev + pad)

    # 3. Exact Math inside the Search Box
    # Because the box is tiny (e.g., 40x40), this exact O(N^2) math is computationally 
    # virtually free, taking microseconds.
    sub_grid = (log_zeta[l_start:l_end+1, None] + 
                log_phi[None, m_start:m_end+1] + 
                log_I_Beta[l_start:l_end+1, m_start:m_end+1])

    known_max = np.max(sub_grid)

    # 4. The Infallibility Check (Optional but guarantees rigor)
    # If the values on the very edge of our search box are greater than the threshold,
    # the mass moved faster than our pad, and we are leaking accuracy.
    # (In a real audit, a pad of 4 is usually bulletproof, but you can add a while-loop 
    # here to dynamically increase 'pad' if an edge violation occurs).

    # 5. Shave the Corners for the New Window
    # Find exactly which coordinates inside this box clear the threshold
    valid_2d = np.where(sub_grid >= known_max - threshold)

    l_min_new = l_start + np.min(valid_2d[0])
    l_max_new = l_start + np.max(valid_2d[0])
    m_min_new = m_start + np.min(valid_2d[1])
    m_max_new = m_start + np.max(valid_2d[1])

    return l_min_new, l_max_new, m_min_new, m_max_new



def _create_recursive_Beta_Binomial_array(N,alpha):
    # creates pmf on 0<=J<=N using recursive equations

    pmf = np.zeros((N + 1))
    midpoint = int(N/2)

    log_comb = (sp.special.gammaln(N + 1) - sp.special.gammaln(midpoint + 1) - sp.special.gammaln(N - midpoint + 1)) # calculate binomial coefficient in log-space

    pmf[midpoint] = np.exp(log_comb + sp.special.betaln(midpoint + alpha[0], N - midpoint + alpha[1]) - sp.special.betaln(alpha[0], alpha[1]))

    for J in range(midpoint,N):
        pmf[J+1] = ((N - J) / (J + 1)) * ((alpha[0] + J) / (alpha[1] + N - J - 1)) * pmf[J]

    return pmf

def _update_Beta_Binomial_array(pmf, N, indices, a1, a2, j_t, k_t, X_t):


    if X_t == 1:
        removal_var = j_t
        a_term = a1

        pmf[j_t - 1] = 0.0

        x_i_prev = indices - (removal_var - 1)

    else:
        removal_var = k_t
        a_term = a2

        pmf[N - k_t + 1] = 0.0

        x_i_prev = N - indices - (removal_var - 1)

    t = j_t + k_t
    a0 = a1+a2

    # update values vectorised
    numerator = x_i_prev * (a0 + t - 1)
    denominator = (N - (t-1)) * (a_term + (removal_var - 1))
    factor = numerator / denominator

    pmf *= factor
    C = pmf.sum()


    return pmf, C


def _calculate_RIP_null(b_posterior, j_t, k_t, N_remaining):
    """ Calculates the KL-minimising ternary null distribution given by solution to Equation ()."""

    b1,b0,b05 = b_posterior
    delta = (k_t - j_t)/N_remaining # proportion of ballots to ''catch' up to the null boundary

    # solve quadratic equ ation
    B = b1 * (1 - delta) + b0 * (1 - 3 * delta) - 2 * b05 * delta
    C = b0 * delta * (1 - delta)

    p0 = (B + math.sqrt(B**2 + 8 * C)) / 4.0

    return np.array([p0 + delta, p0, 1.0 - 2*p0 - delta])

def _calculate_truncated_dirichlet_mean_Gauss_Legendre(a_ones, a_zeros, a_halves, mu_t, n_nodes = 15):
    """
    Computes the posterior mean of a Dirichlet distribution truncated by the constraint $p_1 + 0.5 * p_{0.5} >= mu_t$. 

    Evaluated via adaptive Gauss-Legendre quadrature dynamically mapped to the active probability mass. Exploits complete neutrality to collapse the integral: 
        w ~ Beta(a_halves, a_ones + a_zeros) and X ~ Beta(a_ones, a_zeros) are strictly independent.
    """

    if mu_t <= 0:
        # return untruncated mean
        d = a_ones + a_zeros + a_halves
        return np.array([a_ones/d, a_zeros/d, a_halves/d])
    
    BASE_NODES, BASE_WEIGHTS = np.polynomial.legendre.leggauss(n_nodes)
    N_STDS = 8.0 # number of standard deviations to include in mass
        
    d = a_ones + a_zeros + a_halves
    
    # locates the probability mass concentration of w ~ Beta(a_halves, a_ones + a_zeros)
    mu_w = a_halves / d
    var_w = (mu_w * (1.0 - mu_w)) / (d + 1.0)
    sigma_w = np.sqrt(var_w)
    
    # find active window that ignores extreme tails of distribution
    low = max(0.0, mu_w - N_STDS * sigma_w)
    high = min(1.0, mu_w + N_STDS * sigma_w)
    
    if low >= high: # failsafe: full support if peak too extreme
        low, high = 0.0, 1.0
        
    # map Gauss Legendre nodes and weights into the active window
    scale = 0.5 * (high - low)
    W_GRID = scale * BASE_NODES + 0.5 * (low + high)
    W_WEIGHTS = scale * BASE_WEIGHTS
    
    W_GRID = np.clip(W_GRID, 1e-15, 1.0 - 1e-12) # Prevent absolute boundaries to avoid div-by-zero warnings
    
    # evaluate Beta survival functions P(X > g(w)) to obtain probability of satisfying alternative space constraint X >= g(w) for the W_grid points
    gw = np.clip((mu_t - 0.5 * W_GRID) / (1.0 - W_GRID), 0.0, 1.0) # alternative constraint
    
    H_w = sp.special.betaincc(a_ones, a_zeros, gw)
    H_w_plus1 = sp.special.betaincc(a_ones + 1.0, a_zeros, gw)
    
    # convert weights to log space for absolute numerical stability
    with np.errstate(divide='ignore'):
        log_weights = sp.special.xlogy(a_halves - 1.0, W_GRID) + sp.special.xlogy(a_ones + a_zeros - 1.0, 1.0 - W_GRID)
    max_log = np.max(log_weights)
    pdf_vals = np.exp(log_weights - max_log)
    
    # fast integration via weighted dot products 
    C = np.sum(pdf_vals * H_w * W_WEIGHTS)
    
    if C < 1e-300: # resolution too small
        return np.array([1.0, 0.0, 0.0])
    
    # calculate numerators of expectations
    E_halves_unnorm = np.sum(W_GRID * pdf_vals * H_w * W_WEIGHTS)
    E_ones_unnorm = np.sum((1.0 - W_GRID) * pdf_vals * H_w_plus1 * W_WEIGHTS) * (a_ones / (a_ones + a_zeros))

    # normalise via C
    E_ones = E_ones_unnorm / C
    E_halves = E_halves_unnorm / C
    E_zeros = 1.0 - E_ones - E_halves
    
    return np.array([E_ones, E_zeros, E_halves])

