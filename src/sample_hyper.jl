# ============================================================================
# Stima degli iperparametri della (S)Bartlett:  nu  e  psi  (S = psi * I)
#
# Convenzioni (come nel resto del pacchetto):
#   - nu   = gdl del paper:  b_kk^2 ~ chi^2(nu + z_k);  Wishart df = nu + k - 1
#   - psi  = scala di S = psi * I (psi_scal_mcmc in mcmc);  la matrice psi_mcmc e' chol(S) = sqrt(psi) * I
#   - Lambda = psi * Lambda_1(B), con Lambda_1 la matrice costruita con psi = 1
#
# Nello stato dell'MCMC nu e' salvato come df della Wishart (nu_df = nu + k - 1),
# cioe' lo stesso valore che prima veniva letto da params(prior_lambda)[1].
# Le funzioni vengono chiamate da `mcmc` solo se estimate_nu / estimate_psi = true;
# altrimenti nu e psi restano fissi al valore della prior_lambda.
# Vale solo per GenSparse_V2 (varianze marginali = 1) e psi diagonale costante.
# ============================================================================

# numero di entrate libere (z = 1) sotto la diagonale nella colonna j
function n_free_col(z_mat::Matrix{Float64}, j::Int64)
    k = size(z_mat, 1)
    zj = 0.0
    for i = (j+1):k
        zj += z_mat[i, j]
    end
    return zj
end

# log prior (in b, non in log b) delle diagonali b_kk ~ Chi(nu + z_k)
function log_prior_bdiag(nu::Float64, b_mat::Matrix{Float64}, z_mat::Matrix{Float64})
    k = size(b_mat, 1)
    out = 0.0
    for j = 1:k
        d = nu + n_free_col(z_mat, j)
        b = b_mat[j, j]
        out += (d - 1.0) * log(b) - 0.5 * b^2 - (0.5 * d - 1.0) * log(2.0) - loggamma(0.5 * d)
    end
    return out
end

# imposta psi (scalare) e ricalcola Q, Q sparsa e Lambda dallo stato B corrente
function set_psi_and_recompute!(psi_scal::Float64, psi::Matrix{Float64}, psi_inv::Matrix{Float64},
    b_mat::Matrix{Float64}, q_mat::Matrix{Float64}, q_mat_sparse::Matrix{Float64},
    lambda::Matrix{Float64}, z_mat::Matrix{Float64})

    k = size(psi, 1)
    s = sqrt(psi_scal)
    for i = 1:k
        psi[i, i] = s
        psi_inv[i, i] = 1.0 / s
    end
    from_bmat_to_all_q(b_mat, q_mat, q_mat_sparse, z_mat, psi)
    mul!(lambda, q_mat_sparse, transpose(q_mat_sparse))
    return nothing
end

# ricostruisce bstar (log sulla diagonale) da b_mat, stesso ordine di from_bstar_to_b_mat
function from_b_mat_to_bstar!(b_mat::Matrix{Float64}, bstar::Vector{Float64})
    k = size(b_mat, 1)
    h = 0
    for j = 1:k
        for i = j:k
            h += 1
            bstar[h] = (i == j) ? log(b_mat[i, j]) : b_mat[i, j]
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# nu | B, Z, D : MH random walk su log(nu). Lambda non cambia => niente likelihood.
# ---------------------------------------------------------------------------
function sample_nu!(nu_df_mcmc::Vector{Float64}, b_mat::Matrix{Float64}, z_mat::Matrix{Float64},
    prior_nu::ContinuousUnivariateDistribution, sd_log_nu::Float64, acc::Vector{Int64})

    k = size(b_mat, 1)
    nu_old = nu_df_mcmc[1] - k + 1          # nu del paper
    nu_new = nu_old * exp(sd_log_nu * randn())
    log_ratio = log_prior_bdiag(nu_new, b_mat, z_mat) - log_prior_bdiag(nu_old, b_mat, z_mat) +
                logpdf(prior_nu, nu_new) - logpdf(prior_nu, nu_old) +
                log(nu_new) - log(nu_old)                       # jacobiano RW su log(nu)
    if !isnan(log_ratio) && log(rand()) < log_ratio
        nu_df_mcmc[1] = nu_new + k - 1     # di nuovo come df della Wishart
        acc[1] += 1
    end
    return nothing
end

# ---------------------------------------------------------------------------
# psi | Y, B, Z, D : Gibbs coniugato.  Lambda = psi * Lambda_1(B),
# likelihood  psi^(n k / 2) exp(-psi R1 / 2),  R1 = sum_i y_i' Lambda_1 y_i,
# prior psi ~ Gamma(shape a, scale theta)  =>  Gamma(a + n k / 2, rate 1/theta + R1 / 2)
# ---------------------------------------------------------------------------
function sample_psi_gibbs!(psi_scal::Vector{Float64}, ycentered::Matrix{Float64}, prior_psi::Gamma{Float64},
    psi::Matrix{Float64}, psi_inv::Matrix{Float64}, b_mat::Matrix{Float64}, q_mat::Matrix{Float64},
    q_mat_sparse::Matrix{Float64}, lambda::Matrix{Float64}, z_mat::Matrix{Float64})

    k = size(ycentered, 1)
    n = size(ycentered, 2)
    R = 0.0
    for i = 1:n
        yi = view(ycentered, :, i)
        R += dot(yi, lambda, yi)
    end
    R1 = R / psi_scal[1]
    a, theta = params(prior_psi)
    shape_post = a + 0.5 * n * k
    rate_post = 1.0 / theta + 0.5 * R1
    psi_scal[1] = rand(Gamma(shape_post, 1.0 / rate_post))
    set_psi_and_recompute!(psi_scal[1], psi, psi_inv, b_mat, q_mat, q_mat_sparse, lambda, z_mat)
    return nothing
end

# ---------------------------------------------------------------------------
# Mossa "sulla cresta": psi -> c psi,  b attive -> b / sqrt(c).  Lambda invariata,
# la likelihood si cancella. Rapporto = prior(psi) x prior(b attive) x Jacobiano
# Jacobiano: c (per psi) * c^(-m/2) (m = numero di b attive), proposta log c ~ N(0, sd^2).
# Le b inattive non entrano in Lambda e restano invariate.
# ---------------------------------------------------------------------------
function sample_psi_ridge!(psi_scal::Vector{Float64}, nu::Float64, bstar::Vector{Float64},
    b_mat::Matrix{Float64}, z_mat::Matrix{Float64}, model_sparse::GenSparse_V2,
    prior_psi::Gamma{Float64}, sd_log_psi::Float64, acc::Vector{Int64},
    psi::Matrix{Float64}, psi_inv::Matrix{Float64}, q_mat::Matrix{Float64},
    q_mat_sparse::Matrix{Float64}, lambda::Matrix{Float64})

    k = size(b_mat, 1)
    logc = sd_log_psi * randn()
    c = exp(logc)
    sc = 1.0 / sqrt(c)
    psi_new = psi_scal[1] * c

    log_ratio = logpdf(prior_psi, psi_new) - logpdf(prior_psi, psi_scal[1]) + logc
    m = 0
    for j = 1:k
        d = nu + n_free_col(z_mat, j)
        b = b_mat[j, j]
        bn = b * sc
        log_ratio += (d - 1.0) * (log(bn) - log(b)) - 0.5 * (bn^2 - b^2)
        m += 1
        for i = (j+1):k
            if z_mat[i, j] == 1.0
                b = b_mat[i, j]
                bn = b * sc
                log_ratio += logpdf(model_sparse.distr, bn) - logpdf(model_sparse.distr, b)
                m += 1
            end
        end
    end
    log_ratio += -0.5 * m * logc

    if !isnan(log_ratio) && log(rand()) < log_ratio
        psi_scal[1] = psi_new
        for j = 1:k
            b_mat[j, j] *= sc
            for i = (j+1):k
                if z_mat[i, j] == 1.0
                    b_mat[i, j] *= sc
                end
            end
        end
        from_b_mat_to_bstar!(b_mat, bstar)
        set_psi_and_recompute!(psi_scal[1], psi, psi_inv, b_mat, q_mat, q_mat_sparse, lambda, z_mat)
        acc[1] += 1
    end
    return nothing
end

# blocco unico chiamato da mcmc ad ogni iterazione (se almeno un flag e' true)
function update_hyper!(estimate_nu::Bool, estimate_psi::Bool,
    nu_df_mcmc::Vector{Float64}, psi_scal::Vector{Float64},
    prior_nu::ContinuousUnivariateDistribution, prior_psi::Gamma{Float64},
    sd_log_nu::Float64, sd_log_psi::Float64,
    acc_nu::Vector{Int64}, acc_psi::Vector{Int64},
    ycentered::Matrix{Float64}, bstar::Vector{Float64}, b_mat::Matrix{Float64},
    z_mat::Matrix{Float64}, model_sparse::GenSparse_V2,
    psi::Matrix{Float64}, psi_inv::Matrix{Float64}, q_mat::Matrix{Float64},
    q_mat_sparse::Matrix{Float64}, lambda::Matrix{Float64})

    # b_mat allineata a bstar (per sicurezza)
    from_bstar_to_b_mat(bstar, b_mat)

    if estimate_psi
        sample_psi_gibbs!(psi_scal, ycentered, prior_psi, psi, psi_inv, b_mat, q_mat, q_mat_sparse, lambda, z_mat)
        sample_psi_ridge!(psi_scal, nu_df_mcmc[1] - size(b_mat, 1) + 1, bstar, b_mat, z_mat, model_sparse, prior_psi, sd_log_psi, acc_psi,
            psi, psi_inv, q_mat, q_mat_sparse, lambda)
    end
    if estimate_nu
        sample_nu!(nu_df_mcmc, b_mat, z_mat, prior_nu, sd_log_nu, acc_nu)
    end
    return nothing
end

update_hyper!(estimate_nu::Bool, estimate_psi::Bool, nu_df_mcmc, psi_scal, prior_nu, prior_psi, sd_log_nu, sd_log_psi,
    acc_nu, acc_psi, ycentered, bstar, b_mat, z_mat, model_sparse::TypeSparse, psi, psi_inv, q_mat, q_mat_sparse, lambda) =
    error("Stima di nu/psi implementata solo per GenSparse_V2")

