# ============================================================================
# Mossa congiunta su (z_ij, b_ij) per ogni coppia i > j (riga i, colonna j).
#
# Al posto del Gibbs su z_ij con b_ij fisso (sample_zeta!), si propone insieme
# il nuovo z_ij e il nuovo b_ij:
#   - accensione (0 -> 1): b' ~ g1 = N(mean_b, 1/prec_b), approssimazione gaussiana
#     della condizionata di b_ij se l'arco fosse attivo, ottenuta dal solo termine
#     della colonna j della likelihood (quadratico in q_ij) e da un prior N(0, var(distr));
#   - spegnimento (1 -> 0): b' ~ prior (model_sparse.distr), cioe' la distribuzione
#     delle b inattive.
# g1 dipende solo dal resto dello stato (non da z_ij, b_ij), quindi la mossa
# e' reversibile. Rapporto di MH (L = likelihood esatta, chi = prior di b_jj):
#   on : L1(b')/L0 * pi/(1-pi) * chi(d0+1)/chi(d0) * p(b')/g1(b')
#   off: L0/L1(b) * (1-pi)/pi  * chi(d0)/chi(d0+1) * g1(b)/p(b)
# La proposta g1 serve solo a rendere plausibili le accensioni: la likelihood
# nel rapporto e' quella esatta (incluso l'effetto sui q* delle colonne successive).
#
# Usata da `mcmc` solo con joint_z_move = true; sample_zeta! resta invariata.
# Solo GenSparse_V2.
# ============================================================================

# posizione di b_ij (i >= j) nel vettore bstar (stesso ordine di from_bstar_to_b_mat)
function bstar_index(i::Int64, j::Int64, k::Int64)
    return (j - 1) * (k + 1) - div((j - 1) * j, 2) + (i - j) + 1
end

# log densita' (in b) di b ~ Chi(d), costanti dipendenti da d incluse
function log_chi_b(b::Float64, d::Float64)
    return (d - 1.0) * log(b) - 0.5 * b^2 - (0.5 * d - 1.0) * log(2.0) - loggamma(0.5 * d)
end

function sample_zeta_joint!(bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::GenSparse_V2,
    psi::Matrix{Float64}, nu_orig::Float64, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64},
    lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, ycentered::Matrix{Float64},
    prob_zeta_mcmc::Matrix{Float64}, var_mcmc::Vector{Float64}, pairsvec::Vector{Tuple{Int64,Int64}},
    acc::Vector{Int64})
    # acc = [accettate on, proposte on, accettate off, proposte off]

    k::Int64 = size(z_mcmc, 1)
    n::Int64 = size(ycentered, 2)
    nu::Float64 = nu_orig - k + 1
    ei::Vector{Float64} = zeros(Float64, k)
    app_x::Vector{Float64} = zeros(Float64, k)
    v0::Float64 = Distributions.var(model_sparse.distr)

    # numero di entrate libere per colonna
    n_uno::Vector{Float64} = zeros(Float64, k)
    for jj = 1:(k-1)
        for ii = (jj+1):k
            n_uno[jj] += z_mcmc[ii, jj]
        end
    end

    bmat::Matrix{Float64} = zeros(Float64, k, k)
    from_bstar_to_bmat(bstar_mcmc, bmat, k, n)
    qmat::Matrix{Float64} = zeros(Float64, k, k)
    from_bmat_to_qmat(bmat, qmat, k, n, psi)
    qstar::Matrix{Float64} = zeros(Float64, k, k)
    from_qmat_to_qstar_init(qmat, qstar, k, n, psi, z_mcmc)
    qstar_prop::Matrix{Float64} = copy(qstar)
    z_prop::Matrix{Float64} = copy(z_mcmc)

    like_acc::Vector{Float64} = zeros(Float64, k)
    compute_like_for_z_all(1, n, qstar, like_acc, z_mcmc, k, ycentered, model_sparse, 1, ei, app_x, var_mcmc)
    like_prop::Vector{Float64} = copy(like_acc)

    shuffle!(pairsvec)
    for (i, j) in pairsvec          # i > j

        s = psi[i, i]

        # --- proposta g1 per b_ij attivo: dipende solo dal resto dello stato ---
        P = 0.0
        cross = 0.0
        @inbounds for obs = 1:n
            a = 0.0
            for r = j:k
                if r != i
                    a += ycentered[r, obs] * qstar[r, j]
                end
            end
            yi = ycentered[i, obs]
            P += yi * yi
            cross += a * yi
        end
        prec_b = s^2 * P + 1.0 / v0
        mean_b = -s * cross / prec_b
        sd_b = 1.0 / sqrt(prec_b)
        g1 = Normal(mean_b, sd_b)

        # --- termini di prior che cambiano con z_ij ---
        d0 = nu + n_uno[j] - z_mcmc[i, j]        # gdl di b_jj senza l'arco (i,j)
        d1 = d0 + 1.0
        bjj = bmat[j, j]
        log_odds_z = log(prob_zeta_mcmc[i, j]) - log(1.0 - prob_zeta_mcmc[i, j])
        h = bstar_index(i, j, k)
        b_old = bmat[i, j]

        if z_mcmc[i, j] == 0.0
            # ---------------- accensione ----------------
            b_new = rand(g1)
            z_prop[i, j] = 1.0
            qmat[i, j] = s * b_new
            from_qmat_to_qstar_triangularindex(i, j, qmat, qstar_prop, k, n, psi, z_prop)
            compute_like_for_z_all(j, n, qstar_prop, like_prop, z_prop, k, ycentered, model_sparse, j, ei, app_x, var_mcmc)
            log_alpha = sum(like_prop) - sum(like_acc) + log_odds_z +
                        log_chi_b(bjj, d1) - log_chi_b(bjj, d0) +
                        logpdf(model_sparse.distr, b_new) - logpdf(g1, b_new)
            acc[2] += 1
            if !isnan(log_alpha) && log(rand()) < log_alpha
                acc[1] += 1
                z_mcmc[i, j] = 1.0
                n_uno[j] += 1.0
                bmat[i, j] = b_new
                bstar_mcmc[h] = b_new
                qstar .= qstar_prop
                like_acc .= like_prop
            else
                z_prop[i, j] = 0.0
                qmat[i, j] = s * b_old
                qstar_prop .= qstar
                like_prop .= like_acc
            end
        else
            # ---------------- spegnimento ----------------
            b_new = rand(model_sparse.distr)
            z_prop[i, j] = 0.0
            from_qmat_to_qstar_triangularindex(i, j, qmat, qstar_prop, k, n, psi, z_prop)
            compute_like_for_z_all(j, n, qstar_prop, like_prop, z_prop, k, ycentered, model_sparse, j, ei, app_x, var_mcmc)
            log_alpha = sum(like_prop) - sum(like_acc) - log_odds_z +
                        log_chi_b(bjj, d0) - log_chi_b(bjj, d1) +
                        logpdf(g1, b_old) - logpdf(model_sparse.distr, b_old)
            acc[4] += 1
            if !isnan(log_alpha) && log(rand()) < log_alpha
                acc[3] += 1
                z_mcmc[i, j] = 0.0
                n_uno[j] -= 1.0
                bmat[i, j] = b_new
                bstar_mcmc[h] = b_new
                qmat[i, j] = s * b_new
                qstar .= qstar_prop
                like_acc .= like_prop
            else
                z_prop[i, j] = 1.0
                qstar_prop .= qstar
                like_prop .= like_acc
            end
        end
    end

    # stato finale coerente (come alla fine di sample_zeta!)
    from_bstar_to_b_mat(bstar_mcmc, b_mat_mcmc)
    from_bmat_to_all_q(b_mat_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, z_mcmc, psi)
    mul!(lambda_mcmc, q_mat_sparse_mcmc, transpose(q_mat_sparse_mcmc))
    return nothing
end

sample_zeta_joint!(bstar_mcmc, z_mcmc, model_sparse::TypeSparse, psi, nu_orig, q_mat_mcmc, q_mat_sparse_mcmc,
    lambda_mcmc, b_mat_mcmc, ycentered, prob_zeta_mcmc, var_mcmc, pairsvec, acc) =
    error("joint_z_move implementata solo per GenSparse_V2")
