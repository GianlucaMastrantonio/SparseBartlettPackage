

function mcmc(;
    ydata::Matrix{Float64},
    iterations::NamedTuple{(:iter, :burnin, :thin),Tuple{Int64,Int64,Int64}} = (
        iter=1000,
        burnin=200,
        thin=2
    ),
    covariates::Array{Float64,3},
    prior_lambda::Wishart{Float64, PDMat{Float64, Matrix{Float64}}},
    prior_mu::Normal{Float64},
    m_init::Vector{Float64},
    c_init::Vector{Float64},
    beta_init::Vector{Float64},
    z_init::Matrix{Float64},
    model_sparse::T_SPARSE,
    model_psi::T_PSI,
    prob_zeta::Matrix{Float64},
    type_mcmc_lambda::T_MCMC,
    iter_start_hmc::Int64 = 100,
    iter_start_zeta::Int64 = 0,
    missing_index::Matrix{Int64} = zeros(Int64,0,0),
    # --- stima di nu e psi (S = psi*I); se false restano fissi ai valori di prior_lambda ---
    estimate_nu::Bool = false,
    estimate_psi::Bool = false,
    prior_nu::ContinuousUnivariateDistribution = Gamma(2.0, 5.0),
    prior_psi::Gamma{Float64} = Gamma(1.0, 1.0),
    sd_log_nu::Float64 = 0.3,
    sd_log_psi::Float64 = 0.1,
    iter_start_hyper::Int64 = 0,
    # --- mossa congiunta (z_ij, b_ij) al posto del Gibbs su z (vedi sample_zeta_joint.jl) ---
    joint_z_move::Bool = false

    
) where {T_SPARSE<:TypeSparse, T_PSI<:TypeS, T_MCMC<:TypeMCMC}

    ydata_copy = deepcopy(ydata)
    covariates_copy = deepcopy(covariates)
    ### mcmc par
    iter = Int64(iterations.iter)
    burnin = Int64(iterations.burnin)
    thin = Int64(iterations.thin)
    sampletosave = trunc(Int64, round((iter - burnin) / thin))

    iterMCMC = Int64(0)
    thinburnin = burnin
    p1 = Progress(burnin, desc = "burnin ", offset = 0, showspeed = true)
    p2 = Progress(
        burnin + (sampletosave - 1) * thin,
        desc = "iterations ",
        offset = 0,
        showspeed = true,
    )
    isburn = true
    
    ycentered = deepcopy(ydata_copy)
    #### indices
    n::Int64 = size(ydata_copy,2)
    k::Int64 = size(ydata_copy,1)
    nm::Int64 = (k*k-k)/2
    ncov::Int64 = size(covariates_copy, 2)
    
    #### psi
    println("RR")

    
    #println(params(prior_lambda)[2][1:10,1:10])
    psi_mcmc::Matrix{Float64} =  cholesky(params(prior_lambda)[2]).L
    #println(psi_mcmc[1:10,1:10])
    psi_inv_mcmc::Matrix{Float64} =  inv(psi_mcmc)
    # nu (come df della Wishart, nu + k - 1) e psi (S = psi*I): partono dalla prior_lambda
    nu_df_mcmc::Vector{Float64} = [params(prior_lambda)[1]]
    psi_scal_mcmc::Vector{Float64} = [psi_mcmc[1,1]^2]
    acc_nu::Vector{Int64} = [0]
    acc_psi::Vector{Int64} = [0]
    n_hyper::Int64 = 0
    acc_zjoint::Vector{Int64} = zeros(Int64, 4)   # [acc on, prop on, acc off, prop off]
    array_par_inv::Array{Float64,3} = zeros(Float64, k,k,k)
    for j = 1:(k-1)
        app = psi_mcmc[(j+1):k,(j+1):k]*transpose(psi_mcmc[(j+1):k,(j+1):k])
        array_par_inv[k, (j+1):k,(j+1):k ] = inv(app)
    end

    is_diag_psi::Bool = isdiag(psi_mcmc)
    if is_diag_psi
        
    else
        error("Psi is not diagonal - currently only diagonal psi is supported")
    end
    for iii in 1:(k-1)
        if psi_mcmc[iii,iii] != psi_mcmc[iii+1,iii+1]
            error("Elemnts of the diagonl of Psi Are not the same - not yet implemented")
        end
    end
    #println("Psi is diagonal: ", is_diag_psi)
    #### mcmc
    prob_zeta_mcmc::Matrix{Float64} = deepcopy(prob_zeta)
    zmat_mcmc::Matrix{Float64} = ones(Float64,k,k)*2
    zmat_mcmc[:,:] = z_init[:,:]
    for j = 1:k
        for i = j:k
            zmat_mcmc[j,i] = 1.0
            prob_zeta_mcmc[j,i] = 0.0
        end
    end
#println(psi[1:10,1:10])
    m_mcmc::Vector{Float64} = zeros(Float64, nm)
    m_mcmc[:] = m_init;
    c_mcmc::Vector{Float64} = zeros(Float64, k)
    c_mcmc[:] = c_init;

    beta_mcmc::Vector{Float64} = zeros(Float64, ncov)
    beta_mcmc[:] = beta_init;
    mu_mcmc::Matrix{Float64} = zeros(Float64, k,n)
    for i = 1:n
        mu_mcmc[:, i] .= covariates_copy[:, :, i] * beta_mcmc
    end

    b_mat_mcmc::Matrix{Float64} = zeros(Float64, k,k)
    b_mcmc::Vector{Float64} = zeros(Float64, nm+k)
    
    bstar_mcmc::Vector{Float64} = zeros(Float64, nm+k)
    bstar_mat_mcmc::Matrix{Float64} = zeros(Float64, k,k)
    
    #q_mcmc::Vector{Float64} = zeros(Float64, nm+k)
    q_mat_mcmc::Matrix{Float64} = zeros(Float64, k,k)
    q_mat_sparse_mcmc::Matrix{Float64} = zeros(Float64, k,k)
    lambda_mcmc::Matrix{Float64} = zeros(Float64, k,k)

    var_mcmc::Vector{Float64} = ones(Float64, k)
    var_prop::Vector{Float64} = ones(Float64, k)

    from_c_and_m_to_all_b(c_mcmc, m_mcmc, b_mat_mcmc, b_mcmc, bstar_mcmc, bstar_mat_mcmc)
    from_bmat_to_all_q(b_mat_mcmc,  q_mat_mcmc, q_mat_sparse_mcmc, zmat_mcmc, psi_mcmc)
    lambda_mcmc[:,:] = q_mat_sparse_mcmc*transpose(q_mat_sparse_mcmc)



    # hamiltonian
    r_hmc_mcmc::Vector{Float64} = rand(Normal(0.0,1.0),size(bstar_mcmc,1))
    
    m_out::Array{Float64,2} = zeros(Float64,sampletosave, nm)
    c_out::Array{Float64,2} = zeros(Float64,sampletosave, k)
    beta_out::Array{Float64,2} = zeros(Float64,sampletosave, ncov)
    nu_out::Vector{Float64} = zeros(Float64, sampletosave)   # nu del paper (df - k + 1)
    psi_out::Vector{Float64} = zeros(Float64, sampletosave)  # S = psi * I
    q_mat_sparse_out::Array{Float64,3} = zeros(Float64,sampletosave, k,k)
    lambda_out::Array{Float64,3} = zeros(Float64,sampletosave, k,k)
    #omega_out::Array{Float64,2} = zeros(Float64,sampletosave, nm)
    #eta_out::Array{Float64,2} = zeros(Float64,sampletosave, k)
    zmat_out::Array{Int64,3}  = zeros(Int64,sampletosave, k,k)
    #lambda_out::Array{Float64,3} = zeros(Float64,sampletosave, k,k)

    ##### missing
    nmiss = size(missing_index,1)
    if nmiss == 0
        out_missing = zeros(Float64,sampletosave, 0,0)
    else
        out_missing = zeros(Float64,sampletosave, nmiss)
    end


    
    prec_q::Vector{Matrix{Float64}} = [zeros(Float64, k - l + 1, k - l + 1) for l = 1:k]
    for j = 1:(k-1)
        prec_q[j] = inv(psi_mcmc[j:k, j:k] * transpose(psi_mcmc[j:k, j:k]))
    end

    #max_s = max(max_s,abs(lengthdata[i] - length_w_mcmc[zeta_mcmc[i]]))
    ### prints
    println("MCMC settings ")
    println("Iterations: ", iter)
    println("Burnin: ", burnin)
    println("Thin: ", thin)
    println("Number of posterior samples: ", sampletosave)
    println("Number of threads: ", Threads.nthreads())
    
    
    
    ### algorithm
    for i = 1:n
        ycentered[:,i] = ydata_copy[:,i] - mu_mcmc[:,i]
    end
    
    ### adapt_sd
    adapt_q = create_adapt_sd_lambda_V1(; nm=nm, k=k, sd=0.1, acc_rate=0.234, A=1000.0, B=2000.0, iter_batch=50, burnin=burnin)

    ### chenge order
    order_mcmc = create_order_lambda_V3(; nm=nm, k=k, n=n)
    order_matrix_mcmc::Matrix{Float64} = Matrix(I(k)) .* 1.0
    order_matrix_out::Array{Float64,3} = zeros(Float64, sampletosave, k, k)
    


    

    marginal_variances(model_sparse, q_mat_sparse_mcmc, zeros(Float64,k), zeros(Float64,k), model_sparse.varvec)
    d_diag_miss::Matrix{Float64} = zeros(Float64,k,k)
    log_likelihood_mcmc::Vector{Float64} = zeros(Float64, k)
    log_likelihood_prop::Vector{Float64} = zeros(Float64, k)
    pairsvec = [(i,j) for j in 1:(k-1) for i in (j+1):k]
    #println( model_sparse.varvec)
    
    for iMCMC = 1:sampletosave

        for jMCMC = 1:thinburnin
            
            iterMCMC += 1
            #println(iterMCMC)
            ProgressMeter.next!(p2; showvalues = [(:iterations, iterMCMC), (:n_zeros,  Int64(abs(sum(zmat_mcmc  .- 1)) ))])
            #println(bstar_mcmc[1:4])
            #sample_zeta_and_q!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse,  type_mcmc_lambda, psi, params(prior_lambda)[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, psi_inv, array_par_inv)

            
            #for _ = 1:iter_start_hmc
            #    sample_q!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi, params(prior_lambda)[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered)
            #end
            
             sample_q_metropolis!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi_mcmc, nu_df_mcmc[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered, adapt_q, var_mcmc, log_likelihood_mcmc, log_likelihood_prop)
            #sample_q_elliptical!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi, params(prior_lambda)[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered, adapt_q, var_mcmc)
            #iter_start_hmc = 1
            
            
             if iter_start_zeta < iterMCMC
                if joint_z_move
                    sample_zeta_joint!(bstar_mcmc, zmat_mcmc, model_sparse, psi_mcmc, nu_df_mcmc[1], q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, ycentered, prob_zeta_mcmc, var_mcmc, pairsvec, acc_zjoint)
                else
                    sample_zeta!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi_mcmc, nu_df_mcmc[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, ycentered, prob_zeta_mcmc, var_mcmc, pairsvec, log_likelihood_mcmc, log_likelihood_prop)
                end
                #sample_zeta_efficient!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi, params(prior_lambda)[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, ycentered, prob_zeta_mcmc,  prec_q)
            end
            
             sample_mu!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi_mcmc, nu_df_mcmc[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered, prior_mu, beta_mcmc, covariates_copy, var_mcmc)

             for _ = 1:Int64(floor(0.33*k))
                # k/2 log 2
                change_order_mcmc!(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi_mcmc, nu_df_mcmc[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered, order_mcmc, psi_inv_mcmc, covariates_copy, order_matrix_mcmc, is_diag_psi, var_mcmc, var_prop)
            end
            
            if (estimate_nu || estimate_psi) && (iterMCMC > iter_start_hyper)
                update_hyper!(estimate_nu, estimate_psi, nu_df_mcmc, psi_scal_mcmc, prior_nu, prior_psi, sd_log_nu, sd_log_psi,
                    acc_nu, acc_psi, ycentered, bstar_mcmc, b_mat_mcmc, zmat_mcmc, model_sparse,
                    psi_mcmc, psi_inv_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc)
                n_hyper += 1
            end

             sample_var(iterMCMC, ydata_copy, bstar_mcmc, zmat_mcmc, model_sparse, type_mcmc_lambda, psi_mcmc, nu_df_mcmc[1], r_hmc_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, lambda_mcmc, b_mat_mcmc, mu_mcmc, ycentered, order_mcmc, psi_inv_mcmc, covariates_copy, order_matrix_mcmc, is_diag_psi, var_mcmc, var_prop)
            
            if nmiss > 0
                #diag(d_diag_miss) .= (model_sparse.varvec[1:k] ./ var_mcmc[1:k]).^0.5

                d_diag_miss[diagind(d_diag_miss)] .= (model_sparse.varvec[1:k] ./ var_mcmc[1:k]).^0.5
                mu_mcmc .= order_matrix_mcmc' *mu_mcmc
                ydata_copy .= order_matrix_mcmc' * ydata_copy
                ycentered .= order_matrix_mcmc' * ycentered
                

                lambda_mcmc .= order_matrix_mcmc' * d_diag_miss*lambda_mcmc* d_diag_miss * order_matrix_mcmc
                
                mean_cond = zeros(Float64,k)
                var_cond = 0.0
                for index_miss = 1:nmiss
                    imiss = missing_index[index_miss, 2]
                    kmiss = missing_index[index_miss, 1]

                    var_cond = 1.0/lambda_mcmc[kmiss,kmiss]
                    mean_cond .= -var_cond*lambda_mcmc[:,kmiss]
                    mean_cond[kmiss] = 0.0
                    ydata_copy[kmiss,imiss] = rand(Normal(mu_mcmc[kmiss, imiss]+ sum(mean_cond.*(ydata_copy[:,imiss] - mu_mcmc[:, imiss])), var_cond^0.5  ))
                end
                d_diag_miss[diagind(d_diag_miss)] .= ( var_mcmc[1:k] ./ model_sparse.varvec[1:k] ).^0.5
                mu_mcmc .= order_matrix_mcmc * mu_mcmc
                ydata_copy .= order_matrix_mcmc * ydata_copy
                ycentered .= order_matrix_mcmc * ycentered
                lambda_mcmc .= order_matrix_mcmc * d_diag_miss*lambda_mcmc* d_diag_miss * order_matrix_mcmc'

                # TODO aggiungerlo anche ai GLM
                for i = 1:n
                    ycentered[:, i] .= ydata_copy[:, i] - mu_mcmc[:, i]
                end
            end
            
            
        end

        thinburnin = thin
        isburn = false
        
        from_bmat_to_m_and_c(c_mcmc, m_mcmc, b_mat_mcmc)
        m_out[iMCMC,:] = m_mcmc
        c_out[iMCMC,:] = c_mcmc
        q_mat_sparse_out[iMCMC,:,:] = q_mat_sparse_mcmc[:,:]
        d_diag_miss[diagind(d_diag_miss)] .= (model_sparse.varvec[1:k] ./ var_mcmc[1:k]).^0.5
        lambda_out[iMCMC, :, :] = order_matrix_mcmc' * d_diag_miss*(q_mat_sparse_mcmc * transpose(q_mat_sparse_mcmc))*d_diag_miss * order_matrix_mcmc
        zmat_out[iMCMC, :, :] = order_matrix_mcmc' * zmat_mcmc[:, :] * order_matrix_mcmc
        for i = 1:k
            for j = i:k
                if zmat_out[iMCMC, i, j] == 0.0

                    zmat_out[iMCMC, j, i] = 0.0
                    zmat_out[iMCMC, i, j] = 1.0

                end
            end
        end
        beta_out[iMCMC,:] = beta_mcmc
        nu_out[iMCMC] = nu_df_mcmc[1] - k + 1
        psi_out[iMCMC] = psi_scal_mcmc[1]
        order_matrix_out[iMCMC, :, :] = order_matrix_mcmc[:, :]
        if nmiss > 0
            ydata_copy .= order_matrix_mcmc' * ydata_copy
            for index_miss = 1:nmiss
                imiss = missing_index[index_miss, 2]
                kmiss = missing_index[index_miss, 1]
                out_missing[iMCMC,index_miss] = ydata_copy[kmiss,imiss]
            end
            ydata_copy .= order_matrix_mcmc * ydata_copy
        end

    end

    if joint_z_move
        println("Acceptance z on: ", acc_zjoint[1] / max(acc_zjoint[2], 1), "  off: ", acc_zjoint[3] / max(acc_zjoint[4], 1))
    end
    if n_hyper > 0
        estimate_nu  && println("Acceptance nu: ", acc_nu[1] / n_hyper)
        estimate_psi && println("Acceptance psi (cresta): ", acc_psi[1] / n_hyper)
    end
    return (m_out=m_out, c_out=c_out, q_mat_sparse_out=q_mat_sparse_out, lambda_out=lambda_out, zmat_out=zmat_out, beta_out=beta_out, missing_out=out_missing, order_matrix_out=order_matrix_out, nu_out=nu_out, psi_out=psi_out)
    
end