


function sample_q_metropolis!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE,  type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, adapt_q::adapt_sd_lambda_V1)  where {T_SPARSE<:TypeSparse, T_HMC<:TypeMCMC}

    k::Int64 = size(z_mcmc, 1)
         
    n_uno = zeros(Float64,k)
    #n_zero = zeros(Float64,k)
    n::Int64 = size(ydata,2)
    

    # ! numeri 1 per colonna
    for i = 1:(k-1)
        for j = (i+1):(k)
            n_uno[i] -= Float64(z_mcmc[j,i])
            #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
        end
    end
    nu::Float64 = nu_orig -  k+1 
    
    bstar_prop = adapt_q.bstar_prop
    

    b_mat_prop = adapt_q.b_mat_prop
    q_mat_prop = adapt_q.q_mat_prop
    q_mat_sparse_prop = adapt_q.q_mat_sparse_prop
    lambda_prop = adapt_q.lambda_propc
    
    # initialization
    bstar_prop .= bstar_mcmc
    from_bstar_to_b_mat(bstar_prop, b_mat_prop)
    from_bmat_to_all_q(b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)
    #lambda_prop[:, :] = q_mat_sparse_prop * transpose(q_mat_sparse_prop)


    index_h::Int64 = 0
    log_mh_ratio::Float64 = 0.0
    app_mh::Float64 = 0.0
    for j in 1:k

        for i in j:k

            index_h += 1
            adapt_q.sd_vec[index_h] = rand(Uniform(0.01, 2.0))
            bstar_prop[index_h] = rand(Normal(bstar_mcmc[index_h], adapt_q.sd_vec[index_h]))

            
            if j == i
                b_mat_prop[i, j] = exp(bstar_prop[index_h])
            else
                b_mat_prop[i, j] = bstar_prop[index_h]
            end

            log_mh_ratio = 0.0
            if (z_mcmc[i, j] == 0.0) && (i != j)
                # non ha effetto sulla verosimiglianza e le matrici
                #q_mat_prop .= psi * b_mat_prop
                # FIXME: controllare che possa veramente comemntarlo, ma penso di si
                mul!(q_mat_prop, psi, b_mat_prop)
                #from_bmat_to_all_q_from_col(j, b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)
            else
                # aggiorno
                from_bmat_to_all_q_from_col(j, b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)
                #lambda_prop[:, :] = q_mat_sparse_prop * transpose(q_mat_sparse_prop)


                
                for iobs in 1:n
                    for jcol = j:k

                        app_mh = 0.0
                        for i_app in jcol:k
                            app_mh += ycentered[i_app, iobs] .* q_mat_sparse_prop[i_app, jcol]
                        end
                        log_mh_ratio += log(q_mat_sparse_prop[jcol, jcol]) - 0.5 * app_mh^2.0

                        app_mh = 0.0
                        for i_app in jcol:k
                            app_mh += ycentered[i_app, iobs] .* q_mat_sparse_mcmc[i_app, jcol]
                        end
                        log_mh_ratio -= log(q_mat_sparse_mcmc[jcol, jcol]) - 0.5 * app_mh^2.0


                        #log_mh_ratio += q_mat_sparse_prop[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_prop[jcol:k, jcol]))^2.0
                        #log_mh_ratio -= q_mat_sparse_mcmc[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_mcmc[jcol:k, jcol]))^2.0
                    end
                end

                
            end
            
            
            if i == j

                log_mh_ratio += 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 + 2.0 * log(b_mat_prop[j, j])

                log_mh_ratio -= 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0 + 2.0 * log(b_mat_mcmc[j, j])


            else

                log_mh_ratio += -0.5 * bstar_prop[index_h]^2.0
                log_mh_ratio -= -0.5 * bstar_mcmc[index_h]^2.0

            end
            
            if isnan(log_mh_ratio)
                log_mh_ratio = -Inf
            end
            
            if rand(Uniform(0.0, 1.0)) < exp(log_mh_ratio)

                # accept
                
                bstar_mcmc[index_h] = bstar_prop[index_h]
                b_mat_mcmc[i, j] = b_mat_prop[i, j]
                q_mat_mcmc  .= q_mat_prop
                q_mat_sparse_mcmc  .= q_mat_sparse_prop
                #lambda_mcmc[:, :] = lambda_prop[:, :]


            else

                # reject
                bstar_prop[index_h] = bstar_mcmc[index_h]
                b_mat_prop[i, j] = b_mat_mcmc[i, j]
                q_mat_prop  .= q_mat_mcmc 
                q_mat_sparse_prop  .= q_mat_sparse_mcmc 


            end

            adapt_q.alpha_bar[index_h] += min(1.0, exp(log_mh_ratio))
            #println("Iteration: ", iterMCMC, " - Updating q[", i, ",", j, "] - MH ratio: ", exp(log_mh_ratio), " - sd: ", adapt_q.sd_vec[index_h], " ", adapt_q.alpha_bar[index_h])


        end

    end



    from_bstar_to_b_mat(bstar_mcmc, b_mat_mcmc)
    from_bmat_to_all_q(b_mat_mcmc,  q_mat_mcmc, q_mat_sparse_mcmc, z_mcmc,psi)
    #lambda_mcmc[:,:] = q_mat_sparse_mcmc*transpose(q_mat_sparse_mcmc)
    mul!(lambda_mcmc, q_mat_sparse_mcmc, transpose(q_mat_sparse_mcmc))

    #if iterMCMC < adapt_q.burnin

    #    @inbounds for index_h in 1:size(bstar_mcmc, 1)

    #        if (iterMCMC % adapt_q.n_batch[index_h]) == 0

    #            adapt_q.alpha_bar[index_h] = adapt_q.alpha_bar[index_h] / adapt_q.n_batch[index_h]

    #            # update sd
    #            adapt_q.sd_vec[index_h] = exp( log(adapt_q.sd_vec[index_h]) + adapt_q.vec_a[index_h] /(adapt_q.vec_b[index_h] +iterMCMC) *(adapt_q.alpha_bar[index_h]   - adapt_q.target_rate[index_h]) )
                
                


    #            adapt_q.alpha_bar[index_h] = 0.0

    #        end

    #    end


    #end
    
    

    return nothing

end
