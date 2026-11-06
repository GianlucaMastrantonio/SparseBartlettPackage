


function sample_q_metropolis!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE,  type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, adapt_q::adapt_sd_lambda_V1, var_mcmc::Vector{Float64},log_likelihood_mcmc::Vector{Float64}, log_likelihood_prop::Vector{Float64})  where {T_SPARSE<:TypeSparse, T_HMC<:TypeMCMC}

    k::Int64 = size(z_mcmc, 1)
         
    #n_uno = zeros(Float64,k)
    #for i = 1:(k-1)
    #    for j = (i+1):(k)
    #        n_uno[i] -= Float64(z_mcmc[j, i])
    #        #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
    #    end
    #end
    #n_zero = zeros(Float64,k)
    n::Int64 = size(ydata,2)
    
    add_to_v = zeros(Float64, k)
    for i = 1:(k-1)
        for j = (i+1):(k)
            add_to_v[i] += Float64(z_mcmc[j, i])
            #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
        end
    end
    # ! numeri 1 per colonna
    
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
    ei::Vector{Float64} = zeros(Float64, k)
    app_x::Vector{Float64} = zeros(Float64, k)
    #println(psi[1:10,1:10])
    #@assert 1 == 2
    #! loglike
    log_likelihood_mcmc .= 0.0
    @inbounds for iobs in 1:n
        for jcol = 1:k

            app_mh = 0.0
            for i_app in jcol:k
                app_mh += ycentered[i_app, iobs] * q_mat_sparse_mcmc[i_app, jcol]*sqrt(model_sparse.varvec[i_app] /var_mcmc[i_app])
            end
            log_likelihood_mcmc[jcol] += log(q_mat_sparse_mcmc[jcol, jcol]*sqrt(model_sparse.varvec[jcol] / var_mcmc[jcol])) - 0.5 * app_mh * app_mh
        end
    end

    log_likelihood_prop .= log_likelihood_mcmc

    sqrt_app = zeros(Float64,k)
    
    for j in 1:k

        for i in j:k

            index_h += 1
            #adapt_q.sd_vec[index_h] = rand(Uniform(0.01, 2.0))
            #bstar_prop[index_h] = rand(Normal(bstar_mcmc[index_h], adapt_q.sd_vec[index_h]))
            adapt_q.sd_vec[index_h] = 0.01 + 1.99 * rand()
            bstar_prop[index_h] = bstar_mcmc[index_h] + adapt_q.sd_vec[index_h] * randn()


            
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
                #mul!(q_mat_prop, psi, b_mat_prop)
                new_from_bmat_to_all_q_from_col(j, b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)

#                if iterMCMC> 1
#                q_start  = deepcopy(q_mat_mcmc)

#qs_start = deepcopy(q_mat_sparse_mcmc)

#q_old  = deepcopy(q_start)

#qs_old = deepcopy(qs_start)

#q_new  = deepcopy(q_start)

#qs_new = deepcopy(qs_start)

#b_test = deepcopy(b_mat_prop)

#from_bmat_to_all_q_from_col(

#    j, b_test, q_old, qs_old, z_mcmc, psi

#)

#new_from_bmat_to_all_q_from_col(

#    j, b_test, q_new, qs_new, z_mcmc, psi

#)

#println(maximum(abs.(q_old .- q_new)))

#println(maximum(abs.(qs_old .- qs_new)))
#println(maximum(abs.(psi .- LowerTriangular(psi))))
#                end
                #q_mat_sparse_prop .= q_mat_sparse_mcmc
                #from_bmat_to_all_q_from_col(j, b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)
            else
                # aggiorno
                new_from_bmat_to_all_q_from_col(j, b_mat_prop, q_mat_prop, q_mat_sparse_prop, z_mcmc, psi)
                #lambda_prop[:, :] = q_mat_sparse_prop * transpose(q_mat_sparse_prop)
                

                marginal_variances(model_sparse, q_mat_sparse_prop, ei, app_x, model_sparse.varvec_p)
                
                for jcol = j:k
                    log_likelihood_prop[jcol] = 0.0
                end
                
                for i_app in j:k
                    sqrt_app[i_app] = sqrt(model_sparse.varvec_p[i_app]/var_mcmc[i_app])
                end
                @inbounds for iobs in 1:n
                    for jcol = j:k

                        app_mh = 0.0
                        for i_app in jcol:k
                            #app_mh += ycentered[i_app, iobs] * q_mat_sparse_prop[i_app, jcol]*sqrt(model_sparse.varvec_p[i_app]/var_mcmc[i_app])
                            app_mh += ycentered[i_app, iobs] * q_mat_sparse_prop[i_app, jcol]*sqrt_app[i_app]
                        end
                        #log_likelihood_prop[jcol] += log(q_mat_sparse_prop[jcol, jcol]*sqrt(model_sparse.varvec_p[jcol]/var_mcmc[jcol])) - 0.5 * app_mh * app_mh
                        log_likelihood_prop[jcol] += log(q_mat_sparse_prop[jcol, jcol]*sqrt_app[jcol]) - 0.5 * app_mh * app_mh

                        
                    end
                end
                @inbounds for h = j:k
                    log_mh_ratio += log_likelihood_prop[h] - log_likelihood_mcmc[h]
                end
                

                
            end
            
            
            if i == j

                log_mh_ratio += 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 + 2.0 * log(b_mat_prop[j, j])

                log_mh_ratio -= 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0 + 2.0 * log(b_mat_mcmc[j, j])


            else

                #log_mh_ratio += -0.5 * bstar_prop[index_h]^2.0
                #log_mh_ratio -= -0.5 * bstar_mcmc[index_h]^2.0
                #println(model_sparse.distr)
                log_mh_ratio += logpdf(model_sparse.distr, bstar_prop[index_h])
                log_mh_ratio -= logpdf(model_sparse.distr, bstar_mcmc[index_h])

            end
            
            if isnan(log_mh_ratio)
                log_mh_ratio = -Inf
            end
            
            if rand() < exp(log_mh_ratio)

                # accept
                
                bstar_mcmc[index_h] = bstar_prop[index_h]
                b_mat_mcmc[i, j] = b_mat_prop[i, j]
                q_mat_mcmc  .= q_mat_prop
                q_mat_sparse_mcmc  .= q_mat_sparse_prop
                #lambda_mcmc[:, :] = lambda_prop[:, :]
                model_sparse.varvec .= model_sparse.varvec_p

                @inbounds for h = j:k
                    log_likelihood_mcmc[h] = log_likelihood_prop[h]
                end
            else

                # reject
                bstar_prop[index_h] = bstar_mcmc[index_h]
                b_mat_prop[i, j] = b_mat_mcmc[i, j]
                q_mat_prop  .= q_mat_mcmc 
                q_mat_sparse_prop  .= q_mat_sparse_mcmc 

                model_sparse.varvec_p .= model_sparse.varvec

                @inbounds for h = j:k
                    log_likelihood_prop[h] = log_likelihood_mcmc[h]
                end
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


function like_elliptical!( bstar::Vector{Float64}, z_mcmc::Matrix{Float64}, psi::Matrix{Float64},  q_mat::Matrix{Float64}, q_mat_sparse::Matrix{Float64},  b_mat::Matrix{Float64}, ycentered::Matrix{Float64}, add_to_v::Vector{Float64}, nu::Float64, k::Int64,n::Int64, ei::Vector{Float64}, app_x::Vector{Float64}, var_mcmc::Vector{Float64})::Float64




    from_bstar_to_b_mat(bstar, b_mat)
    from_bmat_to_all_q(b_mat, q_mat, q_mat_sparse, z_mcmc, psi)
    log_like::Float64 = 0.0
    app_mh::Float64 = 0.0
    
    marginal_variances(model_sparse, q_mat_sparse, ei, app_x, model_sparse.varvec)
    for iobs in 1:n
        for jcol = 1:k
            app_mh = 0.0
            for i_app in jcol:k
                app_mh += ycentered[i_app, iobs] .* q_mat_sparse[i_app, jcol]*model_sparse.varvec[i_app]^0.5/var_mcmc[i_app]^0.5
            end
            log_like += log(q_mat_sparse[jcol, jcol]*model_sparse.varvec[jcol]^0.5/ var_mcmc[jcol]^0.5 ) - 0.5 * app_mh^2.0
        end
    end

    index_h::Int64 = 0
    

    for j in 1:k

        for i in j:k

            index_h += 1
            if i == j

                log_like += 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat[j, j]) - b_mat[j, j]^2 / 2.0 + 2.0 * log(b_mat[j, j])
                log_like -= logpdf(Normal(0.0, 1.0), log(b_mat[j, j]))
            else

                log_like += logpdf(model_sparse.distr, b_mat[i,j]) -logpdf(Normal(0.0, 1.0), b_mat[i,j])


            end

        end

    end
    return log_like

end


#function sample_q_elliptical!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE,  type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, adapt_q::adapt_sd_lambda_V1)  where {T_SPARSE<:TypeSparse, T_HMC<:TypeMCMC}

#    k::Int64 = size(z_mcmc, 1)
         

#    n::Int64 = size(ydata,2)
#    ei::Vector{Float64} = zeros(Float64, k)
#    app_x::Vector{Float64} = zeros(Float64, k)
#    add_to_v = zeros(Float64, k)
#    for i = 1:(k-1)
#        for j = (i+1):(k)
#            add_to_v[i] += Float64(z_mcmc[j, i])
#            #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
#        end
#    end
#    # ! numeri 1 per colonna
#    nu::Float64 = nu_orig -  k+1 # questo è b  
    


#    init_dir = adapt_q.bstar_prop
    

#    # step 2
#    b_curent = copy(bstar_mcmc)
#    init_dir .= rand(Normal(0.0,1.0), size(bstar_mcmc, 1))
#    # step 3
#    u_var = rand(Uniform(0.0, 1.0))
#    #log_y_base::Float64 = like_elliptical!(bstar_mcmc, z_mcmc, psi,  q_mat_mcmc, q_mat_sparse_mcmc,  b_mat_mcmc, ycentered, add_to_v, nu, k, n) +  log(u_var)
#    log_y_base::Float64 = like_elliptical!(b_curent, z_mcmc, psi, q_mat_mcmc,q_mat_sparse_mcmc, b_mat_mcmc, ycentered, add_to_v, nu, k, n, ei, app_x) + log(u_var)
#    # step 4
#    theta::Float64 = rand(Uniform(0.0,2*pi))
#    theta_min::Float64 = theta - 2*pi
#    theta_max::Float64 = theta

#    # step 5
#    bstar_mcmc .= b_curent .* cos(theta) + init_dir .* sin(theta)
#    log_like_prop = like_elliptical!(bstar_mcmc, z_mcmc, psi, q_mat_mcmc, q_mat_sparse_mcmc, b_mat_mcmc, ycentered, add_to_v, nu, k, n, ei, app_x)
#    # step 6
#    while  !isfinite(log_like_prop) || (log_like_prop < log_y_base)

#        if theta < 0.0
#            theta_min = theta
#        else
#            theta_max = theta
#        end

#        theta = rand(Uniform(theta_min, theta_max))
#        # step 5 again
#        bstar_mcmc .= b_curent .* cos(theta) + init_dir .* sin(theta)
#        log_like_prop = like_elliptical!(bstar_mcmc, z_mcmc, psi, q_mat_mcmc, q_mat_sparse_mcmc, b_mat_mcmc, ycentered, add_to_v, nu, k, n, ei, app_x)

#    end


#    from_bstar_to_b_mat(bstar_mcmc, b_mat_mcmc)
#    from_bmat_to_all_q(b_mat_mcmc,  q_mat_mcmc, q_mat_sparse_mcmc, z_mcmc,psi)
#    #lambda_mcmc[:,:] = q_mat_sparse_mcmc*transpose(q_mat_sparse_mcmc)
#    mul!(lambda_mcmc, q_mat_sparse_mcmc, transpose(q_mat_sparse_mcmc))
    
    
    
    

#    return nothing

#end
