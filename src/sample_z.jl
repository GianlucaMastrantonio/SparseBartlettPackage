


function compute_like_for_z_all(j::Int64, n::Int64,qstar::Matrix{Float64}, like::Vector{Float64}, z_mcmc::Matrix{Float64} ,k::Int64, ydata::Matrix{Float64}, model_sparse::GenSparse_V2, start::Int64, ei::Vector{Float64}, app_x::Vector{Float64}, var_mcmc::Vector{Float64}) 

    for jcol = start:(k)
        #compute_like_for_z(jcol, n, qstar, like, z_mcmc,k, ydata, model_sparse, var_mcmc )
        new_compute_like_for_z(jcol, n, qstar, like, z_mcmc,k, ydata, model_sparse, var_mcmc )
    end

end

#function compute_like_for_z_all(j::Int64, n::Int64,qstar::Matrix{Float64}, like::Vector{Float64}, z_mcmc::Matrix{Float64} ,k::Int64, ydata::Matrix{Float64}, model_sparse::GenCorSparse_V2, start::Int64, ei::Vector{Float64}, app_x::Vector{Float64}, var_mcmc::Vector{Float64}) 

#    marginal_variances(model_sparse, qstar, ei, app_x, model_sparse.varvec_p)
#    for jcol = 1:k
#        compute_like_for_z(jcol, n, qstar, like, z_mcmc,k, ydata, model_sparse, var_mcmc )
#    end

#end



function old_compute_like_for_z(j::Int64, n::Int64,qstar::Matrix{Float64}, like::Vector{Float64}, z_mcmc::Matrix{Float64} ,k::Int64, ydata::Matrix{Float64}, model_sparse::T_SPARSE, var_mcmc::Vector{Float64}) where {T_SPARSE<:TypeSparse}
    error("DO NOT USE")
    like[j] = 0.0
    @inbounds for iobs = 1:n
        like[j] += compute_like_for_z_colj(iobs, j, n, qstar, like , z_mcmc, k, ydata,model_sparse, var_mcmc )
    end

end

function old_compute_like_for_z_colj(iobs::Int64,  j::Int64, n::Int64, qstar::Matrix{Float64}, like_acc::Vector{Float64} , z_mcmc::Matrix{Float64} ,k::Int64, ydata::Matrix{Float64}, model_sparse::GenSparse_V2, var_mcmc::Vector{Float64})::Float64
error("DO NOT USE")
    ret::Float64 = 0.0
    @inbounds for jcol = j:k
        ret += ydata[jcol,iobs] * qstar[jcol,j]
    end
    return log(qstar[j,j]) -0.5*ret * ret
    #return log(qstar[j,j]) -0.5*( sum(ydata[j:k,iobs] .* qstar[j:k,j] ))^2.0
end

#function compute_like_for_z_colj(iobs::Int64,  j::Int64, n::Int64, qstar::Matrix{Float64}, like_acc::Vector{Float64} , z_mcmc::Matrix{Float64} ,k::Int64, ydata::Matrix{Float64}, model_sparse::GenCorSparse_V2, var_mcmc::Vector{Float64})::Float64

#    ret::Float64 = 0.0
#    @inbounds for jcol = j:k
#        #ret += ydata[jcol,iobs] .* qstar[jcol,j]*model_sparse.varvec_p[jcol]^0.5/var_mcmc[jcol]^0.5
#        ret += ydata[jcol,iobs] .* qstar[jcol,j]*sqrt(model_sparse.varvec_p[jcol]/var_mcmc[jcol])
#    end
#    return log(qstar[j,j]*model_sparse.varvec_p[j]^0.5/var_mcmc[j]^0.5) -0.5*ret * ret
#    #return log(qstar[j,j]) -0.5*( sum(ydata[j:k,iobs] .* qstar[j:k,j] ))^2.0
#end


function new_compute_like_for_z(

    j::Int64, n::Int64, qstar::Matrix{Float64},
    like::Vector{Float64}, z_mcmc::Matrix{Float64},
    k::Int64, ydata::Matrix{Float64},
    model_sparse::GenSparse_V2, var_mcmc::Vector{Float64}
)

    acc::Float64 = 0.0
    qjj::Float64 = qstar[j,j]
    logqjj::Float64 = log(qjj)

    @inbounds for iobs = 1:n
        ret::Float64 = 0.0
        for jcol = j:k
            ret += ydata[jcol,iobs] * qstar[jcol,j]
        end
        acc += logqjj - 0.5 * ret * ret
    end
    like[j] = acc
    return nothing

end




function sample_zeta!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE,  type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64},ycentered::Matrix{Float64},prob_zeta_mcmc::Matrix{Float64}, var_mcmc::Vector{Float64}, pairsvec::Vector{Tuple{Int64, Int64}}, log_likelihood_mcmc::Vector{Float64}, log_likelihood_prop::Vector{Float64}, type_distr_c::T_DISTR_C)  where {T_SPARSE<:TypeSparse, T_HMC<:TypeMCMC, T_DISTR_C<:TypeParChisq}

    sum_zero::Int64 = abs(sum(z_mcmc  .- 1)) +  1
    
    
    #n_sample::Int64 = size(model_sparse.prob_vec_zeros)[1]
    k::Int64 = size(z_mcmc,1)
    n::Int64 = size(ycentered,2)
    nm::Int64 = size(bstar_mcmc,1)- k
    hc::Int64 = 0
    ei::Vector{Float64} = zeros(Float64, k)
    app_x::Vector{Float64} = zeros(Float64, k)
    #hc_vec = zeros(Int64,k)
    #hh::Int64 = 1
    #for j = 1:k
    #    hc_vec[j] = hh
    #    hh = hh + (k-j) + 1
    #end
    #println("start")
    #println([nu, Float64(k)])
    #n_zero_vec = zeros(Float64, k)
    #for j = 1:(k-1)
    #    for i = (j+1):k
    #        n_zero_vec[j] += Float64((1.0 - z_mcmc[i, j]))
    #    end
    #    #@assert     n_zero_vec[j] >= 0
    #    #@assert     n_zero_vec[j] <= k
    #end
    n_uno_vec = zeros(Float64,k)
    n_zero_vec = zeros(Float64,k)
    # ! numeri 1 per riga
    #for j = 2:k
    #    for i = 1:(j-1)
    #        n_uno_vec[j] += Float64(z_mcmc[j,i])
    #        n_zero_vec[j] += Float64(1.0 - z_mcmc[j,i])
    #    end
    #end
    # ! numeri 1 per colonna
    compute_add_to_v(type_distr_c,  n_uno_vec, z_mcmc)
    #for i = 1:(k-1)
    #    for j = (i+1):(k)
    #        n_uno_vec[i] += Float64(z_mcmc[j,i])

    #    end
    #end

    n_uno_vec_prop = deepcopy(n_uno_vec)
    n_zero_vec_prop = deepcopy(n_zero_vec)
    #println(n_zero_vec)

    samp::Int64 = 0
    z_prop::Matrix{Float64} = deepcopy(z_mcmc)

    bmat::Matrix{Float64} = zeros(Float64, k,k) 
    from_bstar_to_bmat(bstar_mcmc, bmat, k, n)
    qmat::Matrix{Float64} = zeros(Float64, k,k) 
    from_bmat_to_qmat(bmat, qmat,k, n, psi)
    qstar::Matrix{Float64} = zeros(Float64, k,k)
    from_qmat_to_qstar_init(qmat, qstar, k, n, psi, z_mcmc)
    qstar_prop = deepcopy(qstar)


    like_acc::Vector{Float64} = zeros(Float64,k)
    #for j = 1:k
    #    compute_like_for_z(j, n, qstar, like_acc, z_mcmc, k ,ycentered)
    #end
    compute_like_for_z_all(1, n, qstar, like_acc, z_mcmc, k ,ycentered, model_sparse, 1, ei, app_x, var_mcmc)
    like_prop::Vector{Float64} = deepcopy(like_acc)

    sum_like_acc = sum(like_acc)
    sum_like_prop = sum_like_acc

    log_prob_0_1::Vector{Float64} = zeros(Float64, 2)
    nzero_0_1::Vector{Int64} = zeros(Int64, 2)

    #sum_prob_over1::Float64 = 0.0
    #for ii = 1:(k-1)
    #    for jj = (ii+1):k
    #        if z_mcmc[jj,ii] == 1
    #            #println(prob_zeta_mcmc[jj,ii])
    #            sum_prob_over1 += prob_zeta_mcmc[jj,ii]
    #        end
            
    #    end
    #end
#    println(prob_zeta_mcmc)
#    println(z_mcmc)
#    println(sum_prob_over1)
#error("")
    par0::Float64 = 0.00
    par1::Float64 = 0.00
    #pairs = [(i,j) for j in 1:(k-1) for i in (j+1):k]
    shuffle!(pairsvec)
    #for (i,j)  = 1:nm
    for (i,j)  in    pairsvec

        #i = sample(1:k,1)[1]
        #j = sample(1:k,1)[1]

        #@assert (z_mcmc[i,j] == 0) | (z_mcmc[i,j]  == 1)
        if i != j
 
            j,i = min(i,j), max(i,j)
            if z_mcmc[i,j] == 0.0
                
                # ! colonna
                #n_uno_vec_prop[i] = n_uno_vec[i] + 1.0
                #n_zero_vec_prop[i] = n_zero_vec[i] - 1.0
                # prob di mettere uno zero
                #par0 = (nu - Float64(k) + 1.0  + maximum(n_uno_vec) -n_uno_vec[i] )
                #par1 = (nu - Float64(k) + 1.0  + maximum(n_uno_vec_prop) -n_uno_vec_prop[i] )
                
                # ! riga
                n_uno_vec_prop[j] = n_uno_vec[j] + 1.0
                n_zero_vec_prop[j] = n_zero_vec[j] - 1.0
                # prob di mettere uno zero
                par0 = (nu - Float64(k) + 1.0  + n_uno_vec[j] )
                par1 = (nu - Float64(k) + 1.0  + n_uno_vec_prop[j] )
            
                
                

                log_prob_0_1[1] = log(1.0 - prob_zeta_mcmc[i, j]) + 2.0 * (par0 / 2.0 - 1.0) * log(bmat[j,j]) + (par0/ 2.0)*log(0.5) - loggamma(par0 / 2.0  )
                log_prob_0_1[2] = log(prob_zeta_mcmc[i, j])       + 2.0 * (par1 / 2.0 - 1.0) * log(bmat[j,j]) + (par1/ 2.0)*log(0.5) - loggamma(par1 / 2.0  )
                nzero_0_1[1] = sum_zero
                nzero_0_1[2] = sum_zero-1

                ### sparsa
                #log_prob_0_1[1] += sum(like_acc[1:k])
                #for iiii = 1:k
                #    log_prob_0_1[1] += like_acc[iiii]
                #end
                log_prob_0_1[1] += sum_like_acc
                ### non sparsa
                z_prop[i,j] = 1.0
                from_qmat_to_qstar_triangularindex(i, j, qmat, qstar_prop, k, n, psi, z_prop)
                #for jcol = j:(k-1)
                #    compute_like_for_z(jcol, n, qstar_prop, like_prop, z_prop , k, ycentered)
                #end
                compute_like_for_z_all(j, n, qstar_prop, like_prop, z_prop, k ,ycentered, model_sparse, j, ei, app_x, var_mcmc)
                #for jcol = 1:k
                #    log_prob_0_1[2] += like_prop[jcol]
                #end
                sum_like_prop = sum(like_prop)
                
                log_prob_0_1[2] += sum_like_prop
                #log_prob_0_1[2] += sum(like_prop[j:(k-1)])

                #@assert !isnan(log_prob_0_1[2])
                if isnan(log_prob_0_1[2])
                    samp = 1
                else
                    samp = argmax(log_prob_0_1 .+ rand(Gumbel(0,1))) 
                end
                
                sum_zero = nzero_0_1[samp]
                z_mcmc[i,j] = Float64(samp-1)

                if z_mcmc[i,j] == 0.0 ## sparsa, vecchio valore
                    # ! colonna
                    #n_uno_vec_prop[i] = n_uno_vec[i]
                    #n_zero_vec_prop[i] = n_zero_vec[i]
                    # ! riga
                    n_uno_vec_prop[j] = n_uno_vec[j]
                    n_zero_vec_prop[j] = n_zero_vec[j]
                    
                    
                    z_prop[i,j] = 0.0
                    #like_prop[1:k] = like_acc[1:k]
                    
                    for iiii = 1:k
                        like_prop[iiii] = like_acc[iiii]
                    end
                    

                    qstar_prop[i,j] = qstar[i,j]
                    if (j+1) < i
                        for jcol  = (j+1):(i-1)
                            for irow = i:k
                                qstar_prop[irow,jcol] = qstar[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end
                    if i<k
                        for jcol = i:(k-1)                            
                            for irow = (jcol+1):k
                                qstar_prop[irow,jcol] = qstar[irow,jcol]
                            
                            end
                        end
                    end

                else 
                    # ! colonna
                    
                    #n_uno_vec[i] = n_uno_vec_prop[i]
                    #n_zero_vec[i] = n_zero_vec_prop[i]
                    # ! riga
                    n_uno_vec[j] = n_uno_vec_prop[j]
                    n_zero_vec[j] = n_zero_vec_prop[j]
                    sum_like_acc = sum_like_prop
                    #like_acc[1:k] = like_prop[1:k] 
                    for iiii = 1:k
                         like_acc[iiii] = like_prop[iiii]
                    end
                    #qstar .= qstar_prop
                    #for irow = (j+1):k
                    #    for jcol = j:(k-1)
                    #        qstar[irow,jcol] = qstar_prop[irow,jcol]
                    #    end
                    #end
                    model_sparse.varvec .= model_sparse.varvec_p
                    qstar[i,j] = qstar_prop[i,j]
                    if (j+1) < i
                        for jcol  = (j+1):(i-1)
                            for irow = i:k
                                qstar[irow,jcol] = qstar_prop[irow,jcol]
                                
                            end
                        end
                    end
                    if i<k
                        for jcol = i:(k-1)                            
                            for irow = (jcol+1):k
                                qstar[irow,jcol] = qstar_prop[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end

                
                end


            else # zeta = 1

                # ! colonna
                #n_uno_vec_prop[i] = n_uno_vec[i] - 1.0
                #n_zero_vec_prop[i] = n_zero_vec[i] + 1.0
                #par0 = (nu - Float64(k) + 1.0  + maximum(n_uno_vec_prop) -n_uno_vec_prop[i] )
                #par1 = (nu - Float64(k) + 1.0  + maximum(n_uno_vec) -n_uno_vec[i] )
                # ! riga
                n_uno_vec_prop[j] = n_uno_vec[j] - 1.0
                n_zero_vec_prop[j] = n_zero_vec[j] + 1.0
                par0 = (nu - Float64(k) + 1.0  + n_uno_vec_prop[j] )
                par1 = (nu - Float64(k) + 1.0  + n_uno_vec[j] )
                
                

                log_prob_0_1[1] = log(1.0 - prob_zeta_mcmc[i, j]) + 2.0 * (par0 / 2.0 - 1.0) * log(bmat[j,j]) + (par0/ 2.0)*log(0.5) - loggamma(par0 / 2.0  )
                log_prob_0_1[2] = log(prob_zeta_mcmc[i, j])       + 2.0 * (par1 / 2.0 - 1.0) * log(bmat[j,j]) + (par1/ 2.0)*log(0.5) - loggamma(par1 / 2.0  )


                #log_prob_0_1[1] = log(1.0 - prob_zeta_mcmc[i, j]) + 2.0 * ((nu - maximum(n_zero_vec_prop) - n_uno_vec_prop[i]) / 2.0 - 1.0) * log(bmat[j, j]) + ((nu - maximum(n_zero_vec_prop) - n_uno_vec_prop[i]) / 2.0)*log(0.5) - loggamma( (nu - maximum(n_zero_vec_prop)- n_uno_vec_prop[i]) / 2.0 )
                #log_prob_0_1[2] = log(prob_zeta_mcmc[i, j]) + 2.0 * ((nu - maximum(n_zero_vec) - n_uno_vec[i]) / 2.0 - 1.0) * log(bmat[j, j]) + ((nu - maximum(n_zero_vec) - n_uno_vec[i]) / 2.0)*log(0.5) - loggamma( (nu - maximum(n_zero_vec)- n_uno_vec[i]) / 2.0 )
                
                nzero_0_1[2] = sum_zero
                nzero_0_1[1] = sum_zero+1


                ### non sparsa
                #log_prob_0_1[2] += sum(like_acc[1:k])
                #log_prob_0_1[2] = 0.0
                #for iiii = 1:k
                #    log_prob_0_1[2] += like_acc[iiii]
                #end
                log_prob_0_1[2] += sum_like_acc
                ### sparsa
                z_prop[i,j] = 0.0
                from_qmat_to_qstar_triangularindex(i, j, qmat, qstar_prop, k, n, psi, z_prop)
                #for jcol = j:(k-1)
                #    compute_like_for_z(jcol, n, qstar_prop, like_prop, z_prop,k, ycentered )
                #end
                compute_like_for_z_all(j, n, qstar_prop, like_prop, z_prop, k ,ycentered, model_sparse, j, ei, app_x,var_mcmc)
                #for jcol = 1:(k)
                #    log_prob_0_1[1] += like_prop[jcol]
                #end
                sum_like_prop = sum(like_prop)
                log_prob_0_1[1] += sum_like_prop
                #log_prob_0_1[1] += sum(like_prop[j:(k-1)])


                #@assert !isnan(log_prob_0_1[1])
                if isnan(log_prob_0_1[1])
                    samp = 2
                else
                    samp = argmax(log_prob_0_1 .+ rand(Gumbel(0,1))) 
                end
                
                sum_zero = nzero_0_1[samp]
                z_mcmc[i,j] = Float64(samp-1)

                if z_mcmc[i,j] == 0.0 ## sparsa, nuovo
                    
                    # ! colonna
                    #n_uno_vec[i] = n_uno_vec_prop[i]
                    #n_zero_vec[i] = n_zero_vec_prop[i]
                    # ! riga
                    n_uno_vec[j] = n_uno_vec_prop[j]
                    n_zero_vec[j] = n_zero_vec_prop[j]

                    #like_acc[1:k] = like_prop[1:k] 
                    for iiii = 1:k
                        like_acc[iiii] = like_prop[iiii]
                    end
                    sum_like_acc = sum_like_prop
                    #qstar .= qstar_prop
                    #for irow = (j+1):k
                    #    for jcol = j:(k-1)
                    #        qstar[irow,jcol] = qstar_prop[irow,jcol]
                    #    end
                    #end

                    model_sparse.varvec .= model_sparse.varvec_p
                    qstar[i,j] = qstar_prop[i,j]
                    if (j+1) < i
                        for jcol  = (j+1):(i-1)
                            for irow = i:k
                                qstar[irow,jcol] = qstar_prop[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end
                    if i<k
                        for jcol = i:(k-1)                            
                            for irow = (jcol+1):k
                                qstar[irow,jcol] = qstar_prop[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end

                else # noo sparsa, vecchio valore
                    # ! colonna
                    #n_uno_vec_prop[i] = n_uno_vec[i]
                    #n_zero_vec_prop[i] = n_zero_vec[i]
                    # ! riga
                    n_uno_vec_prop[j] = n_uno_vec[j]
                    n_zero_vec_prop[j] = n_zero_vec[j]
                    #sum_like_acc = sum_like_prop
                    z_prop[i,j] = 1.0
                    #like_prop[1:k] = like_acc[1:k]
                    for iiii = 1:k
                        like_prop[iiii] = like_acc[iiii]
                    end

                    #qstar_prop .= qstar
                    #for irow = (j+1):k
                    #    for jcol = j:(k-1)
                    #        qstar_prop[irow,jcol] = qstar[irow,jcol]
                    #    end
                    #end

                    qstar_prop[i,j] = qstar[i,j]
                    if (j+1) < i
                        for jcol  = (j+1):(i-1)
                            for irow = i:k
                                qstar_prop[irow,jcol] = qstar[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end
                    if i<k
                        for jcol = i:(k-1)                            
                            for irow = (jcol+1):k
                                qstar_prop[irow,jcol] = qstar[irow,jcol]
                                #@assert irow<= k
                                #@assert jcol <= k
                                #@assert i>j
                            end
                        end
                    end
                    
                end
                
            end

            

        end

    end
    from_bstar_to_b_mat(bstar_mcmc, b_mat_mcmc)
    from_bmat_to_all_q(b_mat_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, z_mcmc, psi)
    #lambda_mcmc[:, :] = q_mat_sparse_mcmc * transpose(q_mat_sparse_mcmc)
    mul!(lambda_mcmc, q_mat_sparse_mcmc, transpose(q_mat_sparse_mcmc))
end

