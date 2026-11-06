## In-place derangement (single cycle) on selected positions of v.
## No new vector is created; it just swaps elements of v at the given positions.
#function derange_on_positions!(v::Vector{Int64}, pos::Vector{Int64})
#  m = length(pos)
#  @inbounds for i = m:-1:2
#    j = rand(1:i-1)            # Sattolo: j < i
#    p_i = pos[i]
#    p_j = pos[j]
#    v[p_i], v[p_j] = v[p_j], v[p_i]
#  end
#end

## In-place derangement (single cycle) on selected positions of v.
## No new vector is created; it just swaps elements of v at the given positions.
#function derange_on_positions!(v::Vector{Int64}, pos::Vector{Int64})
#  m = length(pos)
#  @inbounds for i = m:-1:2
#    j = rand(1:i-1)            # Sattolo: j < i
#    p_i = pos[i]
#    p_j = pos[j]
#    v[p_i], v[p_j] = v[p_j], v[p_i]
#  end
#end


function change_order_mcmc!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE, type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, order_mcmc::order_lambda_V3, psi_inv::Matrix{Float64}, covariates::Array{Float64,3}, order_matrix_mcmc::Matrix{Float64}, is_diag_psi::Bool, var_mcmc::Vector{Float64}, var_prop::Vector{Float64}) where {T_SPARSE<:TypeSparse,T_HMC<:TypeMCMC}

  

  k_half = order_mcmc.k_half
  index_switch = order_mcmc.index_switch

  order_vec = order_mcmc.order_vec
  order_mat = order_mcmc.order_mat
  lambda_prop = order_mcmc.lambda_propc
  q_mat_sparse_prop = order_mcmc.q_mat_sparse_prop
  q_mat_prop = order_mcmc.q_mat_prop
  b_mat_prop = order_mcmc.b_mat_prop
  bstar_prop = order_mcmc.bstar_prop
  zeta_prop = order_mcmc.zeta_prop
  psi_inv_prop = order_mcmc.psi_inv_prop
  psi_prop = order_mcmc.psi_prop

  n_zero = order_mcmc.n_zero
  pos_zero_mcmc = order_mcmc.pos_zero_mcmc
  pos_zero_prop = order_mcmc.pos_zero_prop
  p_prop = order_mcmc.p_prop
  p_mcmc = order_mcmc.p_mcmc
  q_mat_sparse_prop_with_p = order_mcmc.q_mat_sparse_prop_with_p

  psi_psit_mcmc = order_mcmc.psi_psit_mcmc
  psi_psit_prop= order_mcmc.psi_psit_prop

  ycentered_prop = order_mcmc.ycentered_prop

  order_vec .= 0
  order_mat .= 0
  lambda_prop .= 0
  q_mat_sparse_prop .= 0
  q_mat_sparse_prop_with_p .= 0
  q_mat_prop .= 0
  b_mat_prop .= 0
  bstar_prop .= 0
  zeta_prop .= 1.0
  psi_inv_prop .= 0.0 
  psi_prop .= 0.0
  n_zero .= 0
  pos_zero_mcmc .= 0
  pos_zero_prop .= 0
  p_prop .= 0.0
  p_mcmc .= 0.0
  psi_psit_mcmc .= 0.0
  psi_psit_prop .= 0.0

  k::Int64 = size(z_mcmc, 1)
  
ei::Vector{Float64} = zeros(Float64, k)
    app_x::Vector{Float64} = zeros(Float64, k)
  
  # NOTE da decommentare se serve
  order_vec .= 1:k
  if rand(Uniform(0.0,1.0))< 1.5
    
    index_switch[1:2] = sample(1:k, 2, replace=false)
    order_vec[index_switch[1]], order_vec[index_switch[2]] = order_vec[index_switch[2]], order_vec[index_switch[1]]

  else
    order_vec .= sample(1:k, k, replace=false)
  end
  


  order_mat .= 0.0
  @inbounds for i = 1:k
    order_mat[i, order_vec[i]] = 1
  end

 
  mul!(ycentered_prop, order_mat, ycentered)
  mul!(var_prop, order_mat, var_mcmc)

  
  #@assert isapprox(order_mat * order_mat', I)
  
  zeta_prop .= order_mat * z_mcmc * order_mat'
  for i = 1:k
    for j = i:k
      if zeta_prop[i, j] == 0.0

        zeta_prop[j, i] = 0.0
        zeta_prop[i, j] = 1.0

      end
    end
  end

  n_zero[1] = 0
  for j = 1:k
    for i = j:k
      if zeta_prop[i, j] == 0.0
        n_zero[1] += 1
        pos_zero_prop[n_zero[1], 1] = i
        pos_zero_prop[n_zero[1], 2] = j
      end
    end
  end
  #save_n_zero::Int64 = n_zero[1]
  n_zero[1] = 0
  for j = 1:k
    for i = j:k
      if z_mcmc[i, j] == 0.0
        n_zero[1] += 1
        pos_zero_mcmc[n_zero[1], 1] = i
        pos_zero_mcmc[n_zero[1], 2] = j
      end
    end
  end
  #@assert save_n_zero == n_zero[1]

  
  if is_diag_psi
    psi_prop .= order_mat * psi * order_mat'
    for ii = 1:k
      psi_inv_prop[ii,ii] = 1.0 / psi_prop[ii,ii]
    end
    
  else

    psi_psit_mcmc .= psi * transpose(psi)
    psi_psit_prop .= order_mat * psi_psit_mcmc * order_mat'
    psi_prop .= cholesky(Symmetric(psi_psit_prop)).L
    psi_inv_prop .= inv(psi_prop)

  end
  



  #psi_inv_prop .= cholesky(psi_psit_prop).L

  #psi_inv_prop .= order_mat * psi_inv * order_mat'
  #psi_prop .= order_mat * psi * order_mat'
  
  ### creo la matrice Lambda_n che dipende dalla q non zero e dalle p, devo prima simulare le p
  sd_norm::Float64 = rand(Uniform(0.000000000000001, 0.00000000000001))
  if  n_zero[1] > 0
    p_mcmc[1:n_zero[1]] .= 0.0#rand(Normal(0.0, sd_norm), n_zero[1])
    #p_mcmc[1:n_zero[1]] .= 0.0
  end
  
  ## adesso aggiungo alle p deterministiche le p
  q_mat_sparse_prop_with_p .= q_mat_sparse_mcmc
  #h::Int64 = 0
  if  n_zero[1] > 0
    for i = 1:n_zero[1]
      q_mat_sparse_prop_with_p[pos_zero_mcmc[i, 1], pos_zero_mcmc[i, 2]] += p_mcmc[i]
    end
  end
  
  ## costruisco la Lambda proposta, con e p, e cambiuo ordine righe/colonne
  lambda_prop .= order_mat * (q_mat_sparse_prop_with_p * transpose(q_mat_sparse_prop_with_p)) * order_mat'

  ## costruisco la q con p- poi la ricalcolo da B e mi trovo le nuove p
  try

    q_mat_sparse_prop_with_p[:, :] .= cholesky(Symmetric(lambda_prop)).L
  
  catch e
    if isa(e, PosDefException)
        # skip or handle gracefully
        return nothing
    else
        rethrow(e)
    end
  end
  
  for j = 1:k
    for i = j:k
      #q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]

      if zeta_prop[i, j] == 1.0

        q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]

      else

        q_mat_prop[i, j] = rand(Normal(0.0,1.0))/psi_inv_prop[i, i]

      end
    end
  end

  for j = 1:k
    for i = j:k
      b_mat_prop[i, j] = sum(psi_inv_prop[i, j:k] .* q_mat_prop[j:k, j])
      #b_mat_sim_c[i, j] = sum(psi_inv[i, :] .* q_mat_sim_c[j, :])
    end
  end
  h = 0
  for j = 1:k
    for i = j:k
      h += 1
      if j == i
        bstar_prop[h] = log(b_mat_prop[i, j])
      else
        bstar_prop[h] = b_mat_prop[i, j]
      end
    end
  end

  #for j = 1:k
  #  for i = j:k
  #    #q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]
  #    # NOTE: questo va capito
  #    if zeta_prop[i, j] == 1.0

  #      q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]

  #    else

  #      q_mat_prop[i, j] = q_mat_mcmc[i, j]

  #    end
  #  end
  #end



  from_bstar_to_b_mat(bstar_prop, b_mat_prop)
  from_bmat_to_all_q(b_mat_prop, q_mat_prop, q_mat_sparse_prop, zeta_prop, psi_prop)
  lambda_prop .= q_mat_sparse_prop * transpose(q_mat_sparse_prop)
  #@assert isapprox(lambda_prop, q_mat_sparse_prop * transpose(q_mat_sparse_prop))

  ## tiro fuori le nuove p
  if  n_zero[1] > 0
    for i = 1:n_zero[1]
      p_prop[i] = q_mat_sparse_prop_with_p[pos_zero_prop[i, 1], pos_zero_prop[i, 2]] - q_mat_sparse_prop[pos_zero_prop[i, 1], pos_zero_prop[i, 2]]
    end
  end
  


  ## check if accept
  
  n::Int64 = size(ydata, 2)
  # ! numeri 1 per colonna
  #n_uno = zeros(Float64, k)
  #for i = 1:(k-1)
  #  for j = (i+1):(k)
  #    n_uno[i] -= Float64(z_mcmc[j, i])
  #    #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
  #  end
  #end

  add_to_v = zeros(Float64, k)
  for i = 1:(k-1)
    for j = (i+1):(k)
      add_to_v[i] += Float64(z_mcmc[j, i])
      #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
    end
  end
  add_to_v_prop = zeros(Float64, k)
  for i = 1:(k-1)
    for j = (i+1):(k)
      add_to_v_prop[i] += Float64(zeta_prop[j, i])
      #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
    end
  end
  add_to_v_jacob_prop = deepcopy(add_to_v_prop)
  add_to_v_jacob = deepcopy(add_to_v)
  nu::Float64 = nu_orig - k + 1

  log_mh_ratio::Float64 = 0.0
  
  for j = 1:k
    for i = j:k

      if j == i

        # NOTE: propongo in b
        #log_mh_ratio += 2.0 * ((nu + add_to_v_prop[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 #+ 2.0 * log(b_mat_prop[j, j])
        
        #log_mh_ratio +=    (nu + add_to_v_prop[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v_prop[j]) / 2.0  )
        
        #log_mh_ratio += -2.0*(1.0+  add_to_v_prop[j]) * log(b_mat_prop[j, j]) # determinante della trasformazione 


        #log_mh_ratio -= 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0# + 2.0 * log(b_mat_mcmc[j, j])
        #log_mh_ratio -=    (nu + add_to_v[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v[j]) / 2.0  )
        #log_mh_ratio -= -2.0*(1.0+  add_to_v[j]) * log(b_mat_mcmc[j, j]) # determinante della trasformazione 

        # ! versione nuova
        log_mh_ratio += 2.0 * ((nu + add_to_v_prop[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 + log(b_mat_prop[j, j])
        
        log_mh_ratio +=    (nu + add_to_v_prop[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v_prop[j]) / 2.0  )
        
        log_mh_ratio += -(1.0+  add_to_v_jacob_prop[j]) * log(b_mat_prop[j, j]) # determinante della trasformazione 


        log_mh_ratio -= 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0+ log(b_mat_mcmc[j, j])
        log_mh_ratio -=    (nu + add_to_v[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v[j]) / 2.0  )
        log_mh_ratio -= -(1.0+  add_to_v_jacob[j]) * log(b_mat_mcmc[j, j]) # determinante della trasformazione 



        #log_mh_ratio += 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 + 2.0 * log(b_mat_prop[j, j])

        #log_mh_ratio -= 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0 + 2.0 * log(b_mat_mcmc[j, j])


      else

        #log_mh_ratio += -0.5 * b_mat_prop[i, j]^2.0
        #log_mh_ratio -= -0.5 * b_mat_mcmc[i, j]^2.0
        if zeta_prop[i, j] == 1.0
          log_mh_ratio += logpdf(model_sparse.distr, b_mat_prop[i,j])
        end

        if z_mcmc[i, j] == 1.0
          log_mh_ratio -= logpdf(model_sparse.distr, b_mat_mcmc[i,j])
        end
        
        

      end



    end
  end
  
  if  n_zero[1] > 0
    for i = 1:n_zero[1]
      #log_mh_ratio += -0.5 * (p_prop[i]^2.0 / (sd_norm^2.0))
      #log_mh_ratio -= -0.5 * (p_mcmc[i]^2.0 / (sd_norm^2.0))
    end
  end
  
  ## likelihood
  app_mh::Float64 = 0.0
  marginal_variances(model_sparse, q_mat_sparse_prop, ei, app_x, model_sparse.varvec_p)
  #app = 0.0
  #for iobs in 1:n
  #  for jcol = 1:k

  #    app_mh = 0.0
  #    for i_app in jcol:k
  #      app_mh += ycentered_prop[i_app, iobs] .* q_mat_sparse_prop[i_app, jcol]*model_sparse.varvec_p[i_app]^0.5/ var_prop[i_app]^0.5
  #    end
  #    log_mh_ratio += log(q_mat_sparse_prop[jcol, jcol]*model_sparse.varvec_p[jcol]^0.5/var_prop[jcol]^0.5) - 0.5 * app_mh^2.0
  #    #app += log(q_mat_sparse_prop[jcol, jcol]) - 0.5 * app_mh^2.0


  #    app_mh = 0.0
  #    for i_app in jcol:k
  #      app_mh += ycentered[i_app, iobs] .* q_mat_sparse_mcmc[i_app, jcol]*model_sparse.varvec[i_app]^0.5/ var_mcmc[i_app]^0.5
  #    end
  #    log_mh_ratio -= log(q_mat_sparse_mcmc[jcol, jcol]*model_sparse.varvec[jcol]^0.5/var_mcmc[jcol]^0.5) - 0.5 * app_mh^2.0
  #    #  app -= log(q_mat_sparse_mcmc[jcol, jcol]) - 0.5 * app_mh^2.0

  #    #log_mh_ratio += q_mat_sparse_prop[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_prop[jcol:k, jcol]))^2.0
  #    #log_mh_ratio -= q_mat_sparse_mcmc[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_mcmc[jcol:k, jcol]))^2.0
  #  end
  #end
  
  #if iterMCMC  == 2

  #  println("Test ", app)
  #  println(exp(log_mh_ratio))
  #  println("lambda diff ", sum(order_mat * lambda_mcmc * order_mat' - lambda_prop))
  #  println("diff  lambda1 ", sum(lambda_prop - q_mat_sparse_prop * q_mat_sparse_prop'))
  #  println("diff  lambda2 ", sum(lambda_mcmc - q_mat_sparse_mcmc * q_mat_sparse_mcmc'))
  #  println("q diff ", sum(q_mat_sparse_mcmc - q_mat_sparse_prop) )
    

  #  #println("q1=", q_mat_sparse_mcmc)
  #  #println("q2=", q_mat_sparse_prop)


  #  println("p diff ", sum(p_mcmc - p_prop) )

  #  error()
  #end
  
  
  if rand(Uniform(0.0,1.0))< exp(log_mh_ratio)
    
    #println("Accepted change order at iter ", iterMCMC)

    
    psi_inv .= psi_inv_prop
    bstar_mcmc .= bstar_prop
    z_mcmc .= zeta_prop
    psi .= psi_prop
    q_mat_mcmc .= q_mat_prop
    q_mat_sparse_mcmc .= q_mat_sparse_prop
    lambda_mcmc .= lambda_prop
    b_mat_mcmc .= b_mat_prop
    mu_mcmc .= order_mat * mu_mcmc
    ycentered .= ycentered_prop
    var_mcmc .= var_prop
    ydata .= order_mat * ydata
    #covariates .= order_mat * covariates
    for i in axes(covariates, 2)
      @views covariates[:, i, :] .= order_mat * covariates[:, i, :]
    end
    order_matrix_mcmc .= order_mat * order_matrix_mcmc


  end



end

 



## SECTION: Change Order MCMC

#function old_change_order_mcmc!(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::T_SPARSE, type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, order_mcmc::order_lambda_V3, psi_inv::Matrix{Float64}, covariates::Array{Float64,3}, order_matrix_mcmc::Matrix{Float64}, is_diag_psi::Bool) where {T_SPARSE<:TypeSparse,T_HMC<:TypeMCMC}

#  #zeta_prop::Matrix{Float64}
#  #bstar_prop::Vector{Float64}
#  #q_mat_prop::Matrix{Float64}
#  #q_mat_sparse_prop::Matrix{Float64}
#  #lambda_propc::Matrix{Float64}
#  #b_mat_prop::Matrix{Float64}

#  #order_vec::Vector{Int64}
#  #order_mat::Matrix{Float64}
#  #inv_order_mat::Matrix{Float64}
#  #inv_order_vec::Vector{Int64}

#  #n_accepted::Vector{Int64}
#  #n_total::Vector{Int64}
#  # = order_mcmc.
#  #psi_inv_prop = fill(0.0, k, k),
#  #psi_prop = fill(0.0, k, k)

#  k_half = order_mcmc.k_half
#  index_switch = order_mcmc.index_switch

#  order_vec = order_mcmc.order_vec
#  order_mat = order_mcmc.order_mat
#  lambda_prop = order_mcmc.lambda_propc
#  q_mat_sparse_prop = order_mcmc.q_mat_sparse_prop
#  q_mat_prop = order_mcmc.q_mat_prop
#  b_mat_prop = order_mcmc.b_mat_prop
#  bstar_prop = order_mcmc.bstar_prop
#  zeta_prop = order_mcmc.zeta_prop
#  psi_inv_prop = order_mcmc.psi_inv_prop
#  psi_prop = order_mcmc.psi_prop

#  n_zero = order_mcmc.n_zero
#  pos_zero_mcmc = order_mcmc.pos_zero_mcmc
#  pos_zero_prop = order_mcmc.pos_zero_prop
#  p_prop = order_mcmc.p_prop
#  p_mcmc = order_mcmc.p_mcmc
#  q_mat_sparse_prop_with_p = order_mcmc.q_mat_sparse_prop_with_p

#  psi_psit_mcmc = order_mcmc.psi_psit_mcmc
#  psi_psit_prop= order_mcmc.psi_psit_prop

#  ycentered_prop = order_mcmc.ycentered_prop

#  order_vec .= 0
#  order_mat .= 0
#  lambda_prop .= 0
#  q_mat_sparse_prop .= 0
#  q_mat_sparse_prop_with_p .= 0
#  q_mat_prop .= 0
#  b_mat_prop .= 0
#  bstar_prop .= 0
#  zeta_prop .= 1.0
#  psi_inv_prop .= 0.0 
#  psi_prop .= 0.0
#  n_zero .= 0
#  pos_zero_mcmc .= 0
#  pos_zero_prop .= 0
#  p_prop .= 0.0
#  p_mcmc .= 0.0
#  psi_psit_mcmc .= 0.0
#  psi_psit_prop .= 0.0

  
#  #k::Int64 = size(z_mcmc, 1)
#  #n_dim::Int64 = size(index_switch, 1)
#  #n_switch::Int64 = sample(1:n_dim, 1)[1]
#  #index_switch[:] = sample(1:(n_dim*2), n_dim * 2, replace=false)
  
#  #order_vec .= 1:k
  
#  #pos_i::Int64 = 0
#  #pos_j::Int64 = 0
#  #for i = 1:n_switch
#  #  pos_i = index_switch[i,1]
#  #  pos_j = index_switch[i,2]
#  #  order_vec[pos_i], order_vec[pos_j] = order_vec[pos_j], order_vec[pos_i]
#  #end

#  k::Int64 = size(z_mcmc, 1)
#  #n_dim::Int64 = size(index_switch, 1)
#  #n_switch::Int64 = sample(1:n_dim, 1)[1]
  
#  #index_switch[:] = sample(1:(n_dim*2), n_dim * 2, replace=false)
#ei::Vector{Float64} = zeros(Float64, k)
#    app_x::Vector{Float64} = zeros(Float64, k)
  
#  # NOTE da decommentare se serve
#  order_vec .= 1:k
#  if rand(Uniform(0.0,1.0))< 1.5
    
#    index_switch[1:2] = sample(1:k, 2, replace=false)
#    order_vec[index_switch[1]], order_vec[index_switch[2]] = order_vec[index_switch[2]], order_vec[index_switch[1]]

#  else
#    order_vec .= sample(1:k, k, replace=false)
#  end
  

  
#  #order_vec[index_switch[1]], order_vec[index_switch[2]] = order_vec[index_switch[2]], order_vec[index_switch[1]]
  
  
  
  
#  #order_vec[index_switch[1]], order_vec[index_switch[2]] = order_vec[index_switch[1]], order_vec[index_switch[2]]

#  #pos_i::Int64 = 0
#  #pos_j::Int64 = 0
#  #for i = 1:n_switch
#  #  pos_i = index_switch[i, 1]
#  #  pos_j = index_switch[i, 2]
#  #  order_vec[pos_i], order_vec[pos_j] = order_vec[pos_j], order_vec[pos_i]
#  #end
  
#  #order_vec[index_switch] = shuffle(index_switch)
#  order_mat .= 0.0
#  @inbounds for i = 1:k
#    order_mat[i, order_vec[i]] = 1
#  end

 
#  mul!(ycentered_prop, order_mat, ycentered)
#  #@assert isapprox(order_mat * order_mat', I)
  
#  zeta_prop .= order_mat * z_mcmc * order_mat'
#  for i = 1:k
#    for j = i:k
#      if zeta_prop[i, j] == 0.0

#        zeta_prop[j, i] = 0.0
#        zeta_prop[i, j] = 1.0

#      end
#    end
#  end

#  n_zero[1] = 0
#  for j = 1:k
#    for i = j:k
#      if zeta_prop[i, j] == 0.0
#        n_zero[1] += 1
#        pos_zero_prop[n_zero[1], 1] = i
#        pos_zero_prop[n_zero[1], 2] = j
#      end
#    end
#  end
#  #save_n_zero::Int64 = n_zero[1]
#  n_zero[1] = 0
#  for j = 1:k
#    for i = j:k
#      if z_mcmc[i, j] == 0.0
#        n_zero[1] += 1
#        pos_zero_mcmc[n_zero[1], 1] = i
#        pos_zero_mcmc[n_zero[1], 2] = j
#      end
#    end
#  end
#  #@assert save_n_zero == n_zero[1]

  
#  if is_diag_psi
#    psi_prop .= order_mat * psi * order_mat'
#    for ii = 1:k
#      psi_inv_prop[ii,ii] = 1.0 / psi_prop[ii,ii]
#    end
    
#  else

#    psi_psit_mcmc .= psi * transpose(psi)
#    psi_psit_prop .= order_mat * psi_psit_mcmc * order_mat'
#    psi_prop .= cholesky(Symmetric(psi_psit_prop)).L
#    psi_inv_prop .= inv(psi_prop)

#  end
  



#  #psi_inv_prop .= cholesky(psi_psit_prop).L

#  #psi_inv_prop .= order_mat * psi_inv * order_mat'
#  #psi_prop .= order_mat * psi * order_mat'
  
#  ### creo la matrice Lambda_n che dipende dalla q non zero e dalle p, devo prima simulare le p
#  sd_norm::Float64 = rand(Uniform(0.000000000000001, 0.00000000000001))
#  if  n_zero[1] > 0
#    p_mcmc[1:n_zero[1]] .= rand(Normal(0.0, sd_norm), n_zero[1])
#    #p_mcmc[1:n_zero[1]] .= 0.0
#  end
  
#  ## adesso aggiungo alle p deterministiche le p
#  q_mat_sparse_prop_with_p .= q_mat_sparse_mcmc
#  #h::Int64 = 0
#  if  n_zero[1] > 0
#    for i = 1:n_zero[1]
#      q_mat_sparse_prop_with_p[pos_zero_mcmc[i, 1], pos_zero_mcmc[i, 2]] += p_mcmc[i]
#    end
#  end
  
#  ## costruisco la Lambda proposta, con e p, e cambiuo ordine righe/colonne
#  lambda_prop .= order_mat * (q_mat_sparse_prop_with_p * transpose(q_mat_sparse_prop_with_p)) * order_mat'

#  ## costruisco la q con p- poi la ricalcolo da B e mi trovo le nuove p
#  try

#    q_mat_sparse_prop_with_p[:, :] .= cholesky(Symmetric(lambda_prop)).L
  
#  catch e
#    if isa(e, PosDefException)
#        # skip or handle gracefully
#        return nothing
#    else
#        rethrow(e)
#    end
#  end
  
#  for j = 1:k
#    for i = j:k
#      #q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]
#      # NOTE: questo va capito
#      if zeta_prop[i, j] == 1.0

#        q_mat_prop[i, j] = q_mat_sparse_prop_with_p[i, j]

#      else

#        q_mat_prop[i, j] = q_mat_mcmc[i, j]

#      end
#    end
#  end

#  for j = 1:k
#    for i = j:k
#      b_mat_prop[i, j] = sum(psi_inv_prop[i, j:k] .* q_mat_prop[j:k, j])
#      #b_mat_sim_c[i, j] = sum(psi_inv[i, :] .* q_mat_sim_c[j, :])
#    end
#  end
#  h = 0
#  for j = 1:k
#    for i = j:k
#      h += 1
#      if j == i
#        bstar_prop[h] = log(b_mat_prop[i, j])
#      else
#        bstar_prop[h] = b_mat_prop[i, j]
#      end
#    end
#  end

  



#  from_bstar_to_b_mat(bstar_prop, b_mat_prop)
#  from_bmat_to_all_q(b_mat_prop, q_mat_prop, q_mat_sparse_prop, zeta_prop, psi_prop)
#  lambda_prop .= q_mat_sparse_prop * transpose(q_mat_sparse_prop)
#  #@assert isapprox(lambda_prop, q_mat_sparse_prop * transpose(q_mat_sparse_prop))

#  ## tiro fuori le nuove p
#  if  n_zero[1] > 0
#    for i = 1:n_zero[1]
#      p_prop[i] = q_mat_sparse_prop_with_p[pos_zero_prop[i, 1], pos_zero_prop[i, 2]] - q_mat_sparse_prop[pos_zero_prop[i, 1], pos_zero_prop[i, 2]]
#    end
#  end
  


#  ## check if accept
  
#  n::Int64 = size(ydata, 2)
#  # ! numeri 1 per colonna
#  #n_uno = zeros(Float64, k)
#  #for i = 1:(k-1)
#  #  for j = (i+1):(k)
#  #    n_uno[i] -= Float64(z_mcmc[j, i])
#  #    #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
#  #  end
#  #end

#  add_to_v = zeros(Float64, k)
#  for i = 1:(k-1)
#    for j = (i+1):(k)
#      add_to_v[i] += Float64(z_mcmc[j, i])
#      #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
#    end
#  end
#  add_to_v_prop = zeros(Float64, k)
#  for i = 1:(k-1)
#    for j = (i+1):(k)
#      add_to_v_prop[i] += Float64(zeta_prop[j, i])
#      #n_zero[j] += Float64(1.0 - z_mcmc[j,i])
#    end
#  end
#  nu::Float64 = nu_orig - k + 1

#  log_mh_ratio::Float64 = 0.0
  
#  for j = 1:k
#    for i = j:k

#      if j == i

#        # NOTE: propongo in b^2
#        log_mh_ratio += 2.0 * ((nu + add_to_v_prop[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 #+ 2.0 * log(b_mat_prop[j, j])
        
#        log_mh_ratio +=    (nu + add_to_v_prop[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v_prop[j]) / 2.0  )
        
#        log_mh_ratio += -2.0*(1.0+  add_to_v_prop[j]) * log(b_mat_prop[j, j])


#        log_mh_ratio -= 2.0 * ((nu + add_to_v[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0# + 2.0 * log(b_mat_mcmc[j, j])
#        log_mh_ratio -=    (nu + add_to_v[j]) / 2.0 * log(0.5)  -  loggamma( (nu + add_to_v[j]) / 2.0  )
#        log_mh_ratio -= -2.0*(1.0+  add_to_v[j]) * log(b_mat_mcmc[j, j])



#        #log_mh_ratio += 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_prop[j, j]) - b_mat_prop[j, j]^2 / 2.0 + 2.0 * log(b_mat_prop[j, j])

#        #log_mh_ratio -= 2.0 * ((nu - n_uno[j]) / 2.0 - 1.0) * log(b_mat_mcmc[j, j]) - b_mat_mcmc[j, j]^2 / 2.0 + 2.0 * log(b_mat_mcmc[j, j])


#      else

#        log_mh_ratio += logpdf(model_sparse.distr, b_mat_prop[i, j]) 
#        log_mh_ratio -= logpdf(model_sparse.distr, b_mat_mcmc[i, j]) 

        

#      end



#    end
#  end
  
#  if  n_zero[1] > 0
#    for i = 1:n_zero[1]
#      log_mh_ratio += -0.5 * (p_prop[i]^2.0 / (sd_norm^2.0))
#      log_mh_ratio -= -0.5 * (p_mcmc[i]^2.0 / (sd_norm^2.0))
#    end
#  end
  
#  ## likelihood
#  app_mh::Float64 = 0.0
#  #app = 0.0
#  marginal_variances(model_sparse, q_mat_sparse_prop, ei, app_x, model_sparse.varvec_p)
#  for iobs in 1:n
#    for jcol = 1:k

#      app_mh = 0.0
#      for i_app in jcol:k
#        app_mh += ycentered_prop[i_app, iobs] .* q_mat_sparse_prop[i_app, jcol]/model_sparse.varvec_p[i_app]^0.5
#      end
#      log_mh_ratio += log(q_mat_sparse_prop[jcol, jcol]/model_sparse.varvec_p[jcol]^0.5) - 0.5 * app_mh^2.0
#      #app += log(q_mat_sparse_prop[jcol, jcol]) - 0.5 * app_mh^2.0


#      app_mh = 0.0
#      for i_app in jcol:k
#        app_mh += ycentered[i_app, iobs] .* q_mat_sparse_mcmc[i_app, jcol]/model_sparse.varvec[i_app]^0.5
#      end
#      log_mh_ratio -= log(q_mat_sparse_mcmc[jcol, jcol]/model_sparse.varvec[jcol]^0.5) - 0.5 * app_mh^2.0
#      #  app -= log(q_mat_sparse_mcmc[jcol, jcol]) - 0.5 * app_mh^2.0

#      #log_mh_ratio += q_mat_sparse_prop[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_prop[jcol:k, jcol]))^2.0
#      #log_mh_ratio -= q_mat_sparse_mcmc[jcol, jcol] - 0.5 * (sum(ycentered[jcol:k, iobs] .* q_mat_sparse_mcmc[jcol:k, jcol]))^2.0
#    end
#  end
  
#  #if iterMCMC  == 2

#  #  println("Test ", app)
#  #  println(exp(log_mh_ratio))
#  #  println("lambda diff ", sum(order_mat * lambda_mcmc * order_mat' - lambda_prop))
#  #  println("diff  lambda1 ", sum(lambda_prop - q_mat_sparse_prop * q_mat_sparse_prop'))
#  #  println("diff  lambda2 ", sum(lambda_mcmc - q_mat_sparse_mcmc * q_mat_sparse_mcmc'))
#  #  println("q diff ", sum(q_mat_sparse_mcmc - q_mat_sparse_prop) )
    

#  #  #println("q1=", q_mat_sparse_mcmc)
#  #  #println("q2=", q_mat_sparse_prop)


#  #  println("p diff ", sum(p_mcmc - p_prop) )

#  #  error()
#  #end
  
  
#  if rand(Uniform(0.0,1.0))< exp(log_mh_ratio)
    
#    #println("Accepted change order at iter ", iterMCMC)

    
#    psi_inv .= psi_inv_prop
#    bstar_mcmc .= bstar_prop
#    z_mcmc .= zeta_prop
#    psi .= psi_prop
#    q_mat_mcmc .= q_mat_prop
#    q_mat_sparse_mcmc .= q_mat_sparse_prop
#    lambda_mcmc .= lambda_prop
#    b_mat_mcmc .= b_mat_prop
#    mu_mcmc .= order_mat * mu_mcmc
#    ycentered .= ycentered_prop
#    ydata .= order_mat * ydata
#    #covariates .= order_mat * covariates
#    for i in axes(covariates, 2)
#      @views covariates[:, i, :] .= order_mat * covariates[:, i, :]
#    end
#    order_matrix_mcmc .= order_mat * order_matrix_mcmc
#    model_sparse.varvec .= model_sparse.varvec_p

#  end



#end

 