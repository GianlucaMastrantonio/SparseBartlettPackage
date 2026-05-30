function sample_var(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::GenSparse_V2 , type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, order_mcmc::order_lambda_V3, psi_inv::Matrix{Float64}, covariates::Array{Float64,3}, order_matrix_mcmc::Matrix{Float64}, is_diag_psi::Bool, var_mcmc::Vector{Float64}, var_prop::Vector{Float64}) where { T_HMC<:TypeMCMC}
  var_mcmc .= 1.0
  var_prop .= 1.0
  model_sparse.varvec .= 1.0

end

function sample_var(iterMCMC::Int64, ydata::Matrix{Float64}, bstar_mcmc::Vector{Float64}, z_mcmc::Matrix{Float64}, model_sparse::GenCorSparse_V2, type_mcmc_lambda::T_HMC, psi::Matrix{Float64}, nu_orig::Float64, r_hmc_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, lambda_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, mu_mcmc::Matrix{Float64}, ycentered::Matrix{Float64}, order_mcmc::order_lambda_V3, psi_inv::Matrix{Float64}, covariates::Array{Float64,3}, order_matrix_mcmc::Matrix{Float64}, is_diag_psi::Bool, var_mcmc::Vector{Float64}, var_prop::Vector{Float64}) where {T_HMC<:TypeMCMC}


  k::Int64 = size(z_mcmc,1)
  n::Int64 = size(ydata, 2)
  a_ig::Float64 = nu_orig/2.0
  b_ig::Float64 = 0.5*psi_inv[1,1]
  

  mean_norm::Float64 = log(b_ig) - digamma(a_ig)
  var_norm::Float64 =trigamma(a_ig)

  # Draw auxiliary direction from reference Gaussian
  or_dir::Vector{Float64} = rand(Normal(mean_norm,var_norm^0.5), k)

  # Current log slice height
  logy::Float64 = 0.0
  log_prop::Float64 = 0.0
  app_mh::Float64 = 0.0
  for jcol = 1:k
    logy += logpdf(InverseGamma(a_ig, b_ig), var_mcmc[jcol])+ log(var_mcmc[jcol]) - logpdf(Normal(mean_norm, var_norm^0.5), log(var_mcmc[jcol]))
  end
  #if isnan(logy)

  #  error([logpdf(InverseGamma(a_ig, b_ig), var_mcmc[jcol]), log(var_mcmc[jcol]), logpdf(Normal(mean_norm, var_norm^0.5), log(var_mcmc[jcol]))])

  #end

  for iobs in 1:n
    for jcol = 1:k
      app_mh = 0.0
      for i_app in jcol:k
        app_mh += ycentered[i_app, iobs] .* q_mat_sparse_mcmc[i_app, jcol]*model_sparse.varvec[i_app]^0.5/ var_mcmc[i_app]^0.5
        #if isnan(app_mh)
        #  println([ycentered[i_app, iobs] , q_mat_sparse_mcmc[i_app, jcol], model_sparse.varvec[i_app], var_mcmc[i_app]])
        #  error()
        #end
      end
      logy += log(q_mat_sparse_mcmc[jcol, jcol]*model_sparse.varvec[jcol]^0.5/var_mcmc[jcol]^0.5) - 0.5 * app_mh^2.0
      
    end
  end
  logy +=  log(rand(Uniform(0,1)))
  
  # Random bracket
  theta::Float64 = 2π * rand(Uniform(0.0,1.0))
  theta_min::Float64 = theta - 2π
  theta_max::Float64 = theta

  #xc = x_cur .- mu_ref
  #nc = nu .- mu_ref

  while true
    var_prop .= exp.(mean_norm .+ ((log.(var_mcmc) .- mean_norm) .* cos(theta)) .+ ((or_dir .- mean_norm) .* sin(theta)))
    log_prop = 0.0
    for jcol = 1:k
      log_prop += logpdf(InverseGamma(a_ig, b_ig), var_prop[jcol]) +log(var_prop[jcol]) - logpdf(Normal(mean_norm, var_norm^0.5), log(var_prop[jcol]))
    end
    for iobs in 1:n
      for jcol = 1:k
        app_mh = 0.0
        for i_app in jcol:k
          app_mh += ycentered[i_app, iobs] .* q_mat_sparse_mcmc[i_app, jcol]*model_sparse.varvec[i_app]^0.5/ var_prop[i_app]^0.5
        end
        log_prop += log(q_mat_sparse_mcmc[jcol, jcol]*model_sparse.varvec[jcol]^0.5/var_prop[jcol]^0.5) - 0.5 * app_mh^2.0
      end
    end
    

    if log_prop > logy
        var_mcmc .= var_prop
        #println("sss")
        
        return nothing
    end

    if theta < 0.0
        theta_min = theta
    else
        theta_max = theta
    end
    theta = rand(Uniform(0.0,1.0)) * (theta_max - theta_min) + theta_min
  end


end