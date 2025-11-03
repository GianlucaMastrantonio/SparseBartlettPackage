


k = 10


q = rand(Normal(0.0, 1.0), Int64((k^2 - k) / 2))

function test_der(q)

  ret::Float64 = 0.0
  k::Int64 = size(z_mcmc, 1)
  n::Int64 = size(ycentered, 2)

  #bmat::Matrix{Float64} = zeros(Float64, k,k) 
  #qmat::Matrix{Float64} = zeros(Float64, k,k) 
  qstar::Matrix{Float64} = zeros(Float64, k, k)



  ret += -molt_const
  ###### B
  h::Int64 = 1

  ## j = 1
  qstar[1, 1] = exp(bstar_mcmc[h])

  h += 1

  for i = 2:k
    #n_uno += (1-z_mcmc[i,1])
    qstar[i, 1] = bstar_mcmc[h]
    ret += -0.5 * bstar_mcmc[h]^2.0
    h += 1
  end
  ret += 2.0 * ((nu - n_uno_vec[1]) / 2.0 - 1.0) * log(qstar[1, 1]) - qstar[1, 1]^2 / 2.0 + 2.0 * log(qstar[1, 1])
  for j = 2:(k-1)
    qstar[j, j] = exp(bstar_mcmc[h])

    #for i = 1:(j-1)
    #    n_uno += (1-z_mcmc[j,i])
    #end

    h += 1
    for i = (j+1):k
      #n_uno += (1-z_mcmc[i,j])
      qstar[i, j] = bstar_mcmc[h]
      ret += -0.5 * bstar_mcmc[h]^2.0
      h += 1
    end
    ret += 2.0 * ((nu - n_uno_vec[j]) / 2.0 - 1.0) * log(qstar[j, j]) - qstar[j, j]^2 / 2.0 + 2.0 * log(qstar[j, j])
  end
  qstar[k, k] = exp(bstar_mcmc[h])
  ret += 2.0 * ((nu - n_uno_vec[k]) / 2.0 - 1.0) * log(qstar[k, k]) - qstar[k, k]^2 / 2.0 + 2.0 * log(qstar[k, k])


  ###### Q = PsiB
  for j = 1:k
    for i = j:k
      qstar[i, j] = sum(psi[i, j:k] .* qstar[j:k, j]) # mettere solo in funzione di bstar_mcmc
    end
  end

  ##### Q with zero in lambda
  qstar[1, 1] = z_mcmc[1, 1] * qstar[1, 1]
  for i = 2:k
    qstar[i, 1] = z_mcmc[i, 1] * qstar[i, 1]
  end
  for j = 2:(k-1)
    qstar[j, j] = z_mcmc[j, j] * qstar[j, j]
    for i = (j+1):k
      qstar[i, j] = z_mcmc[i, j] * qstar[i, j] + (1.0 - z_mcmc[i, j]) * (-sum(qstar[i, 1:(j-1)] .* qstar[j, 1:(j-1)]) / qstar[j, j])
    end
  end
  qstar[k, k] = z_mcmc[k, k] * qstar[k, k]

  ## like

  for iobs = 1:n
    for j = 1:k
      ret += log(qstar[j, j]) - 0.5 * (sum(ycentered[j:k, iobs] .* qstar[j:k, j]))^2.0
    end
  end



  return ret

end