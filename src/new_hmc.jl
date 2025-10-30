function new_find_reasonable_epsilon(theta_start::Vector{Float64}, log_dens::Function, d_b::Vector{Float64}, epsilon::Vector{Float64}, add_const::Float64)

    iter_gen::Int64 = 0
    r_hmc_start::Vector{Float64} = rand(Normal(0.0, 1.0), size(theta_start, 1))

    theta = deepcopy(theta_start)
    r_hmc = deepcopy(r_hmc_start)
    #epsilon[1] = 1.0

    ltheta_start::Float64 = log_dens(theta)
    #lr_start::Float64 = sum(r_hmc.^2.0) 
    lr_start::Float64 = log_dens_h(r_hmc, add_const)
    fail::Bool = false


    epsilon_start::Float64 = epsilon[1]


    while fail
        iter_gen += 1
        if iter_gen > 1
            epsilon[1] = epsilon[1] * 0.5
            println("New test epsilon -  starting value ", epsilon[1])
        end
        fail = leapfrog(theta, r_hmc, epsilon[1], d_b, log_dens)

        #println("A")
        iterator::Int64 = 0
        while fail

            epsilon[1] = epsilon[1] * 0.5

            fail = retry_leapfrog_type1(theta, r_hmc, theta_start, r_hmc_start, epsilon[1], d_b, log_dens)

            #println([trunc.(theta, digits=3) ])
            iterator += 1
            if iterator > 100
                error("iterator - find_reasonable_epsilon")
            end

        end
        #println("EPX ", epsilon[1])
        #println("B")
        iterator = 0



        a::Float64 = 2.0 * Int64((log_dens(theta) + log_dens_h(r_hmc, add_const) - (ltheta_start + lr_start)) > log(0.5)) - 1.0

        #while (exp(log_dens(theta) - 0.5 * sum(r_hmc.^2.0)) / exp(ltheta_start - 0.5 * lr_start))^a > 2.0^(-a)


        while (a * (log_dens(theta) + log_dens_h(r_hmc, add_const) - (ltheta_start + lr_start)) > -a * log(2.0)) || fail

            theta_start[:] = theta[:]
            r_hmc_start[:] = r_hmc[:]

            epsilon[1] = 2.0^a * epsilon[1]
            println(epsilon[1])
            println([log_dens(theta), -0.5 * sum(r_hmc .^ 2.0), -ltheta_start, -lr_start])
            fail = leapfrog(theta, r_hmc, epsilon[1], d_b, log_dens)

            #while fail
            #    epsilon[1] = epsilon[1] * 0.5
            #    fail =  retry_leapfrog_type1(theta, r_hmc, theta_start, r_hmc_start, epsilon[1], d_b, log_dens)

            #    iterator += 1
            #    if iterator > 100; error("iterator - find_reasonable_epsilon step 2") end

            #end
        end

        if iter_gen > 100
            error("iter_gen - find_reasonable_epsilon step 2")
        end
    end


    theta[:] = theta_start[:]

    #println("A ", sum(d_b))
    return nothing
end


function new_log_dens_h(hvec::Vector{Float64}, add_const::Float64)

    return -0.5 * sum(hvec .^ 2.0) + add_const
end

function new_leapfrog_nocheck(theta::Vector{Float64}, r_hmc::Vector{Float64}, epsilon::Float64, d_b::Vector{Float64}, log_dens::Function)


    gradient!(Reverse, d_b, log_dens, theta)
    r_hmc[:] .= r_hmc .+ (epsilon ./ 2.0) .* d_b

    theta[:] .= theta[:] .+ epsilon .* r_hmc

    gradient!(Reverse, d_b, log_dens, theta)
    r_hmc[:] .= r_hmc .+ (epsilon ./ 2.0) .* d_b

    return nothing

end

function new_leapfrog(theta::Vector{Float64}, r_hmc::Vector{Float64}, epsilon::Float64, d_b::Vector{Float64}, log_dens::Function)::Bool


    gradient!(Reverse, d_b, log_dens, theta)
    r_hmc[:] .= r_hmc + (epsilon / 2.0) .* d_b
    theta[:] .= theta[:] + epsilon * r_hmc
    gradient!(Reverse, d_b, log_dens, theta)
    r_hmc[:] .= r_hmc + (epsilon / 2.0) .* d_b

    return isnan(log_dens(theta)) || isinf(log_dens(theta)) || isnan(sum(r_hmc .^ 2.0)) || isinf(sum(r_hmc .^ 2.0)) || isnan(sum(d_b)) || isinf(sum(d_b))

end

function new_retry_leapfrog_type1(theta::Vector{Float64}, r_hmc::Vector{Float64}, theta_start::Vector{Float64}, r_hmc_start::Vector{Float64}, epsilon::Float64, d_b::Vector{Float64}, log_dens::Function)::Bool

    theta[:] .= theta_start[:]
    r_hmc[:] .= r_hmc_start[:]

    return leapfrog(theta, r_hmc, epsilon, d_b, log_dens)

end

function new_dual_av_update_parameters(iter::Int64, alpha::Float64, type_hmc::T_HMC) where {T_HMC<:TypeMCMC}

    type_hmc.bar_h[1] = (1.0 - 1.0 / (iter + type_hmc.t[1])) * type_hmc.bar_h[1] + 1.0 / (iter + type_hmc.t[1]) * (type_hmc.delta[1] - alpha)

    type_hmc.epsilon[1] = exp(type_hmc.mu[1] - sqrt(iter) / type_hmc.gamma[1] * type_hmc.bar_h[1])

    type_hmc.bar_epsilon[1] = exp(iter^(-type_hmc.k[1]) * log(type_hmc.epsilon[1]) + (1.0 - iter^(-type_hmc.k[1])) * log(type_hmc.bar_epsilon[1]))

end

function new_restart_hmc_par(type_hmc::HMC_DualAv_NoUTurn)
    type_hmc.mu[1] = log(10.0 * type_hmc.epsilon[1])
    type_hmc.bar_epsilon[:] = type_hmc.bar_epsilon_init[:]
    type_hmc.bar_h[:] = type_hmc.bar_h_init[:]
    type_hmc.gamma[:] = type_hmc.gamma_init[:]
    type_hmc.t[:] = type_hmc.t_init[:]
    type_hmc.k[:] = type_hmc.k_init[:]
end


function new_hmc_sample(log_dens::Function, d_b::Vector{Float64}, theta::Vector{Float64}, r_hmc::Vector{Float64}, type_hmc::HMC_DualAv_NoUTurn, iter::Int64, add_const::Float64, hmc_obj::HMCobjects_V3)
    
    new_hmc_dual_av_nouturn(log_dens, d_b, theta, r_hmc, type_hmc, iter, add_const, hmc_obj)
end

function new_hmc_dual_av_nouturn(log_dens::Function, d_b::Vector{Float64}, theta::Vector{Float64}, r_hmc::Vector{Float64}, type_hmc::HMC_DualAv_NoUTurn, iter::Int64, add_const::Float64, hmc_obj::HMCobjects_V3)

    if type_hmc.restart[1]
        new_restart_hmc_par(type_hmc)
    end
    if type_hmc.general_iter[1] == 1

        new_find_reasonable_epsilon(theta, log_dens, d_b, type_hmc.epsilon, add_const)
        type_hmc.mu[1] = log(10.0 * type_hmc.epsilon[1])
    end

    alpha::Float64 = 0.0
    theta_init = hmc_obj.theta_zero
    r_init = hmc_obj.r_zero

    theta_init .= theta[:]
    r_init .= r_hmc[:]

    for m = 1:type_hmc.M[1]

        type_hmc.general_iter[1] += 1
        alpha = new_hmc_dual_av_nouturn_iter(m, log_dens, d_b, theta_init, r_init, type_hmc, add_const, hmc_obj)

        if iter < type_hmc.iter_adapt[1]
            new_dual_av_update_parameters(type_hmc.general_iter[1], alpha, type_hmc)
        end
    end


end


function new_hmc_dual_av_nouturn_iter(m::Int64, log_dens::Function, d_b::Vector{Float64}, theta::Vector{Float64}, r_hmc::Vector{Float64}, type_hmc::HMC_DualAv_NoUTurn, add_const::Float64, hmc_obj::HMCobjects_V3)
    
    # Initialize momentum
    r_hmc[:] = rand(Normal(0.0, 1.0), size(theta, 1))
    
    # Compute initial Hamiltonian and slice variable
    H_init = log_dens(theta) + new_log_dens_h(r_hmc, add_const)
    log_u = H_init - rand(Exponential(1.0))
    
    # Get pre-allocated arrays
    r_minus = hmc_obj.r_minus
    r_plus = hmc_obj.r_plus
    theta_minus = hmc_obj.theta_minus
    theta_plus = hmc_obj.theta_plus
    theta_save = hmc_obj.theta_save
    r_save = hmc_obj.r_save
    diff_theta = hmc_obj.diff
    
    # Initialize trajectory endpoints
    theta_minus .= theta[:]
    theta_plus .= theta[:]
    r_minus .= r_hmc[:]
    r_plus .= r_hmc[:]
    
    # Counters
    j::Int64 = 0
    n::Int64 = 1  # number of valid states in current trajectory
    s::Int64 = 1  # stopping criterion
    sum_alpha::Float64 = 1.0  # start with initial state
    n_alpha::Int64 = 1
    epsilon = type_hmc.epsilon[1]
    
    while s == 1

        # Choose direction randomly (Algorithm 2: choose v ∈ {-1, 1})
        direction = rand(Bool) ? 1.0 : -1.0
        j_tree = j
        if direction == 1.0
        
        end
    end

    while s == 1 && j < type_hmc.depth_max
        

        # Iterative "doubling": match BuildTree(depth=j) which produces 2^j base cases (Alg. 3).
        # Each base case corresponds to one leapfrog step, so take exactly 2^j steps here.
        n_prime::Int64 = 0
        s_prime::Int64 = 1
        num_steps = 2^j

        for step in 1:num_steps
            if direction == 1.0
                # Move forward from theta_plus
                new_leapfrog_nocheck(theta_plus, r_plus, epsilon, d_b, log_dens)
                
                # Check for numerical errors
                H_new = log_dens(theta_plus) + new_log_dens_h(r_plus, add_const)
                if isnan(H_new) || isinf(H_new)
                    s_prime = 0
                    break
                end
                
                # Count if state is in slice
                if log_u <= H_new
                    n_prime += 1
                end
                
                # Update acceptance probability statistics
                n_alpha += 1
                sum_alpha += min(1.0, exp(H_new - H_init))
            else
                # Move backward from theta_minus
                new_leapfrog_nocheck(theta_minus, r_minus, -epsilon, d_b, log_dens)
                
                # Check for numerical errors
                H_new = log_dens(theta_minus) + new_log_dens_h(r_minus, add_const)
                if isnan(H_new) || isinf(H_new)
                    s_prime = 0
                    break
                end
                
                # Count if state is in slice
                if log_u <= H_new
                    n_prime += 1
                end
                
                # Update acceptance probability statistics
                n_alpha += 1
                sum_alpha += min(1.0, exp(H_new - H_init))
            end
        end
        
    

        # FIXED: Accept endpoint with correct probability AFTER all steps
        if s_prime == 1 && n_prime > 0
            accept_prob = min(1.0, Float64(n_prime) / Float64(n))
            if rand() < accept_prob
                if direction == 1.0
                    theta[:] .= theta_plus[:]
                    r_hmc[:] .= r_plus[:]
                else
                    theta[:] .= theta_minus[:]
                    r_hmc[:] .= r_minus[:]
                end
            end
        end
        
        # Update total count
        n += n_prime
        
        # Check U-turn condition
        diff_theta .= theta_plus .- theta_minus
        uturn_minus = dot(diff_theta, r_minus) < 0.0
        uturn_plus = dot(diff_theta, r_plus) < 0.0
        
        if uturn_minus || uturn_plus || s_prime == 0
            s = 0
        end
        
        j += 1
    end
    
    # Return average acceptance probability
    return n_alpha > 0 ? sum_alpha / n_alpha : 0.0
end

for i in reverse(1:10)

println(i)
end
#function iteration_HMC_new!(
#    theta::Vector{Float64},
#    epsilon::Float64,
#    depth_max::Int64,
#    H::Function,
#    grad_H_dq::Function,
#    grad_H_dp::Function,
#    params::NamedTuple,
#    ratio_accepted::Vector{Float64},

#    theta_init::Vector{Float64},
#    r_init::Vector{Float64},
#    theta_minus::Vector{Float64},
#    theta_plus::Vector{Float64},
#    r_minus::Vector{Float64},
#    r_plus::Vector{Float64},
#    r::Vector{Float64}
#)

#    num_dim = length(theta)

#    # Inizializziamo traiettoria
#    #theta_init = copy(theta)
#    #r_init = randn(num_dim)

#    #theta_minus = copy(theta_init)
#    #theta_plus  = copy(theta_init)

#    #r_minus = copy(r_init)
#    #r_plus  = copy(r_init)
#    #r       = copy(r_init)

#    H_init = H(theta_init, r_init, params)
#    logu = - H_init - rand(Exponential(1))
#    num_explored = 1.0
#    alpha = 0.0
#    j = -1
#    s = 1

#    while s == 1 && j <= depth_max
#        j += 1
#        num_steps = 2^j
#        direction = rand(Bool) ? 1.0 : -1.0

#        for _ in 1:num_steps
#            if direction == 1
#                leapfrog(theta_plus, r_plus, epsilon * direction, d_b, log_dens)
#                theta_prime, r_prime = theta_plus, r_plus
#            else
#                leapfrog(theta_minus, r_minus, epsilon * direction, d_b, log_dens)
#                #leapfrog!(theta_minus, r_minus, epsilon * direction, grad_H_dq, grad_H_dp, params)
#                theta_prime, r_prime = theta_minus, r_minus
#            end

#            # Condizioni di validità
#            diff = theta_plus - theta_minus
#            H_prime = H(theta_prime, r_prime, params)
#            s1 = logu <= 1000 - H_prime ? 1 : 0   # controllo numerico (?)
#            s2 = logu <= -H_prime ? 1 : 0
#            s3 = dot(diff, r_minus) >= 0 ? 1 : 0
#            s4 = dot(diff, r_plus)  >= 0 ? 1 : 0

#            s_prime = s1 & s2 & s3 & s4
#            s       = s3 & s4   # stop condition

#            # se s2 NON rispettata → scarto il campione
#            if s_prime == 1
#                num_explored += 1.0
#                alpha += min(1.0, exp(- H_prime + H_init))
#                if rand() < 1 / num_explored
#                    theta .= theta_prime
#                    r     .= r_prime
#                end
#            end

#            if s == 0
#                break
#            end
#        end
#    end

#    ratio_accepted[1] = alpha / num_explored
#end

#function iteration_NUTS!(
#    theta::Vector{Float64},
#    epsilon::Float64,
#    depth_max::Int64,
#    H::Function,
#    grad_H_dq::Function,
#    grad_H_dp::Function,
#    params::NamedTuple,
#    ratio_accepted::Vector{Float64}, theta_init::Vector{Float64},
#    r_init::Vector{Float64},
#    theta_minus::Vector{Float64},
#    theta_plus::Vector{Float64},
#    r_minus::Vector{Float64},
#    r_plus::Vector{Float64},
#    r::Vector{Float64}
#)

#    num_dim = length(theta)

#    # Inizializziamo traiettoria
#    #theta_init = copy(theta)
#    #r_init = randn(num_dim)

#    #theta_minus = copy(theta_init)
#    #theta_plus  = copy(theta_init)

#    #r_minus = copy(r_init)
#    #r_plus  = copy(r_init)
#    #r       = copy(r_init)

#    H_init = H(theta_init, r_init, params)
#    logu = -H_init - rand(Exponential(1))
#    num_explored = 1.0
#    alpha = 0.0
#    j = -1
#    s = 1

#    while s == 1 && j <= depth_max
#        j += 1
#        num_steps = 2^j
#        direction = rand(Bool) ? 1.0 : -1.0

#        for _ in 1:num_steps
#            if direction == 1
#                leapfrog!(theta_plus, r_plus, epsilon * direction, grad_H_dq, grad_H_dp, params)
#                theta_prime, r_prime = theta_plus, r_plus
#            else
#                leapfrog!(theta_minus, r_minus, epsilon * direction, grad_H_dq, grad_H_dp, params)
#                theta_prime, r_prime = theta_minus, r_minus
#            end

#            # Condizioni di validità
#            diff = theta_plus - theta_minus
#            H_prime = H(theta_prime, r_prime, params)
#            s1 = logu <= 1000 - H_prime ? 1 : 0   # controllo numerico (?)
#            s2 = logu <= -H_prime ? 1 : 0
#            s3 = dot(diff, r_minus) >= 0 ? 1 : 0
#            s4 = dot(diff, r_plus) >= 0 ? 1 : 0

#            s_prime = s1 & s2 & s3 & s4
#            s = s3 & s4   # stop condition

#            # se s2 NON rispettata → scarto il campione
#            if s_prime == 1
#                num_explored += 1.0
#                alpha += min(1.0, exp(-H_prime + H_init))
#                if rand() < 1 / num_explored
#                    theta .= theta_prime
#                    r .= r_prime
#                end
#            end

#            if s == 0
#                break
#            end
#        end
#    end

#    ratio_accepted[1] = alpha / num_explored
#end
