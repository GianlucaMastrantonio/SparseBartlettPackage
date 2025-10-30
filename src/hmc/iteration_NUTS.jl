
function iteration_NUTS!(
    theta::Vector{Float64},
    epsilon::Float64,
    depth_max::Int64,
    H::Function,
    grad_H_dq::Function,
    grad_H_dp::Function,
    params::NamedTuple,
    ratio_accepted::Vector{Float64}
)

    num_dim = length(theta)

    # Inizializziamo traiettoria
    theta_init = copy(theta)
    r_init = randn(num_dim)

    theta_minus = copy(theta_init)
    theta_plus  = copy(theta_init)

    r_minus = copy(r_init)
    r_plus  = copy(r_init)
    r       = copy(r_init)

    H_init = H(theta_init, r_init, params)
    logu = - H_init - rand(Exponential(1))
    num_explored = 1.0
    alpha = 0.0
    j = -1
    s = 1

    while s == 1 && j <= depth_max
        j += 1
        num_steps = 2^j
        direction = rand(Bool) ? 1.0 : -1.0

        for _ in 1:num_steps
            if direction == 1
                leapfrog!(theta_plus, r_plus, epsilon * direction, grad_H_dq, grad_H_dp, params)
                theta_prime, r_prime = theta_plus, r_plus
            else
                leapfrog!(theta_minus, r_minus, epsilon * direction, grad_H_dq, grad_H_dp, params)
                theta_prime, r_prime = theta_minus, r_minus
            end

            # Condizioni di validità
            diff = theta_plus - theta_minus
            H_prime = H(theta_prime, r_prime, params)
            s1 = logu <= 1000 - H_prime ? 1 : 0   # controllo numerico (?)
            s2 = logu <= -H_prime ? 1 : 0
            s3 = dot(diff, r_minus) >= 0 ? 1 : 0
            s4 = dot(diff, r_plus)  >= 0 ? 1 : 0

            s_prime = s1 & s2 & s3 & s4
            s       = s3 & s4   # stop condition

            # se s2 NON rispettata → scarto il campione
            if s_prime == 1
                num_explored += 1.0
                alpha += min(1.0, exp(- H_prime + H_init))
                if rand() < 1 / num_explored
                    theta .= theta_prime
                    r     .= r_prime
                end
            end

            if s == 0
                break
            end
        end
    end

    ratio_accepted[1] = alpha / num_explored
end
