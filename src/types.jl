struct order_lambda_V3

    
    zeta_prop::Matrix{Float64}
    bstar_prop::Vector{Float64}
    q_mat_prop::Matrix{Float64}
    q_mat_sparse_prop::Matrix{Float64}
    
    lambda_propc::Matrix{Float64}
    b_mat_prop::Matrix{Float64}
    
    order_vec::Vector{Int64}
    order_mat::Matrix{Float64}
    inv_order_mat::Matrix{Float64}
    inv_order_vec::Vector{Int64}

    n_accepted::Vector{Int64}
    n_total::Vector{Int64}

    psi_inv_prop::Matrix{Float64}
    psi_prop::Matrix{Float64}

    n_zero::Vector{Int64}
    pos_zero_mcmc::Matrix{Int64}
    pos_zero_prop::Matrix{Int64}
    p_prop::Vector{Float64}
    p_mcmc::Vector{Float64}
    q_mat_sparse_prop_with_p::Matrix{Float64}
    psi_psit_mcmc::Matrix{Float64}
    psi_psit_prop::Matrix{Float64}
    k_half::Int64
    index_switch::Matrix{Int64}
    ycentered_prop::Matrix{Float64}

     

    function order_lambda_V3(;
        zeta_prop::Matrix{Float64},
        bstar_prop::Vector{Float64},
        q_mat_prop::Matrix{Float64},
        q_mat_sparse_prop::Matrix{Float64},
        lambda_propc::Matrix{Float64},
        b_mat_prop::Matrix{Float64},
        order_vec::Vector{Int64},
        order_mat::Matrix{Float64},
        inv_order_mat::Matrix{Float64},
        inv_order_vec::Vector{Int64},
        n_accepted::Vector{Int64},
        n_total::Vector{Int64},
        psi_inv_prop::Matrix{Float64},
        psi_prop::Matrix{Float64},
        n_zero::Vector{Int64},
        pos_zero_mcmc::Matrix{Int64},
        pos_zero_prop::Matrix{Int64},
        p_prop::Vector{Float64},
        p_mcmc::Vector{Float64},
        q_mat_sparse_prop_with_p::Matrix{Float64},
        psi_psit_mcmc::Matrix{Float64},
        psi_psit_prop::Matrix{Float64},
        k_half::Int64,
    index_switch::Matrix{Int64},
        ycentered_prop::Matrix{Float64}
    )

        new(zeta_prop::Matrix{Float64},
            bstar_prop::Vector{Float64},
            q_mat_prop::Matrix{Float64},
            q_mat_sparse_prop::Matrix{Float64},
            lambda_propc::Matrix{Float64},
            b_mat_prop::Matrix{Float64},
            order_vec::Vector{Int64},
            order_mat::Matrix{Float64},
            inv_order_mat::Matrix{Float64},
            inv_order_vec::Vector{Int64},
            n_accepted::Vector{Int64},
            n_total::Vector{Int64},
            psi_inv_prop::Matrix{Float64},
            psi_prop::Matrix{Float64},
            n_zero::Vector{Int64},
            pos_zero_mcmc::Matrix{Int64},
            pos_zero_prop::Matrix{Int64},
            p_prop::Vector{Float64},
            p_mcmc::Vector{Float64},
            q_mat_sparse_prop_with_p::Matrix{Float64},
            psi_psit_mcmc::Matrix{Float64},
            psi_psit_prop::Matrix{Float64},
            k_half::Int64,
        index_switch::Matrix{Int64},
            ycentered_prop::Matrix{Float64}
    )

    end

end
function create_order_lambda_V3(; nm::Int64, k::Int64,n::Int64)

    k_half = Int64(floor((k - 1) / 2))
    order_lambda_V3(;
        zeta_prop = fill(0.0, k, k),
        bstar_prop=fill(0.0, nm + k),
        q_mat_prop=fill(0.0, k,k),
        q_mat_sparse_prop=fill(0.0, k, k),
        lambda_propc=fill(0.0, k, k),
        b_mat_prop=fill(0.0, k, k),
        order_vec=fill(0,k),
        order_mat=Matrix(I(k)) .* 1.0,
        inv_order_mat=Matrix(I(k)) .* 1.0,
        inv_order_vec=fill(0, k),
        n_accepted = [0],
        n_total=[0],
        psi_inv_prop=fill(0.0, k, k),
        psi_prop=fill(0.0, k, k),
        n_zero = [0],
        pos_zero_mcmc = fill(0, nm + k, 2),
        pos_zero_prop=fill(0, nm + k, 2),
        p_prop=fill(0.0, nm + k),
        p_mcmc =fill(0.0, nm + k),
        q_mat_sparse_prop_with_p=fill(0.0, k, k),
        psi_psit_mcmc=fill(0.0, k, k), 
        psi_psit_prop=fill(0.0, k, k),
        k_half = k_half,
        index_switch=fill(0, k_half, 2),
        ycentered_prop=fill(0.0, k, n)
    )

end

struct adapt_sd_lambda_V1

    sd_vec::Vector{Float64}
    alpha_bar::Vector{Float64}

    target_rate::Vector{Float64}
    vec_a::Vector{Float64}
    vec_b::Vector{Float64}
    n_batch::Vector{Int64}

    bstar_prop::Vector{Float64}
    q_mat_prop::Matrix{Float64}
    q_mat_sparse_prop::Matrix{Float64}
    lambda_propc::Matrix{Float64}
    b_mat_prop::Matrix{Float64}
    burnin::Int64

    function adapt_sd_lambda_V1(;
        sd_vec::Vector{Float64},
        alpha_bar::Vector{Float64},
        target_rate::Vector{Float64},
        vec_a::Vector{Float64},
        vec_b::Vector{Float64},
        n_batch::Vector{Int64},
        bstar_prop::Vector{Float64},
        q_mat_prop::Matrix{Float64},
        q_mat_sparse_prop::Matrix{Float64},
        lambda_propc::Matrix{Float64},
        b_mat_prop::Matrix{Float64},
        burnin::Int64
    )

        new(sd_vec::Vector{Float64},
            alpha_bar::Vector{Float64},
            target_rate::Vector{Float64},
            vec_a::Vector{Float64},
            vec_b::Vector{Float64},
            n_batch::Vector{Int64},
            bstar_prop::Vector{Float64},
            q_mat_prop::Matrix{Float64},
            q_mat_sparse_prop::Matrix{Float64},
            lambda_propc::Matrix{Float64},
            b_mat_prop::Matrix{Float64},
            burnin::Int64
        )

    end

end

function create_adapt_sd_lambda_V1(; nm::Int64, k::Int64, sd::Float64, acc_rate::Float64, A::Float64, B::Float64, iter_batch::Int64, burnin::Int64)

    adapt_sd_lambda_V1(;
        sd_vec=fill(sd, nm+k),
        alpha_bar=fill(0.0, nm + k),
        target_rate=fill(acc_rate, nm + k),
        vec_a=fill(A, nm + k),
        vec_b=fill(B, nm + k),
        n_batch=fill(iter_batch, nm + k),
        bstar_prop=fill(0.0, nm + k),
        q_mat_prop=fill(0.0, k,k),
        q_mat_sparse_prop=fill(0.0, k, k),
        lambda_propc=fill(0.0, k, k),
        b_mat_prop=fill(0.0, k, k),
        burnin
    )

end


abstract type TypeMCMC end
struct StandardMCMC <: TypeMCMC

    function StandardMCMC()

        new()

    end

end

struct HMC <: TypeMCMC

    epsilon::Vector{Float64}
    iter::Vector{Int64}
    function HMC(epsilon::Float64, iter::Int64)

        new([epsilon], [iter])

    end

end

struct HMCRJ <: TypeMCMC

    epsilon::Vector{Float64}
    iter::Vector{Int64}
    delta::Vector{Float64}
    m_max::Vector{Int64}
    function HMCRJ(epsilon::Float64, iter::Int64, delta::Float64, m_max::Int64)

        new([epsilon], [iter], [delta], [m_max])

    end

end


struct BaseHMC <: TypeMCMC

    epsilon::Vector{Float64}
    L::Vector{Int64}
    M::Vector{Int64}
    restart::Vector{Bool}
    epsilon_init::Vector{Float64}
    L_init::Vector{Int64}
    M_init::Vector{Int64}
    general_iter::Vector{Int64}
    function BaseHMC(; epsilon::Float64, L::Int64, M::Int64, restart::Bool)

        new([epsilon], [L], [M], [restart], [epsilon], [L], [M], [1])

    end

end

struct HMC_DualAv <: TypeMCMC

    epsilon::Vector{Float64}
    L::Vector{Int64}
    M::Vector{Int64}
    delta::Vector{Float64}
    lambda::Vector{Float64}
    iter_adapt::Vector{Int64}

    restart::Vector{Bool}
    epsilon_init::Vector{Float64}
    L_init::Vector{Int64}
    M_init::Vector{Int64}
    general_iter::Vector{Int64}
    delta_init::Vector{Float64}
    lambda_init::Vector{Float64}

    mu::Vector{Float64}
    bar_epsilon::Vector{Float64}
    bar_h::Vector{Float64}
    gamma::Vector{Float64}
    t::Vector{Float64}
    k::Vector{Float64}

    mu_init::Vector{Float64}
    bar_epsilon_init::Vector{Float64}
    bar_h_init::Vector{Float64}
    gamma_init::Vector{Float64}
    t_init::Vector{Float64}
    k_init::Vector{Float64}

    function HMC_DualAv(; epsilon::Float64, L::Int64, M::Int64, restart::Bool, delta::Float64, lambda::Float64, iter_adapt::Int64)

        new([epsilon], [L], [M], [delta], [lambda], [iter_adapt], [restart], [epsilon], [L], [M], [1], [delta], [lambda], [log(10.0)], [1.0], [0.0], [0.5], [10.0], [0.75], [log(10.0)], [1.0], [0.0], [0.5], [10.0], [0.75])

    end

end

struct HMC_DualAv_NoUTurn <: TypeMCMC

    epsilon::Vector{Float64}
    L::Vector{Int64}
    M::Vector{Int64}
    delta::Vector{Float64}
    #lambda::Vector{Float64}
    iter_adapt::Vector{Int64}

    restart::Vector{Bool}
    epsilon_init::Vector{Float64}
    L_init::Vector{Int64}
    M_init::Vector{Int64}
    general_iter::Vector{Int64}
    delta_init::Vector{Float64}
    #lambda_init::Vector{Float64}

    mu::Vector{Float64}
    bar_epsilon::Vector{Float64}
    bar_h::Vector{Float64}
    gamma::Vector{Float64}
    t::Vector{Float64}
    k::Vector{Float64}

    mu_init::Vector{Float64}
    bar_epsilon_init::Vector{Float64}
    bar_h_init::Vector{Float64}
    gamma_init::Vector{Float64}
    t_init::Vector{Float64}
    k_init::Vector{Float64}
    deltamax::Vector{Float64}

    function HMC_DualAv_NoUTurn(; epsilon::Float64, L::Int64, M::Int64, restart::Bool, delta::Float64, iter_adapt::Int64)

        new([epsilon], [L], [M], [delta], [iter_adapt], [restart], [epsilon], [L], [M], [1], [delta], [log(10.0)], [1.0], [0.0], [0.5], [10.0], [0.75], [log(10.0)], [1.0], [0.0], [0.5], [10.0], [0.75], [1000.0])

    end

end
#########


abstract type TypeSparse end
struct NoSparse <: TypeSparse

    function NoSparse()

        new()

    end

end
struct GenSparse <: TypeSparse

    prob_vec_zeros::Vector{Float64}
    function GenSparse(prob_vec::Vector{Float64})

        new(deepcopy(prob_vec))

    end

end

struct FixSparse <: TypeSparse

    zero_indices::Matrix{Int64}

    function FixSparse(zero_indices::Matrix{Int64})

        new(zero_indices)

    end

end

struct SpikeSlabSparse <: TypeSparse

    tau_small::Float64
    tau_large::Float64
    z_prior::Binomial{Float64}

    function SpikeSlabSparse(tau_small::Float64, tau_large::Float64, z_prior::Binomial{Float64})

        new(tau_small, tau_large, z_prior)

    end

end
struct SparseRJ <: TypeSparse

    prob::Vector{Float64}

    function SparseRJ(prob::Vector{Float64})

        new(prob)

    end

end

abstract type TypeS end
struct DiagonalS <: TypeS

    function DiagonalS()

        new()

    end

end


struct GeneralS <: TypeS

    function GeneralS()

        new()

    end

end


#############



