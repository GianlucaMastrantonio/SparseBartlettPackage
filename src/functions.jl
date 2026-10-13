function isdiag(A::Matrix{Float64})
    for j in axes(A, 2), i in axes(A, 1)
        if i != j && !isapprox(A[i, j], 0.0) 
            return false
        end
    end
    return true
end
function from_bstar_to_bmat(bstar_mcmc::Vector{Float64}, bmat::Matrix{Float64}, k::Int64, n::Int64)

    h::Int64 = 1
    ## j = 1
    bmat[1, 1] = exp(bstar_mcmc[h])
    h += 1
    for i = 2:k
        bmat[i, 1] = bstar_mcmc[h]
        h += 1
    end
    for j = 2:(k-1)
        bmat[j, j] = exp(bstar_mcmc[h])
        h += 1
        for i = (j+1):k
            bmat[i, j] = bstar_mcmc[h]
            h += 1
        end
    end
    bmat[k, k] = exp(bstar_mcmc[h])

    return nothing


end

function from_bmat_to_qmat(bmat::Matrix{Float64}, qmat::Matrix{Float64}, k::Int64, n::Int64, psi::Matrix{Float64})

    for j = 1:k
        for i = j:k
            qmat[i, j] = sum(psi[i, j:k] .* bmat[j:k, j]) # mettere solo in funzione di bstar_mcmc
        end
    end

    return nothing


end


function from_qmat_to_qstar_init(qmat::Matrix{Float64}, qstar::Matrix{Float64}, k::Int64, n::Int64, psi::Matrix{Float64}, z_mcmc::Matrix{Float64})



    qstar[1, 1] = z_mcmc[1, 1] * qmat[1, 1]
    for i = 2:k
        from_qmat_to_qstar(i, 1, qmat, qstar, k, n, psi, z_mcmc)
    end
    for j = 2:(k-1)
        qstar[j, j] = z_mcmc[j, j] * qmat[j, j]
        for i = (j+1):k
            from_qmat_to_qstar(i, j, qmat, qstar, k, n, psi, z_mcmc)
        end
    end
    qstar[k, k] = z_mcmc[k, k] * qmat[k, k]


    return nothing


end


function from_qmat_to_qstar_triangularindex(i::Int64, j::Int64, qmat::Matrix{Float64}, qstar::Matrix{Float64}, k::Int64, n::Int64, psi::Matrix{Float64}, z_mcmc::Matrix{Float64})




    from_qmat_to_qstar(i, j, qmat, qstar, k, n, psi, z_mcmc)
    if (j + 1) < i
        for jcol = (j+1):(i-1)
            for irow = i:k
                from_qmat_to_qstar(irow, jcol, qmat, qstar, k, n, psi, z_mcmc)
            end
        end
    end
    if i < k
        for jcol = i:(k-1)
            for irow = (jcol+1):k
                from_qmat_to_qstar(irow, jcol, qmat, qstar, k, n, psi, z_mcmc)
            end
        end
    end

    

    return nothing


end

function from_qmat_to_qstar(i::Int64, j::Int64, qmat::Matrix{Float64}, qstar::Matrix{Float64}, k::Int64, n::Int64, psi::Matrix{Float64}, z_mcmc::Matrix{Float64})



    if j == 1
        qstar[i, 1] = z_mcmc[i, 1] * qmat[i, 1]
    else
        if z_mcmc[i, j] == 1.0
            qstar[i, j] = qmat[i, j]
        else
            qstar[i, j] = 0.0
            s::Float64 = 0.0
            @inbounds for jcol = 1:(j-1)
                s += qstar[i, jcol] * qstar[j, jcol]
            end
            qstar[i, j] = -s/ qstar[j, j]
            #qstar[i,j] = - sum(qstar[i,1:(j-1)] .* qstar[j,1:(j-1)]) / qstar[j,j]
        end
        #qstar[i,j] = z_mcmc[i,j] * qmat[i,j] + (1.0-z_mcmc[i,j]) * (- sum(qstar[i,1:(j-1)] .* qstar[j,1:(j-1)]) / qstar[j,j])
    end
    #qstar[k,k] = z_mcmc[k,k]*qmat[k,k]


    return nothing


end

function permute!(a, n)
    if n == 1
        return [copy(a)]
    else
        result = []
        for i in 1:n
            append!(result, permute!(a, n - 1))
            if n % 2 == 1
                a[1], a[n] = a[n], a[1]  # Swap first and last
            else
                a[i], a[n] = a[n], a[i]  # Swap ith and last
            end
        end
        return result
    end
end

function compute_unique_permutations(k)
    # Generate all permutations using Heap's algorithm
    perms = permute!(collect(1:k), k)

    # Use a Dict to keep track of which "last two numbers" have been seen
    seen = Dict{Tuple{Int,Int},Bool}()
    result = []

    for perm in perms
        last_two = (perm[end-1], perm[end])
        if !haskey(seen, last_two)
            push!(result, perm)
            seen[last_two] = true
        end
    end

    # Convert the result into a matrix
    return transpose(hcat(result...))
end

function from_bmat_to_m_and_c(c_mcmc::Vector{Float64}, m_mcmc::Vector{Float64}, bmat_mcmc::Matrix{Float64})

    k::Int64 = size(bmat_mcmc,1)
    h::Int64 = 1
    
    for j = 1:k
        c_mcmc[j] = bmat_mcmc[j,j]
    end
    for j = 1:(k-1)
        for i = (j+1):k
            m_mcmc[h] = bmat_mcmc[i,j]
            h += 1 
        end

    end


    return nothing

end


function from_c_and_m_to_all_b(c_mcmc::Vector{Float64}, m_mcmc::Vector{Float64}, b_mat_mcmc::Matrix{Float64}, b_mcmc::Vector{Float64}, bstar_mcmc::Vector{Float64}, bstar_mat_mcmc::Matrix{Float64})

    k::Int64 = size(bstar_mat_mcmc,1)
    h::Int64 = 1
    h_m::Int64 = 1

    
    for j = 1:(k-1)
        b_mcmc[h] = c_mcmc[j]
        #bstar_mcmc[h] = log(c_mcmc[j])
        bstar_mcmc[h] = (c_mcmc[j])
        bstar_mat_mcmc[j,j] = bstar_mcmc[h]
        b_mat_mcmc[j,j] = b_mcmc[h]
        h += 1 
        for i = (j+1):k
            b_mcmc[h] = m_mcmc[h_m]
            bstar_mcmc[h] = m_mcmc[h_m]
            bstar_mat_mcmc[i,j] = bstar_mcmc[h]
            b_mat_mcmc[i,j] = b_mcmc[h]
            h += 1 
            h_m += 1
        end

    end
    b_mcmc[h] = c_mcmc[k]
    #bstar_mcmc[h] = log(c_mcmc[k])
    bstar_mcmc[h] = (c_mcmc[k])
    bstar_mat_mcmc[k,k] = bstar_mcmc[h]
    b_mat_mcmc[k,k] = b_mcmc[h]

    return nothing

end


function from_bmat_to_all_q(b_mat_mcmc::Matrix{Float64},  q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, zmat_mcmc::Matrix{Float64}, psi::Matrix{Float64})

    #q_mat_mcmc .= psi*b_mat_mcmc
    mul!(q_mat_mcmc, psi, b_mat_mcmc)
    k::Int64 = size(q_mat_mcmc,1)

    q_mat_sparse_mcmc[1,1] = q_mat_mcmc[1,1]
    for i = 2:k
        q_mat_sparse_mcmc[i,1] = zmat_mcmc[i,1] * q_mat_mcmc[i,1] 
    end
    for j = 2:(k-1)
        q_mat_sparse_mcmc[j,j] = q_mat_mcmc[j,j] 
        for i = (j+1):k
            q_mat_sparse_mcmc[i,j] = zmat_mcmc[i,j] * q_mat_mcmc[i,j] + (1.0-zmat_mcmc[i,j]) * (- sum(q_mat_sparse_mcmc[i,1:(j-1)] .* q_mat_sparse_mcmc[j,1:(j-1)]) / q_mat_sparse_mcmc[j,j])
        end
    end
    q_mat_sparse_mcmc[k,k] = q_mat_mcmc[k,k]
    
    return nothing

end




function new_from_bmat_to_all_q_from_col(col::Int64, b_mat_mcmc::Matrix{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, zmat_mcmc::Matrix{Float64}, psi::Matrix{Float64})

    k::Int64 = size(q_mat_mcmc, 1)
    s::Float64 = 0.0
    # update only column col of q_mat_mcmc
    @inbounds for i in col:k
        q_mat_mcmc[i, col] = 0.0
        for h in col:i
            q_mat_mcmc[i, col] += psi[i, h] * b_mat_mcmc[h, col]
        end
    end
     # recompute q_mat_sparse_mcmc from col onward
    z::Float64 = 0
    qjj::Float64 = 0.0
    qjk::Float64 = 0.0
    h::Int64 = 0
    @inbounds for j = col:(k - 1)
        #q_mat_sparse_mcmc[j, j] = q_mat_mcmc[j, j]
        qjj = q_mat_mcmc[j,j]
        q_mat_sparse_mcmc[j,j] = qjj
        for i = (j + 1):k
            s = 0.0
            #for h in 1:(j - 1)
            #    s += q_mat_sparse_mcmc[i, h] * q_mat_sparse_mcmc[j, h]
            #end
            h = 1
            while h <j
                s += q_mat_sparse_mcmc[i, h] * q_mat_sparse_mcmc[j, h]
                h += 1
            end
            z = zmat_mcmc[i, j]
            q_mat_sparse_mcmc[i, j] = z * q_mat_mcmc[i, j] + (1.0 - z) * ( -s / qjj )
        end
    end

    q_mat_sparse_mcmc[k, k] = q_mat_mcmc[k, k]

    return nothing

end
function old_from_bmat_to_all_q_from_col(col::Int64, b_mat_mcmc::Matrix{Float64}, q_mat_mcmc::Matrix{Float64}, q_mat_sparse_mcmc::Matrix{Float64}, zmat_mcmc::Matrix{Float64}, psi::Matrix{Float64})

    k::Int64 = size(q_mat_mcmc, 1)
    if col == 1
        from_bmat_to_all_q(b_mat_mcmc, q_mat_mcmc, q_mat_sparse_mcmc, zmat_mcmc, psi)
    elseif col == k
        #q_mat_mcmc .= psi * b_mat_mcmc
        mul!(q_mat_mcmc, psi, b_mat_mcmc)
        q_mat_sparse_mcmc[k, k] = q_mat_mcmc[k, k]
    else
        #q_mat_mcmc .= psi * b_mat_mcmc
        mul!(q_mat_mcmc, psi, b_mat_mcmc)
        @inbounds for j = col:(k-1)
            q_mat_sparse_mcmc[j, j] = q_mat_mcmc[j, j]
            for i = (j+1):k
                q_mat_sparse_mcmc[i, j] = zmat_mcmc[i, j] * q_mat_mcmc[i, j] + (1.0 - zmat_mcmc[i, j]) * (-sum(q_mat_sparse_mcmc[i, 1:(j-1)] .* q_mat_sparse_mcmc[j, 1:(j-1)]) / q_mat_sparse_mcmc[j, j])
            end
        end
        q_mat_sparse_mcmc[k, k] = q_mat_mcmc[k, k]

    end

        

    

    return nothing

end

function from_bstar_to_b_mat(bstar_mcmc::Vector{Float64}, b_mat_mcmc::Matrix{Float64} )

    h::Int64 = 1
    k::Int64 = size(b_mat_mcmc,1)

    for j = 1:(k-1)
        b_mat_mcmc[j,j] = exp(bstar_mcmc[h])
        h += 1
        for i = (j+1):k
            b_mat_mcmc[i,j] = bstar_mcmc[h]
            h += 1
        end
    end
    b_mat_mcmc[k,k] = exp(bstar_mcmc[h])
    return nothing
end
#function from_gamma_omega_to_qvec(omega_mcmc::Vector{Float64}, gamma_mcmc::Vector{Float64}, q_mcmc::Vector{Float64}, k::Int64)

#    h::Int64 = 1
#    h_omega::Int64 = 1
#    for j = 1:(k-1)
        
#        q_mcmc[h] = gamma_mcmc[j]
#        h += 1

#        for i = (j+1):k
#            q_mcmc[h] = omega_mcmc[h_omega]
#            h_omega += 1
#            h += 1
#        end

#    end
#    q_mcmc[h] = gamma_mcmc[k]

#end

#function from_qvec_to_qmat(q_mcmc::Vector{Float64}, q_mat_mcmc::Matrix{Float64}, k::Int64)

#    h::Int64 = 1
#    for j = 1:k
#        for i = j:k
            
#            q_mat_mcmc[i,j] = q_mcmc[h]
#            h += 1

#        end
#    end
#end

#function from_qmat_to_bmat(q_mat_mcmc::Matrix{Float64}, b_mat_mcmc::Matrix{Float64}, psi_inv::Matrix{Float64}, k::Int64)

#    b_mat_mcmc[:,:] = psi_inv*q_mat_mcmc

#end

#function from_bmat_to_bvec(b_mat_mcmc::Matrix{Float64}, b_mcmc::Vector{Float64}, k::Int64)

#    h::Int64 = 1
#    for j = 1:k
#        for i = j:k
            
#            b_mcmc[h] = b_mat_mcmc[i,j]
#            h += 1

#        end
#    end
    
#end

#function from_bmat_to_bstarmat(b_mat_mcmc::Matrix{Float64}, bstar_mat_mcmc::Matrix{Float64}, k::Int64)

#    bstar_mat_mcmc[:,:] = b_mat_mcmc[:,:]
#    for j = 1:k
#        bstar_mat_mcmc[j,j] = log(bstar_mat_mcmc[j,j])
#    end
    
#end
