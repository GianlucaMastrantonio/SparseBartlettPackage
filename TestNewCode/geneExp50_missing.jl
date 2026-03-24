
name = "GeneExp50_miss"

#### PARAMETERS ####
# Flag to indicate if this is the first run (for environment setup)
is_first_time::Bool = false # Set to true only the first time you run any simulations

#### SIMULATION  ENVIRONMENT SETUP ####
using Pkg
Pkg.activate("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode") # Activate the Julia environment for the applications folder
if is_first_time
    Pkg.rm("SparseBartlettPackage") # Remove the package if it exists (clean install)
    Pkg.develop(url="/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage") # Develop the SparseBartlettPackage from local or remote
    Pkg.instantiate() # Install all dependencies
end

# Load required Julia packages
using MKL 
using Revise
using RCall 
using Distributions
using PDMats
using LinearAlgebra
using ToggleableAsserts
using Enzyme
using Random
using SparseBartlettPackage

Random.seed!(1); # Set random seed for reproducibility

#### DATA LOADING AND PREPARATION ####
# Use R to load and preprocess the dataset
R"""
gene_data = read.csv("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/data/geneExpression.csv", header = F)
summary(gene_data)
gene_var = apply(gene_data,2,var)
gene_data = t(as.matrix(gene_data))
for(i in 1:nrow(gene_data))
{
    gene_data[i,] = (gene_data[i,]-mean(gene_data[i,]))/sd(gene_data[i,])
}
"""
@rget gene_data; # Retrieve the processed dataset from R to Julia
k_tot::Int64 = size(gene_data,1)
n::Int64 = size(gene_data, 2)
dataset = gene_data[1:50,:]
k::Int64 = size(dataset,1)

dataset_true = deepcopy(dataset)
n_miss = Int64(trunc(ceil(n * k * 0.1)))
true_y = zeros(Float64, n_miss)
index_tot = zeros(Int64, k * n, 2)
h = 0
for ii = 1:n
    for ik = 1:k
        global h = h + 1
        index_tot[h, 1] = ik
        index_tot[h, 2] = ii
    end
end


www = sample(1:(n*k), n_miss, replace=false)
sort!(www)
index_miss_mat = index_tot[www, :]
for imiss = 1:n_miss
    true_y[imiss] = dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]]
    dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]] = rand(Normal(0.0, 1.0))
end

#### MCMC SETUP ####
z_mat_init = ones(Float64, k, k)
molt = 1
nm = Int64((k * k - k) / 2)

# Set MCMC parameters: number of iterations, burn-in, and thinning
mcmc_par = (iter=10000 * molt, burnin=8000 * molt, thin=1 * molt)
covariate = ones(Float64, k, 1, n) # Covariate matrix (all ones, no covariate effect)

#### RUN MCMC SAMPLER ####
Random.seed!(1); out = mcmc(
    ydata=dataset, # Observed data
    iterations=mcmc_par, # MCMC parameters
    covariates=covariate,# Covariate matrix 
    model_sparse=GenSparse(Int64.(nm:(-1):1) ./ sum(Int64.(nm:(-1):1))), # Prior for sparsity pattern
    model_psi=GeneralS(), # Prior for psi parameter
    type_mcmc_lambda=HMC_DualAv_NoUTurn(; epsilon=10.000, L=10, M=10, restart=false, delta=0.5,iter_adapt=10000000), # HMC sampler settings
    #type_mcmc_lambda=HMC_DualAv(; epsilon=10.000, L=10, M=10, restart=false, delta=0.5, iter_adapt=10000000, lambda =1.0), # HMC sampler settings
    #type_mcmc_lambda=BaseHMC(; epsilon=10.000, L=10, M=10, restart=false),
    prior_lambda=Wishart(1.0 * (k + 3 - 1), 1.0 .* Matrix(I(k))), # Wishart prior for precision matrix
    prior_mu=Normal(0.0, 1000.0^0.5),  # Prior for mean
    c_init=ones(Float64, k), # Initial value for c parameter
    m_init=zeros(Float64, nm), # Initial value for m parameter
    beta_init=zeros(Float64, size(covariate, 2)), # Initial value for regression coefficients
    prob_zeta=ones(Float64, k, k) * 0.5, # Initial probability for zeta (sparsity)
    z_init=Float64.(z_mat_init), # Initial value for z (sparsity pattern)
    iter_start_hmc=2, # When to start HMC updates
    iter_start_zeta=5, # When to start zeta updates
    missing_index=index_miss_mat # Missing
)
#### EXTRACT AND SAVE RESULTS ####
# Extract MCMC outputs
m_out = out.m_out;
c_out = out.c_out;
qmat_sparse_out = out.q_mat_sparse_out;
lambda_out = out.lambda_out;
zmat_out = out.zmat_out;
beta_out = out.beta_out;
y = dataset;
missing_out = out.missing_out;

#### EXTRACT AND SAVE RESULTS ####
# Extract MCMC outputs
@rput y;
@rput missing_out;
@rput index_miss_mat;
@rput beta_out;
@rput lambda_out;
@rput name;
@rput k;
@rput zmat_out;
@rput true_y;

# Save the R workspace with all results for later analysis
R"""
    save.image("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/out/test.Rdata")
"""

#@time for i = 1:k
#    cholesky(SS)
#end

##chole
#k = 20
#nm = Int64((k * k - k) / 2)
#bstar_sim = rand(Normal(0.0, 1.0), Int64(k + nm))
#b_mat_sim = zeros(Float64, k, k)
#q_mat_sim = zeros(Float64, k, k)
#q_mat_sparse_sim = zeros(Float64, k, k)
#lambda_sim = zeros(Float64, k, k)

#bstar_sim_c = rand(Normal(0.0, 1.0), Int64(k + nm))
#b_mat_sim_c = zeros(Float64, k, k)
#q_mat_sim_c = zeros(Float64, k, k)
#q_mat_sparse_sim_c = zeros(Float64, k, k)
#lambda_sim_c = zeros(Float64, k, k)

#bstar_sim_app = rand(Normal(0.0, 1.0), Int64(k + nm))
#b_mat_sim_app = zeros(Float64, k, k)
#q_mat_sim_app = zeros(Float64, k, k)
#q_mat_sparse_sim_app = zeros(Float64, k, k)
#lambda_sim_app = zeros(Float64, k, k)


#rorder_vec = sample(1:k,k, replace=false) 
#rorder_mat = zeros(Float64, k, k)
#for i = 1:k
#    rorder_mat[i,rorder_vec[i]] =  1
#end


#psi_sim = Matrix(I(k)) .* 0.5



#z_sim = ones(Float64, k, k)
#z_sim[3, 2] = 0

#z_sim[14, 10] = 0
#z_sim[14, 12] = 0


#z_sim_c = rorder_mat * z_sim * rorder_mat'
#psi_sim_c = rorder_mat * psi_sim * rorder_mat'
#psi_inv_c = inv(psi_sim_c)

#SparseBartlettPackage.from_bstar_to_b_mat(bstar_sim, b_mat_sim)
#SparseBartlettPackage.from_bmat_to_all_q(b_mat_sim, q_mat_sim, q_mat_sparse_sim, z_sim, psi_sim)
#lambda_sim[:, :] = q_mat_sparse_sim * transpose(q_mat_sparse_sim)

#lambda_sim[3,2]
#lambda_sim[14, 10]
#lambda_sim[14, 12]

#lambda_sim_c = (rorder_mat * lambda_sim * rorder_mat')


#lambda_sim_c[findall(x -> x == 3, rorder_vec)[1], findall(x -> x == 2, rorder_vec)[1]]
#lambda_sim_c[findall(x -> x == 14, rorder_vec)[1], findall(x -> x == 10, rorder_vec)[1]]
#lambda_sim_c[findall(x -> x == 14, rorder_vec)[1], findall(x -> x == 12, rorder_vec)[1]]

##lambda_sim_c = Symmetric(lambda_sim )


#findall(x -> x == 2, rorder_vec)[1]
#findall(x -> x == 3, rorder_vec)[1]


#sum(q_mat_sparse_sim_c[15, 1:9] .* q_mat_sparse_sim_c[10, 1:9]) ./ q_mat_sparse_sim_c[10, 10]
#q_mat_sparse_sim_c[15,10]

##lambda_sim_c - lambda_sim
#q_mat_sparse_sim_c[:, :] .= cholesky(Symmetric(lambda_sim_c)).L
#for j = 1:k
#    for i = j:k
#        if z_sim_c[i, j] == 1.0

#            q_mat_sim_c[i,j] = q_mat_sparse_sim_c[i,j]

#        else

#            q_mat_sim_c[i, j] = q_mat_sim[i, j]
#            #q_mat_sim_c[i, j] = q_mat_sparse_sim_c[i, j]

#        end
#    end
#end

#for j = 1:k
#    for i = j:k
#        b_mat_sim_c[i, j] = sum(psi_inv_c[i, j:k] .* q_mat_sim_c[j:k, j])
#        #b_mat_sim_c[i, j] = sum(psi_inv[i, :] .* q_mat_sim_c[j, :])
#    end
#end
#h = 0
#for j = 1:k
#    for i = j:k
#        h += 1
#        if j == i
#            bstar_sim_c[h] = log(b_mat_sim_c[i, j])
#        else
#            bstar_sim_c[h] = b_mat_sim_c[i, j]
#        end
#    end
#end



#SparseBartlettPackage.from_bstar_to_b_mat(bstar_sim_c, b_mat_sim_app)
#SparseBartlettPackage.from_bmat_to_all_q(b_mat_sim_app, q_mat_sim_app, q_mat_sparse_sim_app, z_sim_c, psi_sim_c)
#minimum((lambda_sim_c-q_mat_sparse_sim_app*transpose(q_mat_sparse_sim_app))[:])


#b_mat_sim_app - b_mat_sim_c
#q_mat_sim_app - q_mat_sim_c
#q_mat_sparse_sim_app - q_mat_sparse_sim_c

#lambda_sim_c
