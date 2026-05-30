
name = "GeneExp50_miss"

#iprior = 1
#iq = 1
#idata = 1

#### SIMULATION  ENVIRONMENT SETUP ####
#### PARAMETERS ####
            # Flag to indicate if this is the first run (for environment setup)
            is_first_time::Bool = false # Set to true only the first time you run any simulations


using Pkg
dir = "/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/SparseBartlettPersonal/Real Data" # Directory for the Julia environment
Pkg.activate(dir) # Activate the Julia environment for the applications folder
if is_first_time
    Pkg.rm("SparseBartlettPackage") # Remove the package if it exists (clean install)
    Pkg.develop(
        url = "/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage",
    ) # Develop the SparseBartlettPackage from local or remote
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

R"""

compute_crps <- function(obs, sim) {
  dd <- as.matrix(dist(sim))
  return(mean(abs(obs - sim)) - 1 / (2 * length(sim)^2) * sum(dd))
}
"""
for idata = 1:2
    for iprior = 1:2
        for iq = 1:2
        

            type_q = ["Normal", "Laplace"][iq]
            type_prior = ["nu5", "nu20"][iprior]
            type_data =
                ["geneExp50_miss", "geneExp100_miss", "geneExp50", "geneExp100"][idata]

            name_tot = name * "_" * type_q * "_" * type_prior * "_" * type_data


            

            Random.seed!(1) # Set random seed for reproducibility

            #### DATA LOADING AND PREPARATION ####
            # Use R to load and preprocess the dataset
            @rput dir
            @rput name_tot
            @rput idata
            R"""
            gene_data = read.csv(paste(dir, "/data/geneExpression.csv",sep=""), header = F)
            summary(gene_data)
            gene_var = apply(gene_data,2,var)
            gene_data = t(as.matrix(gene_data))
            for(i in 1:nrow(gene_data))
            {
                gene_data[i,] = (gene_data[i,]-mean(gene_data[i,]))/sd(gene_data[i,])
            }
            """
            @rget gene_data # Retrieve the processed dataset from R to Julia
            k_tot::Int64 = size(gene_data, 1)
            n::Int64 = size(gene_data, 2)


            dataset = gene_data[1:[50, 100, 50, 100][idata], :]
            k::Int64 = size(dataset, 1)

            dataset_true = deepcopy(dataset)
            n_miss = Int64(trunc(ceil(n * k * 0.1)))
            true_y = zeros(Float64, n_miss)
            index_tot = zeros(Int64, k * n, 2)
            h = 0
            for ii = 1:n
                for ik = 1:k
                     h = h + 1
                    index_tot[h, 1] = ik
                    index_tot[h, 2] = ii
                end
            end


            if idata <= 2
                www = sample(1:(n*k), n_miss, replace = false)
                sort!(www)
                index_miss_mat = index_tot[www, :]
                for imiss = 1:n_miss
                    true_y[imiss] =
                        dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]]
                    dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]] =
                        rand(Normal(0.0, 1.0))
                end
            else
                index_miss_mat = zeros(Int64, 0, 0)
            end
            #www = sample(1:(n*k), n_miss, replace=false)
            #sort!(www)
            #index_miss_mat = index_tot[www, :]
            #for imiss = 1:n_miss
            #    true_y[imiss] = dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]]
            #    dataset[index_miss_mat[imiss, 1], index_miss_mat[imiss, 2]] = rand(Normal(0.0, 1.0))
            #end
            # ! Q distribution
            nm = Int64((k*k- k)/2)
            if iprior == 1
                prior_dist = Normal(0.0, 1.0)
            elseif iprior == 2
                prior_dist = Laplace(0.0, 1.0 / sqrt(2))
            else
                error("")
            end
            model_sparse = GenSparse_V2(Int64.(nm:(-1):1) ./ sum(Int64.(nm:(-1):1)), prior_dist)
            #if index_cor == 1

            #elseif index_cor == 2
            #    model_sparse = GenCorSparse_V2(Int64.(nm:(-1):1)./sum(Int64.(nm:(-1):1)),    prior_dist)
            #end

            #### MCMC SETUP ####
            z_mat_init = ones(Float64, k, k)
            molt = 1
            nm = Int64((k * k - k) / 2)

            # Set MCMC parameters: number of iterations, burn-in, and thinning
            mcmc_par = (iter = 10000 * molt, burnin = 8000 * molt, thin = 1 * molt)
            mcmc_par = (iter = 20, burnin = 15, thin = 1)
            covariate = ones(Float64, k, 1, n) # Covariate matrix (all ones, no covariate effect)



            #### RUN MCMC SAMPLER ####
            Random.seed!(1)
            out = mcmc(
                ydata = dataset, # Observed data
                iterations = mcmc_par, # MCMC parameters
                covariates = covariate,# Covariate matrix 
                model_sparse = model_sparse,
                #model_sparse=GenCorSparse_V2(Int64.(nm:(-1):1) ./ sum(Int64.(nm:(-1):1)),Laplace(0, 1/sqrt(2))),
                # Prior for sparsity pattern
                model_psi = GeneralS(), # Prior for psi parameter
                type_mcmc_lambda = HMC_DualAv_NoUTurn(;
                    epsilon = 10.000,
                    L = 10,
                    M = 10,
                    restart = false,
                    delta = 0.5,
                    iter_adapt = 10000000,
                ), # HMC sampler settings
                #type_mcmc_lambda=HMC_DualAv(; epsilon=10.000, L=10, M=10, restart=false, delta=0.5, iter_adapt=10000000, lambda =1.0), # HMC sampler settings
                #type_mcmc_lambda=BaseHMC(; epsilon=10.000, L=10, M=10, restart=false),
                prior_lambda = Wishart(1.0 * (k + [5,20][iprior] - 1), 1.0 .* Matrix(I(k))), # Wishart prior for precision matrix
                prior_mu = Normal(0.0, 1000.0^0.5),  # Prior for mean
                c_init = ones(Float64, k), # Initial value for c parameter
                m_init = zeros(Float64, nm), # Initial value for m parameter
                beta_init = zeros(Float64, size(covariate, 2)), # Initial value for regression coefficients
                prob_zeta = ones(Float64, k, k) * 0.5, # Initial probability for zeta (sparsity)
                z_init = Float64.(z_mat_init), # Initial value for z (sparsity pattern)
                iter_start_hmc = 2, # When to start HMC updates
                iter_start_zeta = 5, # When to start zeta updates
                missing_index = index_miss_mat, # Missing
            )


            #### EXTRACT AND SAVE RESULTS ####
            # Extract MCMC outputs
            m_out = out.m_out
            c_out = out.c_out
            qmat_sparse_out = out.q_mat_sparse_out
            lambda_out = out.lambda_out
            zmat_out = out.zmat_out
            beta_out = out.beta_out
            y = dataset
            missing_out = out.missing_out

            #### EXTRACT AND SAVE RESULTS ####
            # Extract MCMC outputs
            @rput y
            @rput missing_out
            @rput index_miss_mat
            @rput beta_out
            @rput lambda_out
            @rput name
            @rput k
            @rput zmat_out
            @rput true_y

            # Save the R workspace with all results for later analysis

            R"""
                library(ggplot2)
                library(reshape2)
                library(scales)
                library(patchwork)
                crps <- c()
                if(idata <= 2)
                {
                    for(iobs in 1:length(true_y))
                    {
                        miss <- missing_out[,iobs]
                        obs <- true_y[iobs]
                        crps[iobs] <- compute_crps(obs, miss)
                    }
                    
                }

                
                
                zeta_mean <- apply(zmat_out, c(2, 3), mean)
                zeta_mean[upper.tri(zeta_mean)] <- zeta_mean[lower.tri(zeta_mean)]
                df <- melt(zeta_mean)
                names(df) <- c("row", "col", "value")
                p1 <- ggplot(df, aes(x = col, y = row, fill = value)) +
                        geom_tile() +
                        scale_y_reverse() +
                        coord_equal() +
                        labs(title = "Zeta Mean", x = "Column", y = "Row", fill = "Value") + 
                        scale_fill_gradient2( low = "#2166AC", mid = "white", high = "#B2182B", oob = squish     )+
                        theme_minimal()
                
                zeta_low  <- apply(zmat_out, c(2, 3), quantile, probs = 0.025)
                zeta_high <- apply(zmat_out, c(2, 3), quantile, probs = 0.975)

                zeta_diff <- zeta_high - zeta_low
                zeta_diff[upper.tri(zeta_mean)] <- zeta_diff[lower.tri(zeta_mean)]
                df <- melt(zeta_diff)
                names(df) <- c("row", "col", "value")
                p2 <- ggplot(df, aes(x = col, y = row, fill = value)) +
                        geom_tile() +
                        scale_y_reverse() +
                        coord_equal() +
                        labs(title = "Zeta IC length", x = "Column", y = "Row", fill = "Value") + 
                        scale_fill_gradient2( low = "#2166AC", mid = "white", high = "#B2182B", oob = squish     )+
                        theme_minimal()

                lambda_mean <- apply(lambda_out, c(2, 3), mean)
                df <- melt(lambda_mean)
                names(df) <- c("row", "col", "value")
                p3 <- ggplot(df, aes(x = col, y = row, fill = value)) +
                        geom_tile() +
                        scale_y_reverse() +
                        coord_equal() +
                        labs(title = "lambda Mean", x = "Column", y = "Row", fill = "Value") + 
                        scale_fill_gradient2( low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0, oob = squish     )+
                        theme_minimal()
                
                lambda_low  <- apply(lambda_out, c(2, 3), quantile, probs = 0.025)
                lambda_high <- apply(lambda_out, c(2, 3), quantile, probs = 0.975)

                lambda_diff <- lambda_high - lambda_low
                df <- melt(lambda_diff)
                names(df) <- c("row", "col", "value")
                p4 <- ggplot(df, aes(x = col, y = row, fill = value)) +
                        geom_tile() +
                        scale_y_reverse() +
                        coord_equal() +
                        labs(title = "lambda IC length", x = "Column", y = "Row", fill = "Value") + 
                        scale_fill_gradient2( low = "#2166AC", mid = "white", high = "#B2182B", oob = squish     )+
                        theme_minimal()
                

                


                

                pdf(paste(dir,"/output/", name_tot, "mcmc_out.pdf", sep=""))
                par(mfrow=c(1,2))
                if(idata <= 2)
                {
                    plot(crps, pch=20)
                    #plot(hist(crps), main="CRPS of the missing values", xlab="CRPS", ylab="Frequency")

                    plot(true_y, colMeans(missing_out), pch=20, main="True vs Imputed Missing Values", xlab="True Values", ylab="Mean Imputed Values")
                    abline(a=0,b=1, col=2, lwd=2)
                }


                print(p1+p2+p3 +p4 + plot_layout(ncol=2))
                par(mfrow=c(3,3))
                for(irow in 1:(dim(lambda_out)[2]))
                {
                    for(icol in 1:(dim(lambda_out)[3]))
                    {
                        plot(lambda_out[,irow,icol], type="l", main=paste("lambda[", irow, ",", icol, "]", sep=""))
                    }
                }
                if(idata <= 2)
                {
                    for(imiss in 1:dim(missing_out)[2])
                    {
                        plot(missing_out[,imiss], type="l", main=paste("Missing value ", imiss, sep=""))
                        abline(h=true_y[imiss], col=2, lwd=2)
                    }
                }
                dev.off()


                save.image(paste(dir,"/output/", name_tot, ".Rdata",sep=""))
            """
        end


    end

end

