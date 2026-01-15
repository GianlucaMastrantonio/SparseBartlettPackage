library(BDgraph)
swish <- function(n = nsim, adj = zeta, b = beta, D = psi) {
  sim_nostra <- array(0, dim = c(k, k, n))
  # sim
  for (isim in 1:n)
  {
    order_z <- sample(1:k, k, replace = FALSE)
    mat_order <- matrix(0, nrow = k, ncol = k)
    for (i in 1:k) {
      mat_order[i, order_z[i]] <- 1
    }
    zeta_order <- mat_order %*% zeta %*% t(mat_order)

    Bmat <- matrix(0, nrow = k, ncol = k)
    for (icol in 1:(k - 1))
    {
      for (irow in (icol + 1):k)
      {
        Bmat[irow, icol] <- rnorm(1, 0, 1)
      }
    }
    for (ik in 1:k)
    {
      Bmat[ik, ik] <- sqrt(rchisq(1, b + sum(zeta_order[ik, ik:k]) - 1))
    }
    Bmat <- t(chol(D)) %*% Bmat

    Qmat <- matrix(0, nrow = k, ncol = k)
    diag(Qmat) <- diag(Bmat)
    for (icol in 1:(k - 1))
    {
      for (irow in (icol + 1):k)
      {
        if (zeta_order[irow, icol] == 1) {
          Qmat[irow, icol] <- Bmat[irow, icol]
        } else {
          if (icol == 1) {
            Qmat[irow, icol] <- 0
          } else {
            Qmat[irow, icol] <- -sum(Qmat[irow, 1:(icol - 1)] * Qmat[icol, 1:(icol - 1)]) / Qmat[icol, icol]
          }
        }
      }
    }
    cc <- Qmat %*% t(Qmat)
    round(cc, 6)
    sim_nostra[, , isim] <- t(mat_order) %*% cc %*% mat_order
  }
  return(sim_nostra)
}

set.seed(123)
k <- 20
n <- 10
p_zero <- 0.3
psi <- diag(1, k)
beta <- 3
nsim <- 10000

n_elem <- (k^2 - k) / 2
vec_elem <- matrix(0, nrow = n_elem, ncol = 2)
h <- 1
for (i in 1:(k - 1)) {
  for (j in (i + 1):k) {
    vec_elem[h, 1] <- j
    vec_elem[h, 2] <- i
    h <- h + 1
  }
}
w_zero <- sample(1:n_elem, size = round(0.3 * n_elem), replace = FALSE)
zeros <- vec_elem[w_zero, ]
zeta <- matrix(1, nrow = k, ncol = k)

for (i in 1:nrow(zeros)) {
  zeta[zeros[i, 1], zeros[i, 2]] <- 0
  zeta[zeros[i, 2], zeros[i, 1]] <- 0
}
# G-wishart
sim_g <- rgwish(n = nsim, adj = zeta, b = beta, D = psi)

## Nostra
sim_sw <- swish(n = nsim, adj = zeta, b = beta, D = psi)


pdf("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/SimPiero/sim.pdf")
par(mfrow = c(2, 2))
for (irow in 1:k)
{
  for (icol in irow:k)
  {
    q_1 <- quantile(sim_g[irow, icol, ], prob = c(0.025, 0.25, 0.5, 0.75, 0.975))
    q_2 <- quantile(sim_sw[irow, icol, ], prob = c(0.025, 0.25, 0.5, 0.75, 0.975))
    plot(sim_g[irow, icol, ], type = "l", main = paste("irow =", irow, "icol =", icol))
    plot(sim_sw[irow, icol, ], type = "l", col = 2)
    plot(density(sim_g[irow, icol, ]))
    lines(density(sim_sw[irow, icol, ]), col = 2)

    plot(c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2), c(q_1, q_2), col = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2), pch = 20, cex = 2, )
  }
}
dev.off()


pdf("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/SimPiero/sim_bu.pdf")
par(mfrow = c(2, 2))
for (ih in 1:2)
{
  print(ih)
  for (jh in (ih + 1):nrow(vec_elem))
  {
    irow <- vec_elem[ih, 1]
    icol <- vec_elem[ih, 2]
    irow2 <- vec_elem[jh, 1]
    icol2 <- vec_elem[jh, 2]
    plot(sim_g[irow, icol, ], sim_g[irow2, icol2, ])
    points(sim_sw[irow, icol, ], sim_sw[irow2, icol2, ], pch = 20, col = 2)
  }
}

# for (irow in 1:k)
# {
#  for (icol in irow:k)
#  {
#    print(paste("irow =", irow, "icol =", icol))
#    for (irow2 in irow:k)
#    {
#      for (icol2 in irow2:k)
#      {
#        plot(sim_g[irow, icol, ], sim_g[irow2, icol2, ])
#        points(sim_sw[irow, icol, ], sim_sw[irow2, icol2, ], pch = 20, col = 2)
#        # plot(sim_g[irow, icol, ], type = "l")
#        # plot(sim_sw[irow, icol, ], type = "l", col = 2)
#        # plot(density(sim_g[irow, icol, ]))
#        # lines(density(sim_sw[irow, icol, ]), col = 2)
#      }
#    }
#  }
# }
dev.off()
