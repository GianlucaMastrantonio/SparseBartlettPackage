k <- 10

Psi <- diag(k)
nu <- k + 2
nsim <- 10000
zmat <- matrix(1, ncol = k, nrow = k)
nm <- (k^2 - k) / 2

index_mat <- matrix(NA, nrow = nm, ncol = 2)
h <- 1
for (j in 1:(k - 1))
{
  for (i in (j + 1):k)
  {
    index_mat[h, ] <- c(i, j)
    h <- h + 1
  }
}

n_zero <- floor(nm * 0.3)
set.seed(1234)
zero_index <- sample(1:nm, n_zero, replace = FALSE)
zero_index <- sort(zero_index)

for (h in 1:n_zero)
{
  i <- index_mat[zero_index[h], 1]
  j <- index_mat[zero_index[h], 2]
  zmat[i, j] <- 0
}



h <- 1
for (j in 1:(k - 1))
{
  for (i in (j + 1):k)
  {
    index_mat[h, ] <- c(i, j)
    h <- h + 1
  }
}


Bmat <- matrix(0, nrow = k, ncol = k)
for (j in 1:(k - 1))
{
  for (i in (j + 1):k)
  {
    Bmat[i, j] <- rnorm(1, 0, 1)
  }
}

Qmat = Psi %*% Bmat
Qmat_star <- Qmat
Qmat_star[1, 1] <- 1
j <- 1
for (i in (j + 1):k)
{
  if (zmat[i, j] == 1) {
    Qmat_star[i, j] <- Qmat[i, j]
  } else {
    Qmat_star[i, j] <- 0
  }
}
for (j in 2:(k - 1))
{
  Qmat_star[j, j] <- sqrt(1 - sum(Qmat_star[j, 1:(j - 1)]^2)) # questa non è garantito sia positiva dentro la radice
  for (i in (j + 1):k)
  {
    if (zmat[i, j] == 1) {
      Qmat_star[i, j] <- Qmat[i, j]
    } else {
      Qmat_star[i, j] <- -sum(Qmat_star[i, 1:(j - 1)] * Qmat_star[i, 1:(j - 1)] / Qmat_star[j, j])
    }
  }
}
j <- 5
Qmat_star[j, j] <- 1 - sum(Qmat_star[j, 1:(j - 1)]^2)


round(Qmat_star, 3)
round(Qmat_star %*% t(Qmat_star), 3)
