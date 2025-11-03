New <- new.env()
load("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/out/test.Rdata", envir = New)
Old <- new.env()
load("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/out prec/GeneExp50_missmcmc_out_lambda.Rdata", envir = Old)

library(ggplot2)
k = dim(New$lambda_out)[2]

pdf("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/out/plot1.pdf")
par(mfrow = c(3, 3))

for (i in 1:k)
{
  for (j in 1:k)
  {
    # data_plot <- data.frame(
    #  l = c(New$lambda_out[, i, j], Old$lambda_out[, i, j]),
    #  iter = rep(1:dim(New$lambda_out)[1], 2),
    #  type = rep(c("New", "Old"), each = dim(New$lambda_out)[1])
    # )

    # p <- ggplot(data_plot, aes(x = iter, y = l, color = type)) +
    #  geom_line() +
    #  facet_wrap(~ interaction(i, j), scales = "free") +
    #  labs(title = "Lambda Estimates", x = "Iteration", y = "Value")
    plot(New$lambda_out[, i, j], type = "l", main = paste("Lambda[", i, ",", j, "]"), ylim = range(c(New$lambda_out[, i, j], Old$lambda_out[, i, j])))
    lines(Old$lambda_out[, i, j], col = "red")
    # print(p)
  }
}
dev.off()


pdf("/Users/gianlucamastrantonio/Politecnico di Torino Staff Dropbox/Gianluca Mastrantonio/lavori/gitrepo/SparseBartlettPackage/TestNewCode/out/miss.pdf")
par(mfrow = c(3, 3))

for (i in 1:dim(New$missing_out)[2])
{
  # data_plot <- data.frame(
  #  l = c(New$lambda_out[, i, j], Old$lambda_out[, i, j]),
  #  iter = rep(1:dim(New$lambda_out)[1], 2),
  #  type = rep(c("New", "Old"), each = dim(New$lambda_out)[1])
  # )

  # p <- ggplot(data_plot, aes(x = iter, y = l, color = type)) +
  #  geom_line() +
  #  facet_wrap(~ interaction(i, j), scales = "free") +
  #  labs(title = "Lambda Estimates", x = "Iteration", y = "Value")
  plot(New$missing_out[, i], type = "l", main = paste("Lambda[", i, ",", j, "]"), ylim = range(c(New$missing_out[, i], Old$missing_out[, i])))
  lines(Old$missing_out[, i], col = "red")
  # print(p)
}
dev.off()
