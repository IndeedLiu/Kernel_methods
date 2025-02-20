# independence_weights.R

# Required libraries
library(osqp)

# independence_weights function
independence_weights <- function(A, 
                                 X, 
                                 lambda = 0, 
                                 decorrelate_moments = FALSE,
                                 preserve_means = FALSE,
                                 dimension_adj = TRUE) {
  weights <- rep(1, NROW(A))
  
  n <- NROW(A)
  p <- NCOL(X)
  gamma <- 1
  
  stopifnot(gamma >= 0)
  
  Xdist  <- as.matrix(dist(X))
  Adist  <- as.matrix(dist(A))
  
  stopifnot(n == NROW(X))
  
  ## terms for energy-dist(Wtd A, A)
  Q_energy_A  <- -Adist / n ^ 2
  aa_energy_A <- 1 * as.vector(rowSums(Adist)) / (n ^ 2)
  
  ## terms for energy-dist(Wtd X, X)
  Q_energy_X  <- -Xdist / n ^ 2
  aa_energy_X <- 1 * as.vector(rowSums(Xdist)) / (n ^ 2)
  
  mean_Adist <- mean(Adist)
  mean_Xdist <- mean(Xdist)
  
  Xmeans <- colMeans(Xdist)
  Xgrand_mean <- mean(Xmeans)
  XA <- Xdist + Xgrand_mean - outer(Xmeans, Xmeans, "+")
  
  Ameans <- colMeans(Adist)
  Agrand_mean <- mean(Ameans)
  AA <- Adist + Agrand_mean - outer(Ameans, Ameans, "+")
  
  ## quadratic term for weighted total distance covariance
  P <- XA * AA / n ^ 2
  
  #Constraints: positive weights, weights sum to n
  if (preserve_means) {
    if (decorrelate_moments) {
      Constr_mat <- drop(scale(A, scale=FALSE)) * scale(X, scale=FALSE)
      Amat <- rbind(diag(n), rep(1, n), t(X), A, t(Constr_mat))
      
      lvec <- c(rep(0, n), n, colMeans(X), mean(A), rep(0, ncol(X)))
      uvec <- c(rep(Inf, n), n, colMeans(X), mean(A), rep(0, ncol(X)))
    } else {
      Amat <- rbind(diag(n), rep(1, n), t(X), A)
      
      lvec <- c(rep(0, n), n, colMeans(X), mean(A))
      uvec <- c(rep(Inf, n), n, colMeans(X), mean(A))
    }
  } else {
    if (decorrelate_moments) {
      Constr_mat <- drop(scale(A, scale=FALSE)) * scale(X, scale=FALSE)
      
      Amat <- rbind(diag(n), rep(1, n), t(Constr_mat))
      lvec <- c(rep(0, n), n, rep(0, ncol(X)))
      uvec <- c(rep(Inf, n), n, rep(0, ncol(X)))
    } else {
      Amat <- rbind(diag(n), rep(1, n))
      lvec <- c(rep(0, n), n)
      uvec <- c(rep(Inf, n), n)
    }
  }
  
  if (dimension_adj) {
    Q_energy_A_adj <- 1 / sqrt(p)
    Q_energy_X_adj <- 1
    
    sum_adj <- 1*(Q_energy_A_adj + Q_energy_X_adj)
    
    Q_energy_A_adj <- Q_energy_A_adj / sum_adj
    Q_energy_X_adj <- Q_energy_X_adj / sum_adj
    
  } else {
    Q_energy_A_adj <- Q_energy_X_adj <- 1/2
  }
  
  #Optimize. try up to 15 times until there isn't a weird failure of solve_osqp()
  for (na in 1:15) {
    opt.out <- solve_osqp(2 * (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X * Q_energy_X_adj) + lambda * diag(n) / n ^ 2 ),
                          q = 2 * gamma * (aa_energy_A * Q_energy_A_adj + aa_energy_X * Q_energy_X_adj),
                          A = Amat, l = lvec, u = uvec,
                          pars = osqp::osqpSettings(max_iter = 2e5,
                                                    eps_abs = 1e-8,
                                                    eps_rel = 1e-8,
                                                    verbose = FALSE))
    
    if (!identical(opt.out$info$status, "maximum iterations reached") & !(any(opt.out$x > 1e5))) {
      break
    }
  }
  
  weights <- opt.out$x
  weights[weights < 0] <- 0 #due to numerical imprecision
  
  ## quadratic part of the overall objective function
  QM_unpen <- (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X * Q_energy_X_adj) )
  quadpart_unpen <- drop(t(weights) %*% QM_unpen %*% weights)
  quadpart_unweighted <- sum(QM_unpen)
  quadpart <- quadpart_unpen + sum(weights ^ 2) * lambda / n ^ 2
  
  ## linear part of the overall objective function
  qvec <- 2 * gamma * (aa_energy_A * Q_energy_A_adj + aa_energy_X * Q_energy_X_adj)
  linpart  <- drop(weights %*% qvec)
  linpart_unweighted  <- sum(qvec)
  
  ## objective function
  objective_history <- quadpart + linpart + gamma*(-1*mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)
  
  ## D(w)
  D_w <- quadpart_unpen + linpart + gamma*(-1*mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)
  
  D_unweighted <- quadpart_unweighted + linpart_unweighted + gamma*(-1*mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj)
  
  qvec_full <- 2 * (aa_energy_A * Q_energy_A_adj + aa_energy_X * Q_energy_X_adj)
  
  quadpart_energy_A <- drop(t(weights) %*% ((Q_energy_A)) %*% weights) * Q_energy_A_adj
  quadpart_energy_X <- drop(t(weights) %*% ((Q_energy_X)) %*% weights) * Q_energy_X_adj
  
  quadpart_energy <- quadpart_energy_A * Q_energy_A_adj + quadpart_energy_X * Q_energy_X_adj
  
  distcov_history <- drop(t(weights) %*% P %*% weights)
  
  unweighted_dist_cov <- sum(P)
  
  linpart_energy   <- drop(weights %*% qvec_full)
  linpart_energy_A <- 2 * drop(weights %*% (aa_energy_A)) * Q_energy_A_adj
  linpart_energy_X <- 2 * drop(weights %*% (aa_energy_X)) * Q_energy_X_adj
  
  energy_history <- quadpart_energy + linpart_energy - mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj
  energy_A <- quadpart_energy_A + linpart_energy_A - mean_Adist * Q_energy_A_adj
  energy_X <- quadpart_energy_X + linpart_energy_X - mean_Xdist * Q_energy_X_adj
  
  ess <- (sum(weights)) ^ 2 / sum(weights ^ 2)
  
  ret_obj <- list(weights = weights, 
                  A = A,
                  opt = opt.out, 
                  objective = objective_history,
                  D_unweighted = D_unweighted,
                  D_w = D_w,
                  distcov_unweighted = unweighted_dist_cov,
                  distcov_weighted = distcov_history,
                  energy_A = energy_A,
                  energy_X = energy_X,
                  ess = ess)
  
  class(ret_obj) <- c("independence_weights")
  
  return(ret_obj)              
}
