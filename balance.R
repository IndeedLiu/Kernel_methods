library(dplyr)
library(tidyr)
library(purrr)
library(WeightIt)
library(data.table)
library(osqp)
library(causaldrf)
library(ks)
library(np)
library(cobalt)

independence_weights <- function(A, 
                                 X, 
                                 lambda = 0, 
                                 decorrelate_moments = FALSE,
                                 preserve_means = FALSE,
                                 dimension_adj = TRUE)
{
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
  if (preserve_means)
  {
    if (decorrelate_moments)
    {
      Constr_mat <- drop(scale(A, scale=FALSE)) * scale(X, scale=FALSE)
      Amat <- rbind(diag(n), rep(1, n), t(X), A, t(Constr_mat))
      
      lvec <- c(rep(0, n), n, colMeans(X), mean(A), rep(0, ncol(X)))
      uvec <- c(rep(Inf, n), n, colMeans(X), mean(A), rep(0, ncol(X)))
    } else
    {
      Amat <- rbind(diag(n), rep(1, n), t(X), A)
      
      lvec <- c(rep(0, n), n, colMeans(X), mean(A))
      uvec <- c(rep(Inf, n), n, colMeans(X), mean(A))
    }
    
  } else 
  {
    if (decorrelate_moments)
    {
      #Constr_mat <- (A - mean(A)) * scale(X, scale = FALSE)
      Constr_mat <- drop(scale(A, scale=FALSE)) * scale(X, scale=FALSE)
      
      Amat <- rbind(diag(n), rep(1, n), t(Constr_mat))
      lvec <- c(rep(0, n), n, rep(0, ncol(X)))
      uvec <- c(rep(Inf, n), n, rep(0, ncol(X)))
    } else
    {
      Amat <- rbind(diag(n), rep(1, n))
      lvec <- c(rep(0, n), n)
      uvec <- c(rep(Inf, n), n)
    }
  }
  
  if (dimension_adj)
  {
    Q_energy_A_adj <- 1 / sqrt(p)
    Q_energy_X_adj <- 1
    
    sum_adj <- 1*(Q_energy_A_adj + Q_energy_X_adj)
    
    Q_energy_A_adj <- Q_energy_A_adj / sum_adj
    Q_energy_X_adj <- Q_energy_X_adj / sum_adj
    
  } else
  {
    Q_energy_A_adj <- Q_energy_X_adj <- 1/2
  }
  
  #Optimize. try up to 15 times until there isn't a weird failure of solve_osqp()
  for (na in 1:15)
  {
    opt.out <- solve_osqp(2 * (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X * Q_energy_X_adj) + lambda * diag(n) / n ^ 2 ),
                          q = 2 * gamma * (aa_energy_A * Q_energy_A_adj + aa_energy_X * Q_energy_X_adj),
                          A = Amat, l = lvec, u = uvec,
                          pars = osqp::osqpSettings(max_iter = 2e5,
                                                    eps_abs = 1e-8,
                                                    eps_rel = 1e-8,
                                                    verbose = FALSE))
    
    if (!identical(opt.out$info$status, "maximum iterations reached") & !(any(opt.out$x > 1e5)))
    {
      break
    }
  }
  
  weights <- opt.out$x
  weights[weights < 0] <- 0 #due to numerical imprecision
  
  ## quadratic part of the overall objective function
  QM_unpen <- (P + gamma * (Q_energy_A * Q_energy_A_adj + Q_energy_X * Q_energy_X_adj) )
  #QM <- QM_unpen + lambda * diag(n)
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
  
  #quadpart_energy <- drop(t(weights) %*% ((Q_energy_A + Q_energy_X)) %*% weights)
  
  quadpart_energy_A <- drop(t(weights) %*% ((Q_energy_A)) %*% weights) * Q_energy_A_adj
  quadpart_energy_X <- drop(t(weights) %*% ((Q_energy_X)) %*% weights) * Q_energy_X_adj
  
  quadpart_energy <- quadpart_energy_A * Q_energy_A_adj + quadpart_energy_X * Q_energy_X_adj
  
  distcov_history <- drop(t(weights) %*% P %*% weights)
  
  unweighted_dist_cov <- sum(P)
  
  linpart_energy   <- drop(weights %*% qvec_full)
  linpart_energy_A <- 2 * drop(weights %*% (aa_energy_A)) * Q_energy_A_adj
  linpart_energy_X <- 2 * drop(weights %*% (aa_energy_X)) * Q_energy_X_adj
  
  ## sum of energy-dist(wtd A, A)+energy-dist(wtd X, X)
  energy_history <- quadpart_energy + linpart_energy - mean_Xdist * Q_energy_X_adj - mean_Adist * Q_energy_A_adj
  
  ## energy-dist(wtd A, A)
  energy_A <- quadpart_energy_A + linpart_energy_A - mean_Adist * Q_energy_A_adj
  
  ## energy-dist(wtd X, X)
  energy_X <- quadpart_energy_X + linpart_energy_X - mean_Xdist * Q_energy_X_adj
  
  objective_history <- objective_history
  energy_history    <- energy_history
  
  ess <- (sum(weights)) ^ 2 / sum(weights ^ 2)
  
  ret_obj <- list(weights = weights, 
                  A = A,
                  opt = opt.out, 
                  objective = objective_history,     ### the actual objective function value
                  D_unweighted = D_unweighted,
                  D_w = D_w,
                  distcov_unweighted = unweighted_dist_cov,
                  distcov_weighted = distcov_history,         ### the weighted total distance covariance
                  energy_A = energy_A,               ### Energy(Wtd Treatment, Treatment)
                  energy_X = energy_X,               ### Energy(Wtd X, X)
                  ess = ess)        # effective sample size
  
  class(ret_obj) <- c("independence_weights")
  
  return(ret_obj)              
}

# 函数：生成数据矩阵和t_grid
generate_data_matrix_t_grid <- function(data_path = "ihdp.csv", n = 500) {
  ihdp <- read.csv(data_path)
  ihdp <- ihdp[, 3:27]  # 删除第一列（数据索引）和第二列（处理）
  
  normalize_data <- function(data) {
    apply(data, 2, function(x) (x - min(x)) / (max(x) - min(x)))
  }

  ihdp <- as.matrix(normalize_data(ihdp))

  n_feature <- ncol(ihdp)
  n_data <- if (!is.null(n)) n else nrow(ihdp)  # 如果没有指定 n，则使用全部数据

  if (!is.null(n)) {
    ihdp <- ihdp[1:n, ]  # 读取前 n 行数据
  }

  cate_idx1 <- c(4, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  cate_idx2 <- c(16, 17, 18, 19, 20, 21, 22, 23, 24, 25)

  alpha <- 5
  cate_mean1 <- mean(rowMeans(ihdp[, cate_idx1]))
  cate_mean2 <- mean(rowMeans(ihdp[, cate_idx2]))

  x_t <- function(x) {
    x1 <- x[1]
    x2 <- x[2]
    x3 <- x[3]
    x4 <- x[5]
    x5 <- x[6]
    
    t <- x1 / (1 + x2) + max(x3, x4, x5) / (0.2 + min(x3, x4, x5)) + tanh((sum(x[cate_idx2]) / 10 - cate_mean2) * alpha) - 2
    
    return(t)
  }

  x_t_link <- function(t) {
    return(1 / (1 + exp(-2 * t)))
  }

  t_x_y <- function(t, x) {
    x1 <- x[1]
    x2 <- x[2]
    x3 <- x[3]
    x4 <- x[5]
    x5 <- x[6]
    factor1 <- 1
    factor2 <- 1
    y <- 1 / (1.2 - t) * sin(t * 3 * pi) * (
      factor1 * tanh((sum(x[cate_idx1]) / 10 - cate_mean1) * alpha) +
      factor2 * exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4))
    ) 
    return(y)
  }

  data_matrix <- matrix(0, nrow = n_data, ncol = n_feature + 2)

  for (i in 1:n_data) {
    x <- ihdp[i, ]
    
    t <- x_t(x)
    t <- t + rnorm(1, 0, 0.5)
    t <- x_t_link(t)
    y <- t_x_y(t, x)
    y <- y + rnorm(1, 0, 0.5)
    
    data_matrix[i, 1] <- t
    data_matrix[i, n_feature + 2] <- y
    data_matrix[i, 2:(n_feature + 1)] <- x
  }

  # 生成 0 到 1 之间间隔相等的 100 个点
  t_grid <- matrix(0, nrow = 2, ncol = 100)
  t_grid[1, ] <- seq(0, 1, length.out = 100)

  # 计算每个 t 值对应的 psi
  for (i in 1:100) {
    t <- t_grid[1, i]
    psi <- 0
    for (j in 1:n_data) {
      x <- data_matrix[j, 2:(n_feature + 1)]
      psi <- psi + t_x_y(t, x)
    }
    psi <- psi / n_data
    t_grid[2, i] <- psi
  }

  return(list(data_matrix = data_matrix, t_grid = t_grid))
}

# 样本量范围
sample_sizes <- seq(500,521,20)

# 用于存储所有结果的数据框
results_df <- data.frame()

for (rep in 1:1) {
  for (n in sample_sizes) {
    cat("Generating data for sample size:", n, "repetition:", rep, "\n")
    
    # 生成数据矩阵和t_grid
    data_results <- generate_data_matrix_t_grid(n=n)  # 修改此处以生成 n 个数据
    data_matrix <- data_results$data_matrix
    t <- data_matrix[, 1]
    t_grid <- data_results$t_grid

    # 创建数据框
    data_df <- as.data.frame(data_matrix)
    colnames(data_df) <- c("t", paste0("x", 1:(ncol(data_df) - 2)), "y")
    
    # 加权计算
    formula <- as.formula(paste("t ~", paste(colnames(data_df)[2:(ncol(data_df) - 1)], collapse = " + ")))
    gps_est <- weightit(formula, method = "glm", data = data_df, stabilize = TRUE)
   
    cbps_est <- weightit(formula, method = "cbps", data = data_df, over = FALSE, stabilize = TRUE)
    
    gbm_est <- weightit(formula, method = "gbm", data = data_df, stabilize = TRUE)
    
    ebal_est <- weightit(formula, method = "ebal", data = data_df, stabilize = TRUE)
    
    bart_est <- weightit(formula, method = "bart", data = data_df, stabilize = TRUE, n.trees = 250)
    
    # 使用 independence_weights 计算权重
    indep_weights <- independence_weights(data_df$t, data_df %>% select(starts_with("x"))) 

    

    # 将所有大于500的权重设为500
    gps_est$weights[gps_est$weights > 500] <- 500
    cbps_est$weights[cbps_est$weights > 500] <- 500
    gbm_est$weights[gbm_est$weights > 500] <- 500
    ebal_est$weights[ebal_est$weights > 500] <- 500
    bart_est$weights[bart_est$weights > 500] <- 500
    indep_weights$weights[indep_weights$weights > 500] <- 500

    weights_df <- data.frame(
      gps_weights = gps_est$weights,
      cbps_weights = cbps_est$weights,
      gbm_weights = gbm_est$weights,
      ebal_weights = ebal_est$weights,
      bart_weights = bart_est$weights,
      indep_weights = indep_weights$weights
    )
    
    # 合并数据框
    combined_df <- data_df %>%
      mutate(sample_size = n, repetition = rep) %>%
      bind_cols(weights_df %>% select(gps_weights, cbps_weights, gbm_weights, ebal_weights, bart_weights, indep_weights))

    # 将结果添加到总数据框中
    results_df <- bind_rows(results_df, combined_df)
  }
}

# Filter results for a specific sample size (e.g., n = 500)
sample_size <- 500
data_n <- results_df %>% filter(sample_size == sample_size)

# Define the formula used in weighting methods
formula <- as.formula(paste("t ~", paste(colnames(data_n)[2:(ncol(data_n) - 1)], collapse = " + ")))

# Create balance tables for each weighting method
bal_tab_gps <- bal.tab(data_n, treat = "t", weights = "gps_weights", formula = formula)
bal_tab_cbps <- bal.tab(data_n, treat = "t", weights = "cbps_weights", formula = formula)
bal_tab_gbm <- bal.tab(data_n, treat = "t", weights = "gbm_weights", formula = formula)
bal_tab_ebal <- bal.tab(data_n, treat = "t", weights = "ebal_weights", formula = formula)
bal_tab_bart <- bal.tab(data_n, treat = "t", weights = "bart_weights", formula = formula)
bal_tab_indep <- bal.tab(data_n, treat = "t", weights = "indep_weights", formula = formula)

# Plot balance for a selected variable (e.g., "x1")
par(mfrow = c(3, 2))  # Set up the plotting area to display 3x2 grid

bal.plot(bal_tab_gps, var.name = "x1", type = "density", main = "GPS Weights")
bal.plot(bal_tab_cbps, var.name = "x1", type = "density", main = "CBPS Weights")
bal.plot(bal_tab_gbm, var.name = "x1", type = "density", main = "GBM Weights")
bal.plot(bal_tab_ebal, var.name = "x1", type = "density", main = "EBAL Weights")
bal.plot(bal_tab_bart, var.name = "x1", type = "density", main = "BART Weights")
bal.plot(bal_tab_indep, var.name = "x1", type = "density", main = "Independence Weights")

# Reset plotting area
par(mfrow = c(1, 1))