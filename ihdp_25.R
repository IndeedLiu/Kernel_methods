library(dplyr)
library(tidyr)
library(purrr)
library(WeightIt)
library(data.table)
library(osqp)
library(causaldrf)
library(ks)
library(np)
source("independence_weights.R")
# Kernel function
kernel_func <- function(x) {
  return(1 / sqrt(2 * pi) * exp(-0.5 * x^2))
}

dr_pseudo_outcome_value_function <- function(Y, A, X, weights, Aseq, h) {
  data <- data.frame(Y = Y, A = A, X = X)
  
  # Step 1: 线性回归模型
  formula_t <- as.formula("A ~ .")
  y_formula <- as.formula("Y ~ A + .")
  outcome_mod <- lm(y_formula, data = data)
  mu_x_a <- unname(outcome_mod$fitted)
  
  # Step 2: 核方法替代局部多项式回归，计算 value function
  n <- length(Y)
  value_function_estimation <- numeric(length = length(Aseq))
  
  for (i in seq_along(Aseq)) {
    kernel_values <- sapply(A, function(a) kernel_func((Aseq[i] - a) / h))
    weights_adjusted <- weights * kernel_values
    
    numerator <- sum(weights_adjusted * (Y - mu_x_a))
    denominator <- sum(weights_adjusted)
    
    value_function_estimation[i] <- numerator / denominator
  }
  
  return(value_function_estimation)
}







# Function to calculate value function
calculate_value_function <- function(y, T, X, weights, h) {
  n <- length(y)
  value_function_estimation <- numeric(length = length(X))
  
  for (i in seq_along(X)) {
    kernel_values <- sapply(T, function(t) kernel_func((X[i] - t) / h))
    weights_adjusted <- weights * kernel_values
    
    numerator <- sum(weights_adjusted * y)
    denominator <- sum(weights_adjusted)
    
    value_function_estimation[i] <- numerator / denominator
  }
  
  return(value_function_estimation)
}




# 函数：生成数据矩阵和t_grid
generate_data_matrix_t_grid <- function(data_path = "ihdp.csv", n = 747) {
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
    t = x_t_link(t)
    y <- t_x_y(t, x)
    y <- y + rnorm(1, 0, 0.5)
    
    data_matrix[i, 1] <- t
    data_matrix[i, n_feature + 2] <- y
    data_matrix[i, 2:(n_feature + 1)] <- x
  }

  # 生成t_grid：t为测试集的t值
  t_grid <- matrix(0, nrow = 2, ncol = n_data)
  t_grid[1, ] <- data_matrix[, 1]  # 第一行是t值

  # 计算每个 t 值对应的 psi
  for (j in 1:n_data) {
    psi <- 0
    t <- t_grid[1, j]
    for (k in 1:n_data) {
      x <- data_matrix[k, 2:(n_feature + 1)]
      psi <- psi + t_x_y(t, x)
    }
    psi <- psi / n_data
    t_grid[2, j] <- psi
  }

  return(list(data_matrix = data_matrix, t_grid = t_grid))
}
cross_validate_h <- function(data_df, h_values, folds = 5) {
  set.seed(42)
  
  fold_indices <- sample(rep(1:folds, length.out = nrow(data_df)))
  errors <- matrix(NA, nrow = length(h_values), ncol = folds)
  
  for (i in 1:length(h_values)) {
    h_val <- h_values[i]
    
    for (fold in 1:folds) {
      train_fold <- as.data.frame(data_df[fold_indices != fold, ])
      test_fold <- as.data.frame(data_df[fold_indices == fold, ])
      colnames(train_fold) <- c("t", paste0("x", 1:(ncol(train_fold) - 2)), "y")
      colnames(test_fold) <- c("t", paste0("x", 1:(ncol(test_fold) - 2)), "y")
      # 获取 test_fold 在 data_df 中的原始下标
      test_indices <- which(fold_indices == fold)
      formula <- as.formula(paste("t ~", paste(colnames(train_fold)[2:(ncol(train_fold) - 1)], collapse = " + ")))
      gps_est <- weightit(formula, method = "glm", data = train_fold, stabilize = TRUE)
      
      # 计算 value_function
      gps_value <- calculate_value_function(train_fold$y, train_fold$t, test_fold$t, gps_est$weights, h_val)
      
      # 计算均方误差，确保 t_grid[2, ] 使用 test_indices 进行下标索引
      squared_errors <- mean((gps_value - t_grid[2, test_indices])^2)
      errors[i, fold] <- sqrt(squared_errors)
    }
  }
  
  # 计算每个 h 值的平均误差
  avg_errors <- rowMeans(errors, na.rm = TRUE)
  best_h <- h_values[which.min(avg_errors)]
  
  return(best_h)
}

safe_weightit <- function(formula, method, data, stabilize, ...) {
    result <- tryCatch({
        # 尝试使用 weightit 函数来计算权重
        weightit(formula, method = method, data = data, stabilize = stabilize, ...)
    }, error = function(e) {
        # 发生错误时的处理
        cat("Error occurred during weightit. Setting weights to 1.\n")
        # 假设权重的长度与数据的行数相同，设置所有权重为 1
        n <- nrow(data)
        return(list(weights = rep(1, n)))
    })
    
    # 检查是否成功返回结果，并确保没有 Inf 值
    if (any(is.infinite(result$weights))) {
        cat("Infinite weights detected. Setting all weights to 1.\n")
        result$weights <- rep(1, length(result$weights))
    }
    
    return(result)
}

# 样本量范围
sample_sizes <- seq(200, 501, 300)

n_data <- 747

data_results <- generate_data_matrix_t_grid(n = n_data)  # 使用全部数据生成
data_matrix <- data_results$data_matrix
t_grid <- data_results$t_grid

# 样本划分
train_data <- data_matrix[1:500, ]
test_data <- data_matrix[501:nrow(data_matrix), ]
colnames(data_matrix) <- c("t", paste0("x", 1:(ncol(data_matrix) - 2)), "y")
# 初始化IRMSE结果的数据框
IRMSE_results_df <- data.frame()

h_values <- seq(0.01, 0.1, by = 0.01)

best_h <- cross_validate_h(as.data.frame(data_matrix), h_values)


# 交叉验证函数


for (n in sample_sizes) {
  # 初始化用于存储各个方法结果的数据框，每次大循环开始时重置为空数据框
  gps_results <- data.frame(matrix(ncol = 0, nrow = 247))
  gps_DR_results <- data.frame(matrix(ncol = 0, nrow = 247))
  cbps_results <- data.frame(matrix(ncol = 0, nrow = 247))
  gbm_results <- data.frame(matrix(ncol = 0, nrow = 247))
  ebal_results <- data.frame(matrix(ncol = 0, nrow = 247))
  bart_results <- data.frame(matrix(ncol = 0, nrow = 247))
  indep_results <- data.frame(matrix(ncol = 0, nrow = 247))
  indep_DR_results <- data.frame(matrix(ncol = 0, nrow = 247))

  # 小循环：重复多次计算IRMSE
  for (rep in 1:100) {
    # 从训练集中随机抽取n个样本
    set.seed(rep * n)
    sampled_idx <- sample(1:nrow(train_data), n)
    sampled_data <- train_data[sampled_idx, ]
    
    # 创建数据框
    data_df <- as.data.frame(sampled_data)
    
    colnames(data_df) <- c("t", paste0("x", 1:(ncol(data_df) - 2)), "y")
    
    
    # 计算不同方法的权重
    formula <- as.formula(paste("t ~", paste(colnames(data_df)[2:(ncol(data_df) - 1)], collapse = " + ")))
    gps_est <- weightit(formula, method = "glm", data = data_df, stabilize = TRUE)
    summary(gps_est)
    cbps_est <- weightit(formula, method = "cbps", data = data_df, over = FALSE, stabilize = TRUE)
    summary(cbps_est)
    gbm_est <- weightit(formula, method = "gbm", data = data_df, stabilize = TRUE, criterion = "p.mean")
    ebal_est <- weightit(formula, method = "ebal", data = data_df, stabilize = TRUE)
    bart_est <- weightit(formula, method = "bart", data = data_df, stabilize = TRUE, n.trees = 250)
    indep_weights <- independence_weights(data_df$t, data_df %>% select(starts_with("x")))
    
    # 将所有大于500的权重设为500
    gps_est$weights[gps_est$weights > 500] <- 500
    cbps_est$weights[cbps_est$weights > 500] <- 500
    gbm_est$weights[gbm_est$weights > 500] <- 500
    ebal_est$weights[ebal_est$weights > 500] <- 500
    bart_est$weights[bart_est$weights > 500] <- 500
    indep_weights$weights[indep_weights$weights > 500] <- 500
    
    # 使用交叉验证选出的最优h值来计算value_function
    gps_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], gps_est$weights, best_h)
    cbps_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], cbps_est$weights, best_h)
    gbm_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], gbm_est$weights, best_h)
    ebal_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], ebal_est$weights, best_h)
    bart_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], bart_est$weights, best_h)
    indep_value <- calculate_value_function(data_df$y, data_df$t, test_data[, 1], indep_weights$weights, best_h)

    # 使用dr_pseudo_outcome_value_function计算gps_DR和indep_DR方法的value_function
    gps_DR_value <- dr_pseudo_outcome_value_function(data_df$y, data_df$t, data_df %>% select(starts_with("x")), gps_est$weights, test_data[, 1], best_h)
    indep_DR_value <- dr_pseudo_outcome_value_function(data_df$y, data_df$t, data_df %>% select(starts_with("x")), indep_weights$weights, test_data[, 1], best_h)
    
    # 将当前rep的value_function结果添加到数据框中
    gps_results <- cbind(gps_results, gps_value)
    gps_DR_results <- cbind(gps_DR_results, gps_DR_value)
    cbps_results <- cbind(cbps_results, cbps_value)
    gbm_results <- cbind(gbm_results, gbm_value)
    ebal_results <- cbind(ebal_results, ebal_value)
    bart_results <- cbind(bart_results, bart_value)
    indep_results <- cbind(indep_results, indep_value)
    indep_DR_results <- cbind(indep_DR_results, indep_DR_value)
  }
  
  gps_squared_errors <- rowMeans((gps_results - t_grid[2,501:747 ])^2)
  gps_DR_squared_errors <- rowMeans((gps_DR_results - t_grid[2, 501:747])^2)
  cbps_squared_errors <- rowMeans((cbps_results - t_grid[2, 501:747])^2)
  gbm_squared_errors <- rowMeans((gbm_results - t_grid[2, 501:747])^2)
  ebal_squared_errors <- rowMeans((ebal_results - t_grid[2, 501:747])^2, na.rm = TRUE) 
  bart_squared_errors <- rowMeans((bart_results - t_grid[2,501:747 ])^2)
  indep_squared_errors <- rowMeans((indep_results - t_grid[2, 501:747])^2)
  indep_DR_squared_errors <- rowMeans((indep_DR_results - t_grid[2,501:747 ])^2)
  
  gps_rmse <- sqrt(gps_squared_errors)
  gps_DR_rmse <- sqrt(gps_DR_squared_errors)
  cbps_rmse <- sqrt(cbps_squared_errors)
  gbm_rmse <- sqrt(gbm_squared_errors)
  ebal_rmse <- sqrt(ebal_squared_errors)
  bart_rmse <- sqrt(bart_squared_errors)
  indep_rmse <- sqrt(indep_squared_errors)
  indep_DR_rmse <- sqrt(indep_DR_squared_errors)

  # 直接计算均值
  gps_IRMSE <- mean(gps_rmse)
  gps_DR_IRMSE <- mean(gps_DR_rmse)
  cbps_IRMSE <- mean(cbps_rmse)
  gbm_IRMSE <- mean(gbm_rmse)
  ebal_IRMSE <- mean(ebal_rmse)
  bart_IRMSE <- mean(bart_rmse)
  indep_IRMSE <- mean(indep_rmse)
  indep_DR_IRMSE <- mean(indep_DR_rmse)

  # 保存结果
  IRMSE_results <- data.frame(
      sample_size = n,
      gps_IRMSE = gps_IRMSE,
      gps_DR_IRMSE = gps_DR_IRMSE,
      cbps_IRMSE = cbps_IRMSE,
      gbm_IRMSE = gbm_IRMSE,
      ebal_IRMSE = ebal_IRMSE,
      bart_IRMSE = bart_IRMSE,
      indep_IRMSE = indep_IRMSE,
      indep_DR_IRMSE = indep_DR_IRMSE
  )

  IRMSE_results_df <- bind_rows(IRMSE_results_df, IRMSE_results)
  
  # 清空用于存储各个方法结果的数据框，为下一个sample_size的循环做好准备
  gps_results <- NULL
  gps_DR_results <- NULL
  cbps_results <- NULL
  gbm_results <- NULL
  ebal_results <- NULL
  bart_results <- NULL
  indep_results <- NULL
  indep_DR_results <- NULL
}

# 保存IRMSE结果并打印
write.csv(IRMSE_results_df, "IRMSE_results.csv", row.names = FALSE)
print(IRMSE_results_df)


# 画图展示IRMSE结果
library(ggplot2)
IRMSE_plot <- ggplot(IRMSE_results_df, aes(x = sample_size)) +
  geom_line(aes(y = gps_IRMSE, color = "GPS")) +
  geom_line(aes(y = gps_DR_IRMSE, color = "GPS_DR")) +
  geom_line(aes(y = cbps_IRMSE, color = "CBPS")) +
  geom_line(aes(y = gbm_IRMSE, color = "GBM")) +
  geom_line(aes(y = ebal_IRMSE, color = "EBAL")) +
  geom_line(aes(y = bart_IRMSE, color = "BART")) +
  geom_line(aes(y = indep_IRMSE, color = "Independence Weights")) +
  geom_line(aes(y = indep_DR_IRMSE, color = "Indep_DR")) +
  labs(title = "IRMSE vs Sample Size", y = "IRMSE", x = "Sample Size") +
  scale_color_manual(name = "Methods", values = c("GPS" = "blue", "GPS_DR" = "cyan", "CBPS" = "red", "GBM" = "green", "EBAL" = "purple", "BART" = "orange", "Independence Weights" = "brown", "Indep_DR" = "pink"))

print(IRMSE_plot)
