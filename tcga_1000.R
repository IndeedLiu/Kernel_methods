library(dplyr)
library(tidyr)
library(purrr)
library(WeightIt)
library(data.table)
library(osqp)
library(causaldrf)
library(ks)
library(np)
source("/Users/liushucheng/Desktop/independence_weights_for_policy_learning/independence_weights.R")
# Kernel function
kernel_func <- function(x) {
  return(1 / sqrt(2 * pi) * exp(-0.5 * x^2))
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






dr_pseudo_outcome_value_function <- function(Y, A, X, weights, Aseq, h) {
  # 确保X是一个矩阵，并将其转换为数据框
  X <- as.data.frame(X)
  
  # 创建数据框并手动计算A^2和X的平方（逐列计算）
  A_squared <- A^2
  X_squared <- X^2  # 如果X是矩阵，则逐列计算平方
  
  # 将A、A_squared、X和X_squared合并到同一个数据框
  data <- data.frame(Y = Y, A = A, A_squared = A_squared, X, X_squared = X_squared)
  
  # 第一步：广义线性回归模型，包含A、A^2、X、X^2和A与X的交互项
  # 使用 A:X 代表A和每一列X的交互作用
  formula_y <- as.formula(paste("Y ~ A + A_squared +", 
                                paste(colnames(X), collapse = " + "), 
                                "+", paste(colnames(X), collapse = ":A + "), ":A"))

  # 使用广义线性回归拟合模型
  outcome_mod <- glm(formula_y, data = data)
  mu_x_a <- unname(outcome_mod$fitted.values)
  
  # 第二步：核方法计算值函数
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


# 样本量范围
sample_sizes <- seq(2000,6000,2000)

# 读取保存的数据集
data_matrix <- read.csv("dataset/tcga/data_matrix.csv", header = TRUE)
t_grid <- read.csv("dataset/tcga/t_grid.csv", header = TRUE)

# 样本划分
train_data <- data_matrix[1:6000, ]
test_data <- data_matrix[6001:nrow(data_matrix), ]
colnames(data_matrix) <- c("t", paste0("x", 1:(ncol(data_matrix) - 2)), "y")

# 初始化IRMSE结果的数据框
IRMSE_results_df <- data.frame()

# 定义h的候选值
h_values <- seq(0.1, 2, by = 0.1)
best_h <- 0.05




for (n in sample_sizes) {
  # 初始化用于存储各个方法结果的数据框，每次大循环开始时重置为空数据框
  gps_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  gps_DR_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  gbm_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  bart_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  # indep_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  # indep_DR_results <- data.frame(matrix(ncol = 0, nrow = 3659))
  
  # 小循环：重复多次计算IRMSE
  for (rep in 1:1) {
    # 从训练集中随机抽取n个样本
    set.seed(rep * n)
    
    sampled_idx <- sample(1:nrow(train_data), n)
    sampled_data <- train_data[sampled_idx, ]
    
    # 创建数据框
    data_df <- as.data.frame(sampled_data)
    
    colnames(data_df) <- c("t", paste0("x", 1:(ncol(data_df) - 2)), "y")
    

    # 使用交叉验证选择最优h值
    
    
    
    # 计算不同方法的权重
    formula <- as.formula(paste("t ~", paste(colnames(data_df)[2:(ncol(data_df) - 3001)], collapse = " + ")))
    gps_est <- safe_weightit(formula, method = "glm", data = data_df, stabilize = TRUE)
    gbm_est <- safe_weightit(formula, method = "gbm", data = data_df, stabilize = TRUE, criterion = "p.mean")
    bart_est <- safe_weightit(formula, method = "bart", data = data_df, stabilize = TRUE, n.trees = 250)
    # indep_weights <- independence_weights(data_df$t, data_df %>% select(x1:1000))
    
    # 将所有大于500的权重设为500
    gps_est$weights[gps_est$weights > 500] <- 500
    gbm_est$weights[gbm_est$weights > 500] <- 500
    bart_est$weights[bart_est$weights > 500] <- 500
    # indep_weights$weights[indep_weights$weights > 500] <- 500
    
    # 使用交叉验证选出的最优h值来计算value_function
    gps_value <- calculate_value_function(data_df$y, data_df$t, test_data[,1], gps_est$weights, best_h)
    gbm_value <- calculate_value_function(data_df$y, data_df$t, test_data[,1], gbm_est$weights, best_h)
    bart_value <- calculate_value_function(data_df$y, data_df$t, test_data[,1], bart_est$weights, best_h)
    # indep_value <- calculate_value_function(data_df$y, data_df$t, test_data[,1], indep_weights$weights, best_h)
   
    # 使用dr_pseudo_outcome_value_function计算gps_DR方法的value_function
    gps_DR_value <- dr_pseudo_outcome_value_function(data_df$y, data_df$t, data_df %>% select(starts_with("x")), gps_est$weights, test_data[,1], best_h)
    # indep_DR_value <- dr_pseudo_outcome_value_function(data_df$y, data_df$t, data_df %>% select(starts_with("x")), indep_weights$weights, test_data[,1], best_h)
    
    # 将当前rep的value_function结果添加到数据框中
    gps_results <- cbind(gps_results, gps_value)
    gbm_results <- cbind(gbm_results, gbm_value)
    bart_results <- cbind(bart_results, bart_value)
    # indep_results <- cbind(indep_results, indep_value)
    gps_DR_results <- cbind(gps_DR_results, gps_DR_value)
    # indep_DR_results <- cbind(indep_DR_results, indep_DR_value)
  }
  
  # 计算IRMSE
  gps_squared_errors <- rowMeans((gps_results - t(t_grid[2, 6001:nrow(data_matrix)]))^2)
  gps_DR_squared_errors <- rowMeans((gps_DR_results - t(t_grid[2,6001:nrow(data_matrix)]))^2)
  gbm_squared_errors <- rowMeans((gbm_results - t(t_grid[2, 6001:nrow(data_matrix)]))^2)
  bart_squared_errors <- rowMeans((bart_results - t(t_grid[2, 6001:nrow(data_matrix)]))^2)
  # indep_squared_errors <- rowMeans((indep_results - t(t_grid[2, 6001:nrow(data_matrix)]))^2)
  # indep_DR_squared_errors <- rowMeans((indep_DR_results - t(t_grid[2, 6001:nrow(data_matrix)]))^2)
  
  gps_rmse <- sqrt(gps_squared_errors)
  gps_DR_rmse <- sqrt(gps_DR_squared_errors)
  gbm_rmse <- sqrt(gbm_squared_errors)
  bart_rmse <- sqrt(bart_squared_errors)
  # indep_rmse <- sqrt(indep_squared_errors)
  # indep_DR_rmse <- sqrt(indep_DR_squared_errors)

  # 直接计算均值
  gps_IRMSE <- mean(gps_rmse)
  gbm_IRMSE <- mean(gbm_rmse)
  bart_IRMSE <- mean(bart_rmse)
  gps_DR_IRMSE <- mean(gps_DR_rmse)
  # indep_IRMSE <- mean(indep_rmse)
  # indep_DR_IRMSE <- mean(indep_DR_rmse)

  # 保存结果
  IRMSE_results <- data.frame(
      sample_size = n,
      gps_IRMSE = gps_IRMSE,
      gps_DR_IRMSE = gps_DR_IRMSE,
      gbm_IRMSE = gbm_IRMSE,
      bart_IRMSE = bart_IRMSE
  )

  IRMSE_results_df <- bind_rows(IRMSE_results_df, IRMSE_results)
  
  # 清空用于存储各个方法结果的数据框，为下一个sample_size的循环做好准备
  gps_results <- NULL
  gps_DR_results <- NULL
  gbm_results <- NULL
  bart_results <- NULL
  # indep_results <- NULL
  # indep_DR_results <- NULL
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
