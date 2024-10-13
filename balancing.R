library(dplyr)
library(tidyr)
library(purrr)
library(WeightIt)
library(data.table)
library(osqp)
library(causaldrf)
library(ks)
library(np)
library(cobalt)  # Added cobalt library for love.plot

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




generate_news_matrix_t_grid <- function(data_path = "/Users/liushucheng/Desktop/independence_weights_for_policy_learning/news.csv", n_eval = 10, n_tune = 2) {
  # 读取并归一化数据
  
  news <- read.csv(data_path)
  
  news <- news %>% select_if(is.numeric)  # 选择数值列
  
  normalize_data <- function(data) {
    apply(data, 2, function(x) (x - min(x)) / (max(x) - min(x)))
  }

  news <- as.matrix(normalize_data(news))
  
  num_data <- nrow(news)
  num_feature <- ncol(news)
  
  set.seed(5)
  v1 <- rnorm(num_feature)
  v1 <- v1 / sqrt(sum(v1^2))
  v2 <- rnorm(num_feature)
  v2 <- v2 / sqrt(sum(v2^2))
  v3 <- rnorm(num_feature)
  v3 <- v3 / sqrt(sum(v3^2))

  # 定义x_t函数
  x_t <- function(x) {
    alpha <- 2
    tt <- sum(v3 * x) / (2 * sum(v2 * x))
    beta <- (alpha - 1) / tt + 2 - alpha
    beta <- abs(beta) + 0.0001
    t <- rbeta(1, alpha, beta)
    return(t)
  }
  
  # 定义t_x_y函数
  t_x_y <- function(t, x) {
    res1 <- max(-2, min(2, exp(0.3 * (sum(3.14159 * sum(v2 * x) / sum(v3 * x)) - 1))))
    res2 <- 20 * sum(v1 * x)
    res <- 2 * (4 * (t - 0.5)^2 * sin(0.5 * 3.14159 * t)) * (res1 + res2)
    return(res)
  }
  
  # 初始化数据矩阵
  data_matrix <- matrix(0, nrow = num_data, ncol = num_feature + 2)
  
  for (i in 1:num_data) {
    x <- news[i, ]
    t <- x_t(x)
    y <- t_x_y(t, x)
    y <- y + rnorm(1, 0, sqrt(0.5))
    
    data_matrix[i, 1] <- t
    data_matrix[i, num_feature + 2] <- y
    data_matrix[i, 2:(num_feature + 1)] <- x
  }
  
  # 生成t_grid
  t_grid <- matrix(0, nrow = 2, ncol = num_data)
  t_grid[1, ] <- data_matrix[, 1]  # 第一行是t值
  
  for (i in 1:num_data) {
    psi <- 0
    t <- t_grid[1, i]
    for (j in 1:num_data) {
      x <- data_matrix[j, 2:(num_feature + 1)]
      psi <- psi + t_x_y(t, x)
    }
    psi <- psi / num_data
    t_grid[2, i] <- psi
  }

  
  
  return(list(data_matrix = data_matrix, t_grid = t_grid))
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
# 样本量范围
sample_sizes <- seq(2000, 2001, 1000)

n_data <- 2990

data_results <- generate_news_matrix_t_grid()  # 使用全部数据生成
data_matrix <- data_results$data_matrix
t_grid <- data_results$t_grid

# 样本划分
train_data <- data_matrix[1:2000, ]
test_data <- data_matrix[2001:nrow(data_matrix), ]

# 初始化IRMSE结果的数据框
IRMSE_results_df <- data.frame()

# 定义h的候选值
h_values <- seq(0.1, 2, by = 0.1)

# 交叉验证函数
cross_validate_h <- function(data_df, folds = 5) {
  set.seed(42)
  fold_indices <- sample(rep(1:folds, length.out = nrow(data_df)))
  errors <- matrix(NA, nrow = length(h_values), ncol = folds)
  
  for (i in 1:length(h_values)) {
    h_val <- h_values[i]
    
    for (fold in 1:folds) {
      train_fold <- data_df[fold_indices != fold, ]
      test_fold <- data_df[fold_indices == fold, ]
      
      # 计算不同方法的权重
      gps_est <- safe_weightit(formula, method = "glm", data = train_fold, stabilize = TRUE)
      
      # 计算value_function
      gps_value <- calculate_value_function(train_fold$y, train_fold$t, test_fold$t, gps_est$weights, h_val)
      
      # 计算均方误差
      squared_errors <- mean((gps_value - t_grid[2, ])^2)
      errors[i, fold] <- sqrt(squared_errors)
    }
  }
  
  # 计算每个h值的平均误差
  avg_errors <- rowMeans(errors, na.rm = TRUE)
  best_h <- h_values[which.min(avg_errors)]
  
  return(best_h)
}

for (n in sample_sizes) {
  # 初始化用于存储各个方法结果的数据框，每次大循环开始时重置为空数据框
  gps_results <- data.frame(matrix(ncol = 0, nrow = 993))
  gps_DR_results <- data.frame(matrix(ncol = 0, nrow = 993))
  cbps_results <- data.frame(matrix(ncol = 0, nrow = 993))
  gbm_results <- data.frame(matrix(ncol = 0, nrow = 993))
  #ebal_results <- data.frame(matrix(ncol = 0, nrow = 993))
  #bart_results <- data.frame(matrix(ncol = 0, nrow = 993))
  indep_results <- data.frame(matrix(ncol = 0, nrow = 993))
  indep_DR_results <- data.frame(matrix(ncol = 0, nrow = 993))

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
    
    best_h = 0.1
    # 计算不同方法的权重
    formula <- as.formula(paste("t ~", paste(colnames(data_df)[2:(ncol(data_df) - 390)], collapse = " + ")))
    gps_est <- WeightIt::weightit(formula, method = "glm", data = data_df, stabilize = TRUE)
    cbps_est <- WeightIt::weightit(formula, method = "cbps", data = data_df, over = FALSE, stabilize = TRUE)
    gbm_est <- WeightIt::weightit(formula, method = "gbm", data = data_df, stabilize = TRUE, criterion = "p.mean")
    ebal_est <- WeightIt::weightit(formula, method = "ebal", data = data_df, stabilize = TRUE)
    bart_est <- WeightIt::weightit(formula, method = "bart", data = data_df, stabilize = TRUE, n.trees = 250)
    #indep_weights <- independence_weights(data_df$t, data_df %>% select(starts_with("x")))
   
    
    

    # 当样本量为2000时，绘制love.plot，并结束代码
    
      # 绘制GPS方法的love.plot
    love.plot(gps_est)
    
      # 结束代码，不再继续计算
    
  }
  
  

}
