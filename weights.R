library(WeightIt)
library(data.table)
options(warn = -1)  # Suppress all warnings

simulate_policy_evaluation_data <- function(seed = 2,
                                            nobs = 1000,
                                            MX1 = -0.5,
                                            MX2 = 1,
                                            MX3 = 0.3,
                                            A_effect = TRUE) {
  set.seed(seed)
  
  # Generate covariates and dose
  X1 <- rnorm(nobs, mean = MX1, sd = 1)
  X2 <- rnorm(nobs, mean = MX2, sd = 1)
  X3 <- rnorm(nobs, mean = 0, sd = 1)
  X4 <- rnorm(nobs, mean = MX2, sd = 1)
  X5 <- rbinom(nobs, 1, prob = MX3)
  
  Z1 <- exp(X1 / 2)
  Z2 <- (X2 / (1 + exp(X1))) + 10
  Z3 <- (X1 * X3 / 25) + 0.6
  Z4 <- (X4 - MX2)^2
  Z5 <- X5
  
  muA <- 5 * X1 + 6 * X2 + 3 * X5 + X4
  
  A <- (rnorm(nobs, sd = 2) + muA)
  
  if (A_effect) {
    Cnum <- 1161.25
    Y <- -0.15 * A^2 + A * (X1^2 + X2^2) - 15 + (X1 + 3)^2 + 2 * (X2 - 25)^2 + X3 - Cnum + rnorm(nobs)
    Y <- Y /50
    truth <- -0.15 * A^2 + A * 3.25-0.3
    truth <- truth 
  } else {
    Y <- X1 + X1^2 + X2 + X2^2 + X1 * X2 + X5 + rnorm(nobs)
    truth <- 5.05
  }
  
  data <- data.frame('Y' = Y, 'A' = A, 'Z1' = Z1, 'Z2' = Z2, 'Z3' = Z3, 'Z4' = Z4, 'Z5' = Z5, 'truth' = truth)
  return(data)
}

# 样本量范围
sample_sizes <- seq(10, 500, 10)

weights_filename <- "weights_results.csv"
data_filename <- "simulated_data.csv"

for (rep in 1:10) {  # Repeat the process 100 times
  batch_data <- NULL  # Reset for each batch
  for (n in sample_sizes) {
    # Generate new data
    new_data <- simulate_policy_evaluation_data(seed = rep * n, nobs = 10)
    new_data$nobs <- n
    new_data$batch <- rep  # Use rep as the batch identifier
    
    # Append new data to the batch data
    batch_data <- rbind(batch_data, new_data)
    batch_data$nobs <- n 
    propens_formul <- as.formula("A ~ Z1 + Z2 + Z3 + Z4 + Z5")
    
    # Calculate weights and trim
    gps_est <- weightit(propens_formul, method = "glm", data = batch_data, over = FALSE, stabilize = TRUE)
    gps_est$weights <- trim(gps_est$weights, prop = .1)  # Trim 10% of extreme weights

    cbps_est <- weightit(propens_formul, method = "cbps", data = batch_data, over = FALSE, stabilize = TRUE)
    cbps_est$weights <- trim(cbps_est$weights, prop = .1)

    ebal_est <- weightit(propens_formul, method = "ebal", data = batch_data, over = FALSE, stabilize = TRUE)
    ebal_est$weights <- trim(ebal_est$weights, prop = .1)

    bart_est <- weightit(propens_formul, method = "bart", data = batch_data, over = FALSE, stabilize = TRUE)
    bart_est$weights <- trim(bart_est$weights, prop = .1)
    
    weights_df <- data.frame(
      nobs = n,
      batch = rep,
      gps_weights = gps_est$weights,
      cbps_weights = cbps_est$weights,
      ebal_weights = ebal_est$weights,
      bart_weights = bart_est$weights
    )
    
    # Save weights and data to CSV using fwrite
    if (rep == 1 && n == 10) {
      fwrite(batch_data, data_filename, row.names = FALSE)
      fwrite(weights_df, weights_filename, row.names = FALSE)
    } else {
      fwrite(batch_data, data_filename, append = TRUE, row.names = FALSE)
      fwrite(weights_df, weights_filename, append = TRUE, row.names = FALSE)
    }
  }
}







