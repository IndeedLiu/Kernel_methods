library(WeightIt)
library(data.table)
options(warn = -1)  # Suppress all warnings

simulate_data_for_policy_evaluation_2018 <- function(seed = 2, nobs = 1000, MX1 = -0.5, MX2 = 1, MX3 = 0.3, A_effect = TRUE) {
  set.seed(seed)
  
  # Generate covariates
  X1 <- runif(nobs, min = 0, max = 1)
  X2 <- runif(nobs, min = 0, max = 1)
  X3 <- runif(nobs, min = 0, max = 1)
  X4 <- runif(nobs, min = 0, max = 1)
  X5 <- runif(nobs, min = 0, max = 1)

  # Generate treatment
  A <- ((X1 + X2 + X3 + X4 + X5) + 0.1+0.5 * rnorm(nobs))/10

  # Calculate outcome based on the treatment effect flag
  if (A_effect) {
    Y <- 2 * abs((X1 + X2 + X3 + X4 + X5 )/10- A)^3 + 0.2 * rnorm(nobs)
    truth <- A^2 - 1.22 * A + 2.994
  } else {
    Y <- X1 + X1^2 + X2 + X2^2 + X1 * X2 + X5 + rnorm(nobs)
    truth <- 5.05
  }

  data <- data.frame(Y = Y, A = A, X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5, truth = truth)
  return(data)
}

# Sample size range
sample_sizes <- seq(10, 300, 10)

weights_filename <- "weights_results1.csv"
data_filename <- "simulated_data1.csv"

for (rep in 1:100) {
  batch_data <- NULL  # Reset for each batch
  for (n in sample_sizes) {
    # Generate new data
    new_data <- simulate_data_for_policy_evaluation_2018(seed = rep * n, nobs = 10)
    new_data$nobs <- n
    new_data$batch <- rep  # Use rep as the batch identifier

    # Append new data to the batch data
    batch_data <- rbind(batch_data, new_data)
    batch_data$nobs <- n
    propens_formul <- as.formula("A ~ X1 + X2 + X3 + X4 + X5")

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
