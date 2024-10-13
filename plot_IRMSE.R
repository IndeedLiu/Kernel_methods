library(ggplot2)
library(dplyr)

# 读取IRMSE结果数据
IRMSE_results_df <- read.csv("IRMSE_results.csv")

# 筛选前500个样本量的数据
IRMSE_filtered_df <- IRMSE_results_df %>% filter(sample_size <= 485)

# 将数据转换为长格式以便绘图
IRMSE_long_df <- IRMSE_filtered_df %>%
  pivot_longer(cols = starts_with("gps_IRMSE") | starts_with("cbps_IRMSE") | starts_with("gbm_IRMSE") | starts_with("ebal_IRMSE") | starts_with("bart_IRMSE") | starts_with("indep_IRMSE"),
               names_to = "Method",
               values_to = "IRMSE")

# 重命名方法列
IRMSE_long_df$Method <- gsub("_IRMSE", "", IRMSE_long_df$Method)

# 绘制折线图
ggplot(IRMSE_long_df, aes(x = sample_size, y = IRMSE, color = Method)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "IRMSE of Different Methods for Sample Sizes up to 500",
       x = "Sample Size",
       y = "IRMSE",
       color = "Method") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5))

# 保存图形
ggsave("IRMSE_plot.png", width = 10, height = 6)
