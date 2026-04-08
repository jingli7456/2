rm(list = ls())
# 加载所需的R包
library("tm")             # 文本挖掘包
library("topicmodels")    
library("textstem")
library("slam")  
library("textmineR")  
library("ldatuning")  
library("Matrix")        #对数似然值
library("ggplot2")        # 绘图包
library("reshape2") #合成图
library("gridExtra")
library("wordcloud2")     # 词云包
library("dplyr")
library("tidyr")
library("forecast")
library("LDAvis")
library("zoo")
library("igraph")
library("ggraph")
# 设置工作目录
setwd("/Users/yujingli/Desktop/gymnastics1")

# 定义文件夹路径
# 定义包含十个文件夹的父文件夹路径
parent_folder <- "/Users/yujingli/Desktop/体操（250312）/CNKI"

# 获取十个子文件夹的路径
sub_folders <- list.files(path = parent_folder, full.names = TRUE)

# 只选取前十个文件夹（如果存在超过十个文件夹的情况）
if (length(sub_folders) > 10) {
  sub_folders <- sub_folders[1:10]
}

# 创建一个空列表来存储每个子文件夹的语料库
corpus_list <- list()

# 遍历每个子文件夹
for (i in seq_along(sub_folders)) {
  folder <- sub_folders[i]
  tryCatch({
    # 从当前子文件夹创建语料库
    corpus <- VCorpus(DirSource(folder, encoding = "UTF-8"))
    
    # 将语料库添加到列表中
    corpus_list[[i]] <- corpus
    
    # 打印成功信息
    cat(paste("成功为文件夹", folder, "创建语料库\n"))
  }, error = function(e) {
    # 打印错误信息
    cat(paste("为文件夹", folder, "创建语料库时出错:", conditionMessage(e), "\n"))
  })
}

# 合并所有语料库为一个大语料库
final_corpus <- do.call(c, corpus_list)

corpus <- final_corpus
# 文本预处理
corpus <- tm_map(corpus, content_transformer(tolower))    # 转换为小写
corpus <- tm_map(corpus, removePunctuation)               # 去除标点符号
corpus <- tm_map(corpus, removeNumbers)                  # 去除数字

# 定义自定义停用词列表
custom_stopwords <- c(stopwords("en"),  "which","were","also","and","has","our","have",
                                        "having","self","old","caused","when","most","among","out","from","uses",
                                        "high","may","after","than","more","between","different","can",
                                        "less","should","fromed","follows","feelings","times","basis","weeks",
                                        "xia","using", "two","show","showed","seven","pabc", "mainly","meet","based",
                                        "joint","right","standard","limb","left","divide","numb","dynasty","video",
                      "three","period","fitness","change","literature","chinese","method","main","name","effect",
                      "analysis","halphysical","difficulty","processmeasure","human","think","social","purpunodel",
                      "power","give","time","game","key","history","late","china","qing","Seath","major","first",
                      "follow","field","one","low","play","find","resuset","modern","model","top","way","take",
                      "lack","mode","ltrain","early","rate","present","forward","trend","japan","difficult",
                      "explore","motion","will","movement","mean","improve","use","need","study","event",
                      "promote","good","state","military","reform","provide","selection","review","sport",
                      "development","train","rhythmic","gymnast","value","gymnastic","new","paper group",
                      "control","intervention","experimental train","group","culture","performance","people",
                      "carry","make","hip","highly","add","get","set","last","cause","still","four","short",
                      "many","effort","year","form","young","help","put","back","full","however","etc","pay",
                      "abstract","result","competitive","significantly","great","paper","significant","total")   # 去除停用词

# 去除停用词
corpus <- tm_map(corpus, removeWords, custom_stopwords)

# 词形还原 - 优化的版本
corpus <- tm_map(corpus, content_transformer(function(x) {
  # 将文本分割成单词
  words <- unlist(strsplit(x, "\\s+"))
  # 对单词进行词形还原
  lemmatized_words <- lemmatize_words(words)
  # 将还原后的单词重新组合成文本
  paste(lemmatized_words, collapse = " ")
}))

# 再次移除停用词以确保新形式的停用词也被移除
corpus <- tm_map(corpus, removeWords, custom_stopwords)  # 传入自定义停用词列表

# 可选：去除多余的空格
corpus <- tm_map(corpus, stripWhitespace)

# 创建文档-词项矩阵
dtm <- DocumentTermMatrix(corpus)

# 根据词频过滤稀疏项
dtm_filtered <- removeSparseTerms(dtm, sparse = 0.95)

# 查看处理后的词项矩阵信息
print("处理后的文档-词项矩阵信息：")
print(dtm_filtered)

# 检查并移除全零行
dtm_filtered <- dtm_filtered[row_sums(dtm_filtered) > 0, ]

# 计算词频
word_freq <- colSums(as.matrix(dtm_filtered))
word_freq <- sort(word_freq, decreasing = TRUE)

# 创建词频数据框
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)

# 将词频写入 CSV 文件
write.csv(word_freq_df, file = "word_frequency.csv", row.names = FALSE)

# 创建词云
wordcloud2(data = data.frame(word = names(word_freq), freq = word_freq))

# 计算TF-IDF
dtm_tfidf <- weightTfIdf(dtm_filtered)

# 将TF-IDF矩阵转换为数据框
tfidf_matrix <- as.matrix(dtm_tfidf)
tfidf_data <- data.frame(
  word = colnames(tfidf_matrix),
  tfidf = colSums(tfidf_matrix)
)

# 排序TF-IDF数据
tfidf_data <- tfidf_data[order(tfidf_data$tfidf, decreasing = TRUE), ]

# 移除缺失值
tfidf_data_clean <- na.omit(tfidf_data)

tfidf_data_clean <- tfidf_data %>%
  filter(!is.na(tfidf))

summary(tfidf_data$tfidf)

# 查看清理后的数据框
str(tfidf_data_clean)
summary(tfidf_data_clean$tfidf)
nrow(tfidf_data_clean)

ggplot(tfidf_data_clean[1:18, ], aes(x = reorder(word, tfidf), y = tfidf)) +
  geom_point(size = 3, color = "blue", na.rm = TRUE) +  # 忽略缺失值
  coord_flip() +
  labs(title = "Top 20 TF-IDF Words", x = "Words", y = "TF-IDF") +
  theme_minimal()

ggplot(tfidf_data_clean[1:18, ], aes(x = reorder(word, tfidf), y = tfidf)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Top 18 TF-IDF Words", x = "Words", y = "TF-IDF") +
  theme_minimal()

# 设置主题数范围
topic_range <- 2:30

# 初始化存储结果的列表
results <- data.frame(Topics = integer(),
                      LogLikelihood = numeric(),
                      AIC = numeric(),
                      BIC = numeric())

# 计算每个主题数的对数似然值、AIC和BIC
for (num_topics in topic_range) {
  # 运行LDA模型
  lda_model <- LDA(dtm_filtered, k = num_topics, control = list(seed = 1234))
  
  # 获取对数似然值
  log_likelihood <- logLik(lda_model)
  
  # 计算AIC和BIC
  aic_value <- AIC(lda_model)
  bic_value <- BIC(lda_model)
  
  # 存储结果
  results <- rbind(results, data.frame(Topics = num_topics,
                                       LogLikelihood = as.numeric(log_likelihood),
                                       AIC = aic_value,
                                       BIC = bic_value))
}

# 打印结果
print(results)

# 绘制AIC和BIC的折线图
results_long <- reshape2::melt(results, id.vars = "Topics", measure.vars = c("AIC", "BIC"))

ggplot(results_long, aes(x = Topics, y = value, color = variable)) +
  geom_line(size = 1.2) +  # 设置线条宽度
  geom_point(size = 2) +   # 设置点的大小
  labs(title = "AIC and BIC Values for Different Number of Topics",
       x = "Number of Topics",
       y = "Information Criterion Value",
       color = "Criterion") +
  theme_minimal() +  # 使用简约主题
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)) +
  scale_color_manual(values = c("AIC" = "#b0d097", "BIC" = "#c5a6c4"))  # 自定义颜色

# 绘制对数似然值的折线图
library(ggplot2)
ggplot(results, aes(x = Topics, y = LogLikelihood)) +
  geom_line(color = "green", size = 1.2) +  # 设置线条颜色和宽度
  geom_point(size = 2) +  # 设置点的大小
  labs(title = "Log-Likelihood for Different Number of Topics",
       x = "Number of Topics",
       y = "Log-Likelihood") +
  theme_minimal() +  # 使用简约主题
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))

# 1. 准备数据：将三个指标统一到长格式
results_long <- results %>%
  pivot_longer(
    cols = c(AIC, BIC, LogLikelihood),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    # 将指标类型分组：AIC/BIC为"信息准则"，LogLikelihood单独一组
    Metric_Group = case_when(
      Metric %in% c("AIC", "BIC") ~ "Information_Criteria",
      Metric == "LogLikelihood" ~ "Log_Likelihood"
    ),
    # 对指标名称进行更清晰的标签
    Metric = factor(Metric, 
                    levels = c("AIC", "BIC", "LogLikelihood"),
                    labels = c("AIC", "BIC", "Log-Likelihood"))
  )

# 2. 创建第一个图：AIC和BIC（信息准则）
p_aic_bic <- results_long %>%
  filter(Metric_Group == "Information_Criteria") %>%
  ggplot(aes(x = Topics, y = Value, color = Metric, linetype = Metric)) +
  geom_line(size = 0.8) +
  geom_point(size = 1.5, alpha = 0.7) +
  scale_color_manual(
    values = c("AIC" = "#1f77b4", "BIC" = "#ff7f0e"),  # 使用色盲友好的颜色
    name = "Information_Criteria"
  ) +
  scale_linetype_manual(
    values = c("AIC" = "solid", "BIC" = "dashed"),
    name = "Information_Criteria"
  ) +
  labs(
    title = "(a) AIC and BIC Trends Across Topic Numbers",
    x = NULL,  # 共用x轴标签
    y = "Information Criterion Value\n(Lower is Better)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.position = c(0.85, 0.85),  # 将图例放在图内右上角
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.4, "cm"),
    legend.background = element_rect(fill = "white", color = "grey80", size = 0.2),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(size = 0.3, color = "grey90"),
    plot.margin = unit(c(5, 5, 0, 5), "pt")  # 调整边距
  )

# 3. 创建第二个图：对数似然值
p_loglik <- results_long %>%
  filter(Metric_Group == "Log_Likelihood") %>%
  ggplot(aes(x = Topics, y = Value)) +
  geom_line(color = "#2ca02c", size = 0.8) +  # 使用绿色系
  geom_point(color = "#2ca02c", size = 1.5, alpha = 0.7) +
  labs(
    title = "(b) Log-Likelihood Trend Across Topic Numbers",
    x = "Number of Topics",
    y = "Log-Likelihood Value\n(Higher is Better)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(size = 0.3, color = "grey90"),
    plot.margin = unit(c(0, 5, 5, 5), "pt")  # 调整边距
  )

# 4. 使用gridExtra将两个图垂直组合
combined_plot <- grid.arrange(
  p_aic_bic, 
  p_loglik, 
  ncol = 1, 
  heights = c(0.5, 0.5)  # 两个图高度相等
)

# 5. 保存高质量图片（可选）
# ggsave("LDA模型评估指标.png", combined_plot, 
#        width = 18, height = 15, units = "cm", dpi = 600)

# 6. 打印组合图
print(combined_plot)

# 7. 输出图片信息
cat("\n=== 图片信息 ===\n")
cat("图片格式: 组合图 (两图垂直排列)\n")
cat("图(a): AIC与BIC趋势 - 双线图，值越小越好\n")
cat("图(b): 对数似然值趋势 - 单线图，值越大越好\n")
cat("设计特点:\n")
cat("  1. 使用色盲友好配色 (Set2配色方案)\n")
cat("  2. 统一的字体大小和网格样式\n")
cat("  3. 优化的图例位置和边距\n")
cat("  4. 符合学术期刊的简洁风格\n")
cat("  5. 直接标注每个子图的标题(a)和(b)\n")

# 自动选择最佳主题数的函数
choose_best_topic <- function(results) {
  # 寻找AIC和BIC的最低点
  best_aic <- results$Topics[which.min(results$AIC)]
  best_bic <- results$Topics[which.min(results$BIC)]
  
  # 寻找对数似然值增长开始减缓的点
  # 计算对数似然值的差分
  loglik_diff <- diff(results$LogLikelihood)
  
  # 找到差分开始减小的点
  threshold <- 0.01  # 设置一个阈值，例如0.01，表示增长开始减缓
  best_loglik <- results$Topics[which(loglik_diff < threshold)[1] + 1]
  
  # 返回一个列表，包含所有可能的最佳主题数
  list(best_aic = best_aic, best_bic = best_bic, best_loglik = best_loglik)
}

# 使用函数选择最佳主题数
best_topics <- choose_best_topic(results)
print(best_topics)

# 只保留一个改进版的函数定义
choose_best_topic <- function(results) {
  # 寻找AIC和BIC的最低点
  best_aic <- results$Topics[which.min(results$AIC)]
  best_bic <- results$Topics[which.min(results$BIC)]
  
  # 寻找对数似然值增长开始减缓的点
  # 计算对数似然值的差分
  loglik_diff <- diff(results$LogLikelihood)
  
  # 找到差分开始减小的点
  threshold <- 0.01
  diff_change_points <- which(loglik_diff < threshold)
  if (length(diff_change_points) > 0) {
    best_loglik <- results$Topics[diff_change_points[1] + 1]
  } else {
    best_loglik <- results$Topics[which.max(results$LogLikelihood)]
  }
  
  # 返回一个列表
  list(best_aic = best_aic, best_bic = best_bic, best_loglik = best_loglik)
}

# 计算最佳主题数
cat("基于统计指标的最佳主题数：\n")
best_topics <- choose_best_topic(results)
print(best_topics)

# 计算主题重叠度（Jensen-Shannon散度）
if (!require("philentropy")) install.packages("philentropy")
if (!require("topicmodels")) install.packages("topicmodels")
library(philentropy)
library(topicmodels)

# 插入修正的JSD计算函数（使用自然对数log而非log2）
calculate_jsd_correct <- function(beta_matrix) {
  k <- nrow(beta_matrix)
  jsd_values <- numeric()
  
  for (i in 1:(k-1)) {
    for (j in (i+1):k) {
      p <- beta_matrix[i, ]
      q <- beta_matrix[j, ]
      
      # 确保是有效的概率分布
      p <- p / sum(p)
      q <- q / sum(q)
      
      # 处理log(0)的情况
      epsilon <- 1e-10
      p <- p + epsilon
      q <- q + epsilon
      p <- p / sum(p)
      q <- q / sum(q)
      
      m <- 0.5 * (p + q)
      
      # 关键修改：使用自然对数log()而不是log2()
      kl_pm <- sum(p * log(p / m), na.rm = TRUE)
      kl_qm <- sum(q * log(q / m), na.rm = TRUE)
      
      # 计算JSD（标准定义）
      jsd <- 0.5 * (kl_pm + kl_qm)
      jsd_values <- c(jsd_values, jsd)
    }
  }
  
  return(mean(jsd_values, na.rm = TRUE))
}

# 初始化存储JSD结果的数据框
coherence_results <- data.frame(
  Topics = integer(),
  JSD = numeric(),  # 只存储原始JSD
  stringsAsFactors = FALSE
)

if (!exists("topic_range_overlap")) {
  topic_range_overlap <- 2:30  # 与之前的topic_range保持一致
}

# 计算主题数2-30的原始JSD
cat("\n计算原始JSD值...\n")
for (num_topics in topic_range_overlap) {
  cat("计算主题数:", num_topics, "\n")
  
  # 运行LDA模型
  lda_model <- LDA(dtm_filtered, k = num_topics, control = list(seed = 1234))
  
  # 提取主题-词分布矩阵
  beta_matrix <- exp(lda_model@beta)
  
  # 使用修正函数计算JSD
  mean_jsd <- calculate_jsd_correct(beta_matrix)
  
  # 存储结果（只存储原始JSD）
  coherence_results <- rbind(coherence_results, 
                             data.frame(Topics = num_topics, 
                                        JSD = mean_jsd))
  
  cat(sprintf("  结果: 平均JSD = %.4f\n", mean_jsd))
}

# 所有主题数计算完成后，再计算标准化JSD和重叠度分数
cat("\n=== 计算标准化JSD和重叠度分数 ===\n")

# 计算标准化JSD（归一化到[0,1]区间）
coherence_results$JSD_normalized <- (coherence_results$JSD - min(coherence_results$JSD)) / 
  (max(coherence_results$JSD) - min(coherence_results$JSD))

# 计算重叠度分数：使用 1 - normalized_JSD
coherence_results$Overlap_Score <- 1 - coherence_results$JSD_normalized

cat("\n标准化后的JSD和重叠度分数:\n")
print(head(coherence_results, 10))

# 处理Inf或NA值
if (any(is.infinite(coherence_results$JSD)) || any(is.na(coherence_results$JSD))) {
  cat("\n警告: 发现Inf或NA值，正在处理...\n")
  
  for (i in 1:nrow(coherence_results)) {
    if (is.infinite(coherence_results$JSD[i]) || is.na(coherence_results$JSD[i])) {
      # 找到相邻的有效值
      neighbors <- c()
      if (i > 1 && !is.infinite(coherence_results$JSD[i-1]) && !is.na(coherence_results$JSD[i-1])) {
        neighbors <- c(neighbors, coherence_results$JSD[i-1])
      }
      if (i < nrow(coherence_results) && !is.infinite(coherence_results$JSD[i+1]) && !is.na(coherence_results$JSD[i+1])) {
        neighbors <- c(neighbors, coherence_results$JSD[i+1])
      }
      
      if (length(neighbors) > 0) {
        coherence_results$JSD[i] <- mean(neighbors)
        coherence_results$Overlap_Score[i] <- 1 - coherence_results$JSD_normalized[i]
        cat(sprintf("  修正主题数 %d: JSD = %.4f\n", 
                    coherence_results$Topics[i], coherence_results$JSD[i]))
      }
    }
  }
}

# 找到基于重叠度的最佳主题数
best_jsd <- coherence_results$Topics[which.max(coherence_results$JSD)]
best_overlap <- coherence_results$Topics[which.min(coherence_results$Overlap_Score)]

cat("\n=== 主题重叠度分析结果 ===\n")
cat(sprintf("基于最大JSD（最小重叠）的最佳主题数: %d (JSD = %.4f)\n", 
            best_jsd, max(coherence_results$JSD)))
cat(sprintf("基于最小重叠度分数的主题数: %d (重叠度分数 = %.4f)\n", 
            best_overlap, min(coherence_results$Overlap_Score)))

# 重新计算关键候选主题数的JSD
candidate_topics <- unique(c(best_topics$best_aic, best_topics$best_bic, 
                             best_topics$best_loglik, best_jsd))
cat("\n=== 重新计算候选主题数的JSD ===\n")
cat("候选主题数:", paste(candidate_topics, collapse = ", "), "\n")

# 可视化JSD趋势
cat("\n=== 绘制JSD趋势图 ===\n")

# 创建图形设备
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))

# 1. JSD随主题数变化的趋势
plot(coherence_results$Topics, coherence_results$JSD, 
     type = "b", pch = 19, col = "blue", lwd = 2,
     xlab = "Number of Topics", ylab = "JSD (Jensen-Shannon Divergence)",
     main = "Inter-topic Distinction Analysis (JSD)",
     ylim = c(0, 1))
abline(v = best_jsd, col = "red", lty = 2, lwd = 1.5)
text(best_jsd, max(coherence_results$JSD) * 0.9, 
     sprintf("Optimal: %d Topics", best_jsd), 
     pos = 4, col = "red")

# 2. 重叠度分数趋势
plot(coherence_results$Topics, coherence_results$Overlap_Score, 
     type = "b", pch = 19, col = "darkgreen", lwd = 2,
     xlab = "Number of Topics", ylab = "Overlap Score",
     main = "Topic Overlap Score")
abline(v = best_overlap, col = "red", lty = 2, lwd = 1.5)
text(best_overlap, max(coherence_results$Overlap_Score) * 0.9, 
     sprintf("Minimum Overlap: %d Topics", best_overlap), 
     pos = 4, col = "red")

# 3. 候选主题数对比
candidate_data <- coherence_results[coherence_results$Topics %in% candidate_topics, ]
plot(candidate_data$Topics, candidate_data$JSD, 
     type = "h", lwd = 10, col = "steelblue",
     xlab = "Candidate Topic Numbers", ylab = "JSD Value",
     main = "JSD Comparison of Candidate Topic Numbers",
     xlim = c(min(candidate_topics)-2, max(candidate_topics)+2))
text(candidate_data$Topics, candidate_data$JSD, 
     round(candidate_data$JSD, 3), pos = 3, cex = 0.8)
points(candidate_data$Topics, candidate_data$JSD, pch = 19, col = "red", cex = 1.2)

# 4. 综合指标比较
# 标准化处理
coherence_results$JSD_normalized <- (coherence_results$JSD - min(coherence_results$JSD)) / 
  (max(coherence_results$JSD) - min(coherence_results$JSD))
coherence_results$Overlap_normalized <- (coherence_results$Overlap_Score - min(coherence_results$Overlap_Score)) / 
  (max(coherence_results$Overlap_Score) - min(coherence_results$Overlap_Score))

plot(coherence_results$Topics, coherence_results$JSD_normalized, 
     type = "l", col = "blue", lwd = 2,
     xlab = "Number of Topics", ylab = "Normalized Values",
     main = "Comparative Analysis of Normalized Metrics",
     ylim = c(0, 1))
lines(coherence_results$Topics, 1 - coherence_results$Overlap_normalized, 
      col = "darkgreen", lwd = 2)
legend("topright", legend = c("JSD (Higher is Better)", "1 - Overlap Score (Higher is Better)"), 
       col = c("blue", "darkgreen"), lwd = 2, bty = "n", cex = 0.8)

# 添加图例说明
cat("已生成4个子图:\n")
cat("1. JSD趋势图 - 值越大表示主题区分度越好\n")
cat("2. 重叠度趋势图 - 值越小表示主题重叠度越低\n")
cat("3. 候选主题数对比 - 显示各候选主题数的具体JSD值\n")
cat("4. 综合比较 - 同时显示两个指标的走势\n")

# 返回最佳主题数建议
cat("\n=== 最终主题数选择建议 ===\n")
cat("1. 基于主题区分度 (JSD最大化):", best_jsd, "个主题\n")
cat("2. 基于主题重叠度 (重叠度最小化):", best_overlap, "个主题\n")
cat("3. 模型拟合指标候选:", paste(setdiff(candidate_topics, c(best_jsd, best_overlap)), collapse = ", "), "个主题\n")
cat("\n推荐使用", best_jsd, "个主题，因为JSD值最高(", round(max(coherence_results$JSD), 4), 
    ")，主题区分度最好。\n")

library(ggplot2)

p1 <- ggplot(coherence_results, aes(x = Topics, y = JSD)) +
  geom_line(color = "blue", linewidth = 1.2) +
  geom_point(size = 2, color = "red") +
  geom_vline(xintercept = candidate_topics, linetype = "dashed", color = "gray", alpha = 0.5) +
  labs(title = "JSD值随主题数变化趋势",
       x = "主题数", y = "JSD值（越大越好）") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))

p2 <- ggplot(coherence_results, aes(x = Topics, y = Overlap_Score)) +
  geom_line(color = "green", linewidth = 1.2) +
  geom_point(size = 2, color = "orange") +
  labs(title = "主题重叠度分数趋势",
       x = "主题数", y = "重叠度分数（越小越好）") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))

# 显示图形
if (require("gridExtra", quietly = TRUE)) {
  library(gridExtra)
  grid.arrange(p1, p2, ncol = 2)
} else {
  install.packages("gridExtra")
  library(gridExtra)
  grid.arrange(p1, p2, ncol = 2)
}

# 综合所有指标的分析
cat("\n=== 综合所有指标的分析 ===\n")

# 收集所有推荐
all_recommendations <- c(
  AIC = best_topics$best_aic,
  BIC = best_topics$best_bic,
  对数似然差分 = best_topics$best_loglik,
  JSD = best_jsd
)

cat("各指标推荐的主题数:\n")
for (i in 1:length(all_recommendations)) {
  cat(sprintf("  %s: %d\n", names(all_recommendations)[i], all_recommendations[i]))
}

# 熵权法选择最佳主题数
cat("\n=== 熵权法选择最佳主题数 ===\n")

# 检查必要的数据框是否存在
if (!exists("results")) {
  stop("错误: results 数据框不存在")
}
if (!exists("coherence_results")) {
  stop("错误: coherence_results 数据框不存在")
}

# 合并数据
merged_data <- merge(results, coherence_results, by = "Topics")

# 构建指标矩阵
indicators_matrix <- cbind(
  merged_data$AIC,
  merged_data$BIC,
  merged_data$LogLikelihood,
  merged_data$JSD
)

colnames(indicators_matrix) <- c("AIC", "BIC", "LogLik", "JSD")
rownames(indicators_matrix) <- merged_data$Topics

cat("原始指标矩阵（前5行）:\n")
print(head(indicators_matrix, 5))

# 数据标准化
normalize_indicators <- function(ind_matrix) {
  n_indicators <- ncol(ind_matrix)
  n_topics <- nrow(ind_matrix)
  norm_matrix <- matrix(0, nrow = n_topics, ncol = n_indicators)
  
  for (j in 1:n_indicators) {
    if (j <= 2) {
      # AIC和BIC：越小越好，使用负向标准化
      min_val <- min(ind_matrix[, j])
      max_val <- max(ind_matrix[, j])
      # 使用更稳健的浮点数相等判断
      if (max_val - min_val < .Machine$double.eps^0.5) { 
        norm_matrix[, j] <- 0  # 所有值相等时设为0，表示该指标无区分能力
      } else {
        norm_matrix[, j] <- (max_val - ind_matrix[, j]) / (max_val - min_val)
      }
    } else {
      # LogLik和JSD：越大越好，使用正向标准化
      min_val <- min(ind_matrix[, j])
      max_val <- max(ind_matrix[, j])
      # 使用更稳健的浮点数相等判断
      if (max_val - min_val < .Machine$double.eps^0.5) { 
        norm_matrix[, j] <- 0  # 所有值相等时设为0，表示该指标无区分能力
      } else {
        norm_matrix[, j] <- (ind_matrix[, j] - min_val) / (max_val - min_val)
      }
    }
  }
  
  colnames(norm_matrix) <- colnames(ind_matrix)
  rownames(norm_matrix) <- rownames(ind_matrix)
  return(norm_matrix)
}
norm_indicators <- normalize_indicators(indicators_matrix)
cat("\n标准化后的指标矩阵（前5行）:\n")
print(head(norm_indicators, 5))

# 计算比重矩阵
p_matrix <- norm_indicators
for (j in 1:ncol(p_matrix)) {
  col_sum <- sum(p_matrix[, j], na.rm = TRUE)
  if (col_sum > 0) {
    p_matrix[, j] <- p_matrix[, j] / col_sum
  }
}

# 计算熵值
k <- 1 / log(nrow(p_matrix))
e_j <- numeric(ncol(p_matrix))

for (j in 1:ncol(p_matrix)) {
  col <- p_matrix[, j]
  col <- col[col > 0]
  if (length(col) > 0) {
    e_j[j] <- -k * sum(col * log(col), na.rm = TRUE)
  } else {
    e_j[j] <- 0
  }
}

# 计算差异系数
d_j <- 1 - e_j

# 计算权重
if (sum(d_j) > 0) {
  weights <- d_j / sum(d_j)
} else {
  weights <- rep(1/length(d_j), length(d_j))
}

# 创建权重数据框
weights_df <- data.frame(
  指标 = c("AIC", "BIC", "对数似然", "JSD"),
  熵值 = round(e_j, 4),
  差异系数 = round(d_j, 4),
  权重 = round(weights, 4)
)

cat("\n熵权法计算的权重:\n")
print(weights_df)

# 计算综合得分
scores <- norm_indicators %*% weights

results_entropy <- data.frame(
  主题数 = as.numeric(rownames(norm_indicators)),
  熵权法综合得分 = scores
)

# 按综合得分排序
results_entropy <- results_entropy[order(-results_entropy$熵权法综合得分), ]

cat("\n各主题数的熵权法综合得分（前10名）:\n")
print(head(results_entropy, 10))

# 找到综合得分最高的主题数
best_entropy <- results_entropy$主题数[which.max(results_entropy$熵权法综合得分)]
cat(sprintf("\n熵权法推荐的最佳主题数: %d\n", best_entropy))
cat(sprintf("综合得分: %.4f\n", max(results_entropy$熵权法综合得分)))

# 可视化熵权法结果
cat("\n=== 熵权法结果可视化 ===\n")

p3 <- ggplot(results_entropy, aes(x = 主题数, y = 熵权法综合得分)) +
  geom_line(color = "purple", linewidth = 1.2) +
  geom_point(size = 2, color = "red") +
  geom_vline(xintercept = best_entropy, linetype = "dashed", 
             color = "blue", linewidth = 1) +
  annotate("text", x = best_entropy, y = max(results_entropy$熵权法综合得分),
           label = sprintf("最佳: %d", best_entropy), 
           hjust = -0.2, vjust = 1, color = "blue", size = 4) +
  labs(title = "熵权法综合得分随主题数变化",
       x = "主题数", y = "熵权法综合得分") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))

print(p3)

# 比较所有方法的结果
cat("\n=== 所有方法推荐结果比较 ===\n")

# 添加熵权法推荐
all_recommendations <- c(all_recommendations, 熵权法 = best_entropy)

cat("\n各方法推荐的主题数:\n")
for (i in 1:length(all_recommendations)) {
  cat(sprintf("  %s: %d\n", names(all_recommendations)[i], all_recommendations[i]))
}

# 计算中位数
median_choice <- median(all_recommendations)
cat(sprintf("\n所有推荐值的中位数: %.1f\n", median_choice))
cat(sprintf("四舍五入: %.0f\n", round(median_choice)))

# 计算加权平均值
cat("\n=== 加权平均推荐 ===\n")
final_weights <- c(0.2, 0.1, 0.1, 0.2, 0.4)
weighted_avg <- sum(all_recommendations * final_weights)
cat(sprintf("加权平均值: %.2f\n", weighted_avg))
cat(sprintf("四舍五入: %.0f\n", round(weighted_avg)))

# 最终推荐
cat("\n=== 最终推荐 ===\n")
cat("基于熵权法的分析，推荐主题数:", best_entropy, "\n")
cat("理由: 熵权法综合了AIC、BIC、对数似然和JSD四个指标，\n")
cat("      考虑了各指标的信息量和变异程度，是最客观的综合评价方法。\n\n")

# 查看熵权法推荐的主题
cat("=== 查看熵权法推荐的主题关键词 ===\n")
lda_model_entropy <- LDA(dtm_filtered, k = best_entropy, control = list(seed = 1234))
terms_entropy <- terms(lda_model_entropy, 10)

cat(sprintf("\n熵权法推荐的主题数: %d\n", best_entropy))
for (i in 1:best_entropy) {
  cat(sprintf("主题 %d: %s\n", i, paste(terms_entropy[, i], collapse = ", ")))
}

# 检查熵权法推荐与其他方法的差异
cat("\n=== 备选方案检查 ===\n")

# 问题1: 变量名错误 - 应为all_recommendations
other_recommendations <- all_recommendations[names(all_recommendations) != "熵权法"]
cat("其他方法推荐: ", paste(other_recommendations, collapse = ", "), "\n")
cat("其他方法中位数: ", median(other_recommendations), "\n")

# 检查差异是否大于5
diff_value <- abs(best_entropy - median(other_recommendations))
cat(sprintf("差异: %.1f\n", diff_value))

if (diff_value > 5) {
  cat("注意: 熵权法推荐与其他方法差异较大，建议同时考虑以下备选:\n")
  
  # 找到综合得分前3的主题数
  # 问题2: 确保results_entropy已定义
  if (exists("results_entropy")) {
    top3 <- head(results_entropy$主题数, 3)
  } else {
    # 如果不存在，创建一个简单的备用方案
    all_topics <- unique(c(best_topics$best_aic, best_topics$best_bic, best_topics$best_loglik, best_jsd))
    top3 <- head(sort(all_topics), 3)
  }
  cat("熵权法综合得分前三的主题数:", paste(top3, collapse = ", "), "\n")
  
  # 计算备选主题数的加权得分
  alternative_scores <- data.frame(
    主题数 = numeric(),
    推荐理由 = character(),
    综合得分 = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (k in top3) {
    reasons <- c()
    if (exists("best_topics")) {
      if (k == best_topics$best_aic) reasons <- c(reasons, "AIC最小")
      if (k == best_topics$best_bic) reasons <- c(reasons, "BIC最小")
      if (k == best_topics$best_loglik) reasons <- c(reasons, "对数似然差分推荐")
    }
    if (exists("best_jsd") && k == best_jsd) reasons <- c(reasons, "JSD最大")
    
    if (length(reasons) == 0) reasons <- "熵权法推荐"
    
    # 获取综合得分
    if (exists("results_entropy") && k %in% results_entropy$主题数) {
      score <- results_entropy$熵权法综合得分[results_entropy$主题数 == k]
    } else {
      score <- NA
    }
    
    alternative_scores <- rbind(alternative_scores,
                                data.frame(
                                  主题数 = k,
                                  推荐理由 = paste(reasons, collapse = ", "),
                                  综合得分 = score
                                ))
  }
  
  cat("\n备选方案详细比较:\n")
  print(alternative_scores)
} else {
  cat("熵权法推荐与其他方法差异不大，可放心采用熵权法推荐。\n")
}

# 查找特定主题数的熵权法综合得分
specific_topics <- c(3, 5, 22, 29)

cat("\n=== 特定主题数的熵权法综合得分 ===\n")

# 方法1: 直接从结果中提取
if (exists("results_entropy")) {
  for (topic in specific_topics) {
    if (topic %in% results_entropy$主题数) {
      score <- results_entropy$熵权法综合得分[results_entropy$主题数 == topic]
      cat(sprintf("主题数 %d: 熵权法综合得分 = %.4f\n", topic, score))
    } else {
      cat(sprintf("主题数 %d: 在结果中未找到\n", topic))
    }
  }
} else {
  cat("警告: results_entropy 数据框不存在\n")
}

# 使用熵权法推荐
if (exists("best_entropy")) {
  # 使用熵权法推荐
  best_topic_number <- best_entropy
  method_used <- "熵权法"
  cat("使用熵权法推荐的主题数\n")
} else {
  # 如果熵权法不存在，回退到中位数
  if (exists("best_topics")) {
    all_values <- c(best_topics$best_aic, best_topics$best_bic, best_topics$best_loglik)
    if (exists("best_jsd")) all_values <- c(all_values, best_jsd)
    best_topic_number <- median(all_values)
  } else {
    best_topic_number <- 10  # 默认值
  }
  method_used <- "中位数法"
  cat("警告: 熵权法结果不存在，使用中位数方法\n")
}

# 打印最终选择的最佳主题数
cat(sprintf("\n最终选择的最佳主题数是: %d (基于%s)\n", best_topic_number, method_used))

# 设置随机种子确保结果可重复
set.seed(1234)  # 保持与LDA控制参数中的种子一致

# 使用最佳主题数训练LDA模型
cat(sprintf("\n正在使用最佳主题数 %d 训练LDA模型...\n", best_topic_number))
lda_model <- LDA(dtm_filtered, k = best_topic_number, control = list(seed = 1234))

cat("LDA模型训练完成\n")

# 直接打印模型对象获取完整信息
cat("\n=== LDA模型完整信息 ===\n")
str(lda_model, max.level = 1)  # 只显示第一级结构

# 或者使用模型摘要
cat("\n=== LDA模型摘要 ===\n")
print(lda_model)

# ========== 第一部分：基础结果提取 ==========
cat("\n=== LDA模型基础信息 ===\n")
cat(sprintf("模型设置主题数: %d\n", lda_model@k))

# 修正1: 删除错误的迭代次数行
# cat(sprintf("迭代次数: %d\n", lda_model@iterations))  # 这行会导致错误

# 正确获取控制参数
if (!is.null(lda_model@control)) {
  cat("模型控制参数:\n")
  control_params <- lda_model@control
  for (param_name in names(control_params)) {
    cat(sprintf("  %s: %s\n", param_name, as.character(control_params[[param_name]])))
  }
}

# 1. 提取主题关键词
terms <- terms(lda_model, 10)
cat("\n=== 各主题前10个关键词 ===\n")
for (i in 1:ncol(terms)) {
  cat(sprintf("\n主题 %d 的前10个关键词:\n", i))
  cat(paste(terms[, i], collapse = ", "))
  cat("\n")
}

# 2. 获取文档-主题分布
topic_distributions <- lda_model@gamma
cat(sprintf("\n文档-主题分布矩阵维度: %d 个文档 × %d 个主题\n", 
            nrow(topic_distributions), ncol(topic_distributions)))

# 3. 统计文档数量
doc_topics <- topics(lda_model, 1)
cat("\n=== 每个主题的文档数量统计 ===\n")
topic_counts <- table(doc_topics)
for (i in 1:lda_model@k) {
  count <- ifelse(i %in% names(topic_counts), topic_counts[as.character(i)], 0)
  percentage <- round((count / length(doc_topics)) * 100, 1)
  cat(sprintf("主题 %d: %d 个文档 (%.1f%%)\n", i, count, percentage))
}

# 直接打印模型对象获取完整信息
cat("\n=== LDA模型完整信息 ===\n")
str(lda_model, max.level = 1)  # 只显示第一级结构

# 或者使用模型摘要
cat("\n=== LDA模型摘要 ===\n")
print(lda_model)

# ========== 第一部分：基础结果提取 ==========
cat("\n=== LDA模型基础信息 ===\n")
cat(sprintf("模型设置主题数: %d\n", lda_model@k))

# 正确获取控制参数
if (!is.null(lda_model@control)) {
  cat("模型控制参数:\n")
  control_params <- lda_model@control
  for (param_name in names(control_params)) {
    cat(sprintf("  %s: %s\n", param_name, as.character(control_params[[param_name]])))
  }
}

# 1. 提取主题关键词
terms <- terms(lda_model, 10)
cat("\n=== 各主题前10个关键词 ===\n")
for (i in 1:ncol(terms)) {
  cat(sprintf("\n主题 %d 的前10个关键词:\n", i))
  cat(paste(terms[, i], collapse = ", "))
  cat("\n")
}

# 2. 获取文档-主题分布
topic_distributions <- lda_model@gamma
cat(sprintf("\n文档-主题分布矩阵维度: %d 个文档 × %d 个主题\n", 
            nrow(topic_distributions), ncol(topic_distributions)))

# 3. 统计文档数量
doc_topics <- topics(lda_model, 1)
cat("\n=== 每个主题的文档数量统计 ===\n")
topic_counts <- table(doc_topics)
for (i in 1:lda_model@k) {
  count <- ifelse(i %in% names(topic_counts), topic_counts[as.character(i)], 0)
  percentage <- round((count / length(doc_topics)) * 100, 1)
  cat(sprintf("主题 %d: %d 个文档 (%.1f%%)\n", i, count, percentage))
}

# ========== 计算主题强度 ==========
cat("\n=== 计算主题强度 ===\n")

# 获取beta矩阵（主题-词分布的对数概率）
beta_matrix <- lda_model@beta

# 方法1: 基于前N个高概率词的概率和
calculate_topic_strength <- function(beta_matrix, N = 10) {
  # 将对数概率转换为概率
  topic_probs <- exp(beta_matrix)
  
  # 计算每个主题的强度（前N个高概率词的概率和）
  topic_strength <- apply(topic_probs, 1, function(row) {
    # 获取概率最高的前N个词
    top_probs <- sort(row, decreasing = TRUE)[1:min(N, length(row))]
    # 计算这些词的概率和
    sum(top_probs)
  })
  
  return(topic_strength)
}

# 计算主题强度
N_words <- 10
topic_strength <- calculate_topic_strength(beta_matrix, N_words)

# 创建结果数据框
strength_results <- data.frame(
  主题 = 1:length(topic_strength),
  强度_值 = round(topic_strength, 4)
)

# 计算强度排名
strength_results$强度_排名 <- rank(-strength_results$强度_值)  # 值越大排名越靠前

# 计算相对强度（标准化到0-1范围）
strength_results$相对强度 <- round(
  (strength_results$强度_值 - min(strength_results$强度_值)) / 
    (max(strength_results$强度_值) - min(strength_results$强度_值)), 
  4
)

cat(sprintf("\n主题强度计算结果（基于前%d个高概率词的概率和）：\n", N_words))
print(strength_results)

# 可视化主题强度
if (require(ggplot2)) {
  cat("\n正在生成主题强度可视化图表...\n")
  
  p <- ggplot(strength_results, aes(x = reorder(factor(主题), 强度_值), y = 强度_值)) +
    geom_bar(stat = "identity", fill = "#2E8B57", alpha = 0.8) +
    geom_hline(yintercept = mean(strength_results$强度_值), 
               linetype = "dashed", color = "#333333", alpha = 0.7) +
    geom_text(aes(label = sprintf("%.4f", 强度_值)), vjust = -0.5, size = 3.2) +
    geom_text(aes(y = 强度_值 + 0.01, label = paste0("第", 强度_排名, "名")), 
              vjust = -2, size = 2.8, color = "#666666") +
    labs(title = "主题强度分布",
         subtitle = sprintf("基于前%d个高概率词的概率和", N_words),
         x = "主题", 
         y = "强度值") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, color = "#666666"))
  
  print(p)
} else {
  cat("警告: 未安装ggplot2包，跳过可视化。\n")
}

# 输出强度统计摘要
cat("\n=== 主题强度统计摘要 ===\n")
cat(sprintf("平均主题强度: %.4f\n", mean(strength_results$强度_值)))
cat(sprintf("最强主题: 主题 %d (强度: %.4f, 排名: 第1)\n", 
            strength_results$主题[which.max(strength_results$强度_值)],
            max(strength_results$强度_值)))
cat(sprintf("最弱主题: 主题 %d (强度: %.4f, 排名: 第%d)\n", 
            strength_results$主题[which.min(strength_results$强度_值)],
            min(strength_results$强度_值),
            max(strength_results$强度_排名)))
cat(sprintf("强度标准差: %.4f\n", sd(strength_results$强度_值)))
cat(sprintf("强度变异系数: %.2f%%\n", (sd(strength_results$强度_值)/mean(strength_results$强度_值))*100))

# 保存结果到文件
write.csv(strength_results, "topic_strength_results.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("\n主题强度结果已保存到: topic_strength_results.csv\n")

# 可选：将主题强度与关键词合并展示
cat("\n=== 主题强度与关键词对应表 ===\n")
for (i in 1:nrow(strength_results)) {
  topic_id <- strength_results$主题[i]
  cat(sprintf("\n主题 %d (强度: %.4f, 排名: 第%d名):\n", 
              topic_id, strength_results$强度_值[i], strength_results$强度_排名[i]))
  cat(sprintf("  关键词: %s\n", paste(terms[, topic_id], collapse = ", ")))
}

# 计算每个主题的热度（基于文档占比）
# 确保所有主题都有计数
hotness_doc_count <- numeric(best_topic_number)
for (i in 1:best_topic_number) {
  if (as.character(i) %in% names(topic_counts)) {
    hotness_doc_count[i] <- topic_counts[as.character(i)]
  } else {
    hotness_doc_count[i] <- 0
  }
}

# 创建主题热度数据框
topic_hotness_df <- data.frame(
  Topic = 1:best_topic_number,
  Hotness = hotness_doc_count,  # 文档数量
  Percentage = hotness_doc_count / length(doc_topics) * 100  # 文档占比
)

# 计算热度的均值作为阈值
hotness_threshold <- mean(topic_hotness_df$Hotness, na.rm = TRUE)

# 打印每个主题的热度
cat("\n=== 各主题热度（文档数量） ===\n")
for (i in 1:nrow(topic_hotness_df)) {
  cat(sprintf("主题 %d: %d 个文档 (%.1f%%)\n", 
              i, 
              topic_hotness_df$Hotness[i], 
              topic_hotness_df$Percentage[i]))
}

# 标识热点主题
hot_topics <- which(topic_hotness_df$Hotness > hotness_threshold)
cat("\n=== 热点主题识别（热度高于平均值） ===\n")
for (i in hot_topics) {
  cat(sprintf("主题 %d: %d 个文档 (%.1f%%)\n", 
              i, 
              topic_hotness_df$Hotness[i], 
              topic_hotness_df$Percentage[i]))
}

# 绘制热度柱状图（与强度图相同样式）
p_hotness <- ggplot(topic_hotness_df, aes(x = factor(Topic), y = Hotness)) +
  geom_bar(stat = "identity", 
           fill = ifelse(topic_hotness_df$Hotness > hotness_threshold, "#e7d3ed", "#cee9dc"),
           width = 0.6) +
  geom_hline(yintercept = hotness_threshold, linetype = "dashed", color = "#333333") +
  labs(title = "主题热度（文档数量）柱状图",
       x = "主题", 
       y = "热度（文档数量）") +
  theme_minimal() +
  annotate("text", x = 1, y = hotness_threshold + max(topic_hotness_df$Hotness) * 0.01, 
           label = paste("平均值:", round(hotness_threshold, 2)), 
           color = "#333333", size = 3) +
  geom_text(aes(label = Hotness), vjust = -0.5) +  # 显示文档数量
  scale_x_discrete(labels = paste0("主题", 1:best_topic_number)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.title = element_text(hjust = 0.5, face = "bold"))

print(p_hotness)

# 绘制强度柱状图（英文版）
library(ggplot2)
library(gridExtra)  # 添加 gridExtra 包用于并排显示图表

p_intensity <- ggplot(topic_intensity_df, aes(x = factor(Topic), y = Intensity)) +
  geom_bar(stat = "identity", 
           fill = ifelse(topic_intensity_df$Intensity > threshold_intensity, "#e7d3ed", "#cee9dc"),
           width = 0.6) +
  geom_hline(yintercept = threshold_intensity, linetype = "dashed", color = "#333333") +
  labs(title = "Topic Strength (Cohesion) Bar Chart",
       x = "Topic", 
       y = "Strength (Cohesion)") +
  theme_minimal() +
  annotate("text", x = 1, y = threshold_intensity + 0.05, 
           label = paste("Mean:", round(threshold_intensity, 2)), 
           color = "#333333", size = 3) +
  geom_text(aes(label = round(Intensity, 2)), vjust = -0.5) +
  scale_x_discrete(labels = paste0("Topic ", 1:nrow(beta))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 10))

# 绘制热度柱状图（英文版）
p_hotness <- ggplot(topic_hotness_df, aes(x = factor(Topic), y = Hotness)) +
  geom_bar(stat = "identity", 
           fill = ifelse(topic_hotness_df$Hotness > hotness_threshold, "#e7d3ed", "#cee9dc"),
           width = 0.6) +
  geom_hline(yintercept = hotness_threshold, linetype = "dashed", color = "#333333") +
  labs(title = "Topic Hotness (Document Count) Bar Chart",
       x = "Topic", 
       y = "Hotness (Document Count)") +
  theme_minimal() +
  annotate("text", x = 1, y = hotness_threshold + max(topic_hotness_df$Hotness) * 0.01, 
           label = paste("Mean:", round(hotness_threshold, 2)), 
           color = "#333333", size = 3) +
  geom_text(aes(label = Hotness), vjust = -0.5) +
  scale_x_discrete(labels = paste0("Topic ", 1:best_topic_number)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 10))

# 并排显示两个图表
combined_plot <- grid.arrange(p_intensity, p_hotness, ncol = 2, 
                              top = "Topic Analysis: Strength vs Hotness")
print(combined_plot)


# 创建二维分析数据框
topic_analysis_df <- data.frame(
  Topic = 1:best_topic_number,
  Strength = topic_intensity_df$Intensity,
  Hotness = topic_hotness_df$Hotness,
  Percentage = topic_hotness_df$Percentage
)

# 计算平均值作为象限划分的阈值
strength_mean <- mean(topic_analysis_df$Strength, na.rm = TRUE)
hotness_mean <- mean(topic_analysis_df$Hotness, na.rm = TRUE)

# 添加象限分类
topic_analysis_df$Quadrant <- factor(
  ifelse(topic_analysis_df$Strength >= strength_mean & 
           topic_analysis_df$Hotness >= hotness_mean, "I",
         ifelse(topic_analysis_df$Strength >= strength_mean & 
                  topic_analysis_df$Hotness < hotness_mean, "II",
                ifelse(topic_analysis_df$Strength < strength_mean & 
                         topic_analysis_df$Hotness < hotness_mean, "III", "IV"))),
  levels = c("I", "II", "III", "IV")
)

# 添加象限标签描述
topic_analysis_df$Quadrant_Label <- factor(
  topic_analysis_df$Quadrant,
  levels = c("I", "II", "III", "IV"),
  labels = c("核心主题\n(高强度, 高热度)", 
             "专业主题\n(高强度, 低热度)",
             "边缘主题\n(低强度, 低热度)",
             "泛化主题\n(低强度, 高热度)")
)

# 定义象限颜色（使用RColorBrewer的Set1配色）
library(RColorBrewer)
quadrant_colors <- c("#FF6B6B", "#4ECDC4", "#FFD166", "#06AED5")
names(quadrant_colors) <- c("I", "II", "III", "IV")

# 绘制二维象限分析图
library(ggplot2)
library(ggrepel)  # 用于避免文本重叠

p_quadrant <- ggplot(topic_analysis_df, aes(x = Hotness, y = Strength, 
                                            color = Quadrant, label = paste0("T", Topic))) +
  # 绘制象限分界线
  geom_vline(xintercept = hotness_mean, linetype = "dashed", color = "gray50", alpha = 0.7) +
  geom_hline(yintercept = strength_mean, linetype = "dashed", color = "gray50", alpha = 0.7) +
  
  # 绘制散点
  geom_point(size = 8, alpha = 0.8) +
  
  # 添加主题编号标签（使用ggrepel避免重叠）
  geom_text_repel(size = 4, fontface = "bold", color = "black", 
                  box.padding = 0.5, point.padding = 0.5) +
  
  # 添加象限区域标注
  annotate("text", x = hotness_mean*1.8, y = strength_mean*1.8, 
           label = "Quadrant I\nCore Topics", size = 4, color = quadrant_colors["I"], 
           fontface = "bold", hjust = 0.5, vjust = 0.5) +
  annotate("text", x = hotness_mean*0.5, y = strength_mean*1.8, 
           label = "Quadrant II\nSpecialized Topics", size = 4, color = quadrant_colors["II"], 
           fontface = "bold", hjust = 0.5, vjust = 0.5) +
  annotate("text", x = hotness_mean*0.5, y = strength_mean*0.5, 
           label = "Quadrant III\nMarginal Topics", size = 4, color = quadrant_colors["III"], 
           fontface = "bold", hjust = 0.5, vjust = 0.5) +
  annotate("text", x = hotness_mean*1.8, y = strength_mean*0.5, 
           label = "Quadrant IV\nGeneralized Topics", size = 4, color = quadrant_colors["IV"], 
           fontface = "bold", hjust = 0.5, vjust = 0.5) +
  
  # 设置颜色
  scale_color_manual(
    name = "Quadrant Classification",
    values = quadrant_colors,
    labels = c("Quadrant I: Core Topics\n(High Strength, High Hotness)",
               "Quadrant II: Specialized Topics\n(High Strength, Low Hotness)",
               "Quadrant III: Marginal Topics\n(Low Strength, Low Hotness)",
               "Quadrant IV: Generalized Topics\n(Low Strength, High Hotness)")
  ) +
  
  # 坐标轴和标题设置
  labs(
    title = "Topic Analysis Quadrant Plot: Strength vs Hotness",
    subtitle = paste0("X-axis: Hotness (Document Count, Mean=", round(hotness_mean, 1), 
                      "), Y-axis: Strength (Cohesion, Mean=", round(strength_mean, 2), ")"),
    x = "Topic Hotness (Document Count)",
    y = "Topic Strength (Cohesion)",
    caption = "Note: T1, T2, ... represent topic numbers; dashed lines represent mean values"
  ) +
  
  # 主题美化
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, margin = margin(b = 10)),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", margin = margin(b = 15)),
    plot.caption = element_text(color = "gray50", size = 9, margin = margin(t = 10)),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  
  # 扩展坐标轴范围，为象限标签留出空间
  scale_x_continuous(expand = expansion(mult = c(0.15, 0.25))) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.25)))

print(p_quadrant)

# 输出详细的象限分析结果
cat("\n=== 主题二维象限分析结果 ===\n")
cat(paste0("强度平均值: ", round(strength_mean, 4), "\n"))
cat(paste0("热度平均值: ", round(hotness_mean, 1), "\n\n"))

for (quad in c("I", "II", "III", "IV")) {
  topics_in_quad <- topic_analysis_df$Topic[topic_analysis_df$Quadrant == quad]
  if (length(topics_in_quad) > 0) {
    quad_label <- as.character(unique(topic_analysis_df$Quadrant_Label[topic_analysis_df$Quadrant == quad]))
    cat(paste0(quad_label, " (象限 ", quad, "):\n"))
    for (t in topics_in_quad) {
      idx <- which(topic_analysis_df$Topic == t)
      cat(sprintf("  主题 %d: 强度=%.4f, 热度=%d (%.1f%%)\n", 
                  t, topic_analysis_df$Strength[idx], 
                  topic_analysis_df$Hotness[idx], topic_analysis_df$Percentage[idx]))
    }
    cat("\n")
  }
}

# 创建分析报告摘要
cat("=== 主题分析报告摘要 ===\n")
cat(sprintf("分析主题总数: %d\n", nrow(topic_analysis_df)))
cat(sprintf("文档总数: %d\n", length(doc_topics)))
cat("\n各象限主题分布:\n")
quadrant_summary <- table(topic_analysis_df$Quadrant_Label)
for (i in 1:length(quadrant_summary)) {
  cat(sprintf("  %s: %d 个主题 (%.1f%%)\n", 
              names(quadrant_summary)[i], 
              quadrant_summary[i], 
              100 * quadrant_summary[i] / nrow(topic_analysis_df)))
}

# 保存分析结果
write.csv(topic_analysis_df, "topic_quadrant_analysis.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("\n象限分析结果已保存到: topic_quadrant_analysis.csv\n")

# 可选：绘制四个象限的详细分布图（子图）
library(patchwork)

# 为每个象限创建单独的子图
plots <- list()
for (quad in c("I", "II", "III", "IV")) {
  quad_data <- topic_analysis_df[topic_analysis_df$Quadrant == quad, ]
  if (nrow(quad_data) > 0) {
    p <- ggplot(quad_data, aes(x = reorder(paste0("T", Topic), Strength), y = Strength)) +
      geom_bar(stat = "identity", fill = quadrant_colors[quad], alpha = 0.8) +
      labs(title = paste0("象限 ", quad, ": ", 
                          switch(quad, 
                                 "I" = "核心主题", 
                                 "II" = "专业主题", 
                                 "III" = "边缘主题", 
                                 "IV" = "泛化主题")),
           x = "主题编号", y = "强度") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 10, face = "bold", hjust = 0.5))
    plots[[quad]] <- p
  }
}

# 将四个子图组合成一个图形
if (length(plots) > 0) {
  combined_quad_plots <- wrap_plots(plots, ncol = 2) + 
    plot_annotation(title = "各象限主题强度分布", 
                    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold")))
  print(combined_quad_plots)
}












# 定义数据文件夹路径
data_dir <- "/Users/yujingli/Desktop/体操（250312）/CNKI"

# 获取所有年份文件夹
year_folders <- list.dirs(data_dir, full.names = TRUE, recursive = FALSE)

# 初始化数据框来存储文档和年份信息
documents <- data.frame(Year = integer(), Text = character(), stringsAsFactors = FALSE)

# 遍历每个年份文件夹
for (year_folder in year_folders) {
  # 提取年份
  year <- basename(year_folder)
  
  # 获取该年份文件夹中的所有文本文件
  files <- list.files(year_folder, pattern = "\\.txt$", full.names = TRUE)
  
  if (length(files) == 0) {
    cat("No text files found in folder:", year_folder, "\n")
  }
  
  # 读取每个文件并提取文本内容
  for (file in files) {
    # 提取文件名中的年份信息（假设文件名格式为 "year-docX.txt"）
    file_year <- sub("(\\d+)-.*", "\\1",  basename(file))
    
    # 读取文件内容
    text <- readLines(file, warn = FALSE)
    text <- paste(text, collapse = " ")  # 合并多行文本为一行
    
    # 添加到数据框
    documents <- rbind(documents, data.frame(Year = as.integer(file_year), Text = text))
  }
}

# 检查 documents 的行数
cat("Number of documents read:", nrow(documents), "\n")

# 确保 documents 不是空的
if (nrow(documents) == 0) {
  stop("No documents were read. Please check the file paths and file formats.")
}

# 提取每个文档的主题分布
topic_distributions <- lda_model@gamma

# 检查 topic_distributions 的行数
cat("Number of topic distributions:", nrow(topic_distributions), "\n")

# 确保行数匹配
if (nrow(documents) == nrow(topic_distributions)) {
  data_with_topics <- cbind(documents, topic_distributions)
} else {
  stop("The number of rows in documents and topic_distributions do not match.")
}

# 将主题分布添加到数据集中
data_with_topics <- cbind(documents, topic_distributions)

# 从 beta 矩阵中获取每个词在每个主题下的权重
beta_matrix <- exp(lda_model@beta)
# 获取所有词的名称
terms_names <- lda_model@terms

# 初始化一个空列表来存储每个主题的关键词及其权重
topic_keywords_weights <- list()

# 遍历每个主题
for (topic in 1:lda_model@k) {
  # 获取当前主题下每个词的权重
  topic_weights <- beta_matrix[topic, ]
  
  # 对权重进行排序，获取前 10 个关键词及其权重
  top_terms_indices <- order(topic_weights, decreasing = TRUE)[1:10]
  top_terms <- terms_names[top_terms_indices]
  top_weights <- topic_weights[top_terms_indices]
  
  # 创建一个数据框存储当前主题的关键词及其权重
  topic_keywords_weights[[topic]] <- data.frame(
    Topic = paste0("Topic", topic),
    Keyword = top_terms,
    Weight = top_weights
  )
}

# 将所有主题的关键词及其权重合并为一个数据框
all_keywords_weights <- do.call(rbind, topic_keywords_weights)

# 打印结果
print(all_keywords_weights)

# 计算每个年度的关键词加权值
calculate_weighted_keywords <- function(year_data, all_keywords_weights) {
  weighted_keywords <- data.frame(Keyword = character(), WeightedValue = numeric(), stringsAsFactors = FALSE)
  
  for (topic in unique(all_keywords_weights$Topic)) {
    topic_num <- gsub("Topic", "", topic)
    topic_weight <- year_data[[topic_num]]
    
    # 检查该年度是否有文档
    if (length(topic_weight) > 0) {
      topic_keywords <- all_keywords_weights %>%
        filter(Topic == topic)
      
      for (i in 1:nrow(topic_keywords)) {
        keyword <- topic_keywords$Keyword[i]
        keyword_weight_in_topic <- topic_keywords$Weight[i]
        # 这里假设每个关键词的加权值为该主题的平均权重乘以该关键词在主题中的权重
        keyword_weight <- mean(topic_weight) * keyword_weight_in_topic
        weighted_keywords <- rbind(weighted_keywords, data.frame(Keyword = keyword, WeightedValue = keyword_weight))
      }
    }
  }
  
  if (nrow(weighted_keywords) > 0) {
    weighted_keywords <- weighted_keywords %>%
      group_by(Keyword) %>%
      summarise(TotalWeightedValue = sum(WeightedValue))
  }
  
  return(weighted_keywords)
}

# 按年度分组并计算关键词加权值
yearly_keywords <- data_with_topics %>%
  group_by(Year) %>%
  do(calculate_weighted_keywords(., all_keywords_weights))

# 查看结果
print(yearly_keywords)

# 假设 data_with_topics 已经包含了文档的年度和主题分布信息

# 检查数据框的列名
print(colnames(data_with_topics))

# 根据实际的主题列名更新 topic_col_pattern
topic_col_pattern <- c("1", "2", "3", "4")  # 使用原始列名

# 计算每个主题在每年的频率
annual_topic_frequency <- data_with_topics %>%
  group_by(Year) %>%
  summarise(across(all_of(topic_col_pattern), sum), .groups = "drop")

# 打印 annual_topic_frequency 的列名以进行检查
print(colnames(annual_topic_frequency))

# 检查列名是否包含主题列
topic_cols <- colnames(annual_topic_frequency)[colnames(annual_topic_frequency) %in% topic_col_pattern]
if (length(topic_cols) == 0) {
  stop("annual_topic_frequency 数据框中没有匹配到主题列，请检查列名。")
}

# 计算年度变化率
annual_topic_change_rate <- annual_topic_frequency %>%
  mutate(across(all_of(topic_col_pattern), ~ (. - lag(.)) / lag(.)))

# 计算移动平均（以3年为窗口）
library(zoo)
annual_topic_moving_avg <- annual_topic_frequency %>%
  mutate(across(all_of(topic_col_pattern), ~ zoo::rollmean(., k = 3, fill = NA, align = "right")))

# 计算标准差和方差
annual_topic_stats <- annual_topic_frequency %>%
  summarise(across(all_of(topic_col_pattern), list(sd = sd, var = var)))

# 应用指数平滑
library(tidyr)
library(forecast)
annual_topic_forecast <- annual_topic_frequency %>%
  pivot_longer(cols = all_of(topic_col_pattern), names_to = "Topic", values_to = "Frequency") %>%
  group_by(Topic) %>%
  do({
    if (nrow(.) > 0 && is.finite(max(.$Year))) {
      model <- ets(.$Frequency, model = "AAN")  # 使用ETS模型
      forecast_data <- forecast(model, h = 5)   # 预测未来5年的趋势
      data.frame(Year = seq(max(.$Year) + 1, by = 1, length.out = 5), 
                 Frequency = forecast_data$mean)
    } else {
      # 处理无效数据，这里简单返回一个空数据框
      data.frame(Year = numeric(0), Frequency = numeric(0))
    }
  })

# 将 annual_topic_frequency 转换为长格式以便于绘图
long_format_data <- annual_topic_frequency %>%
  pivot_longer(cols = all_of(topic_col_pattern), names_to = "Topic", values_to = "Frequency")

# 绘制年度变化趋势图
ggplot(long_format_data, aes(x = Year, y = Frequency, color = Topic)) +
  geom_line(size = 1.2) +  # 绘制折线
  geom_point(size = 2) +   # 添加数据点
  
  # 添加标题和标签
  labs(title = "年度主题变化趋势",
       subtitle = "不同主题在各年度的频率变化",
       x = "年度",
       y = "频率",
       color = "主题") +
  
  # 自定义主题
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        plot.subtitle = element_text(hjust = 0.5, size = 14),
        legend.position = "right",
        legend.title = element_text(face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_blank()) +
  
  # 设置 x 轴为每一年显示一次
  scale_x_continuous(breaks = seq(min(annual_topic_frequency$Year), max(annual_topic_frequency$Year), by = 1)) +
  
  # 使用 facet_wrap 为每个主题单独创建面板
  facet_wrap(~ Topic, scales = "free_y")  # 每个主题的 y 轴可以自由调整

library(ggplot2)

# 绘制年度变化趋势图
ggplot(annual_topic_frequency, aes(x = Year)) +
  geom_line(aes(y = `1`, color = "Topic1"), size = 1.2) +
  geom_line(aes(y = `2`, color = "Topic2"), size = 1.2) +
  geom_line(aes(y = `3`, color = "Topic3"), size = 1.2) +
  geom_line(aes(y = `4`, color = "Topic4"), size = 1.2) +
  geom_line(aes(y = `5`, color = "Topic5"), size = 1.2) +
  geom_line(aes(y = `6`, color = "Topic6"), size = 1.2) +
  
  # 添加数据点
  geom_point(aes(y = `1`, color = "Topic1"), size = 2) +
  geom_point(aes(y = `2`, color = "Topic2"), size = 2) +
  geom_point(aes(y = `3`, color = "Topic3"), size = 2) +
  geom_point(aes(y = `4`, color = "Topic4"), size = 2) +
  geom_point(aes(y = `5`, color = "Topic5"), size = 2) +
  geom_point(aes(y = `6`, color = "Topic6"), size = 2) +
  
  # 添加标题和标签
  labs(title = "年度主题变化趋势",
       subtitle = "不同主题在各年度的频率变化",
       x = "年度",
       y = "频率",
       color = "主题") +
  
  # 自定义主题
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        plot.subtitle = element_text(hjust = 0.5, size = 14),
        legend.position = "right",
        legend.title = element_text(face = "bold"),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_blank()) +
  
  # 设置 x 轴为每一年显示一次
  scale_x_continuous(breaks = seq(min(annual_topic_frequency$Year), max(annual_topic_frequency$Year), by = 1))

# 绘制预测趋势图
ggplot(annual_topic_forecast, aes(x = Year, y = Frequency, color = Topic)) +
  geom_line() +
  labs(title = "未来主题趋势预测", x = "年度", y = "预测频率") +
  theme_minimal()

# 计算词频
term_freq <- colSums(as.matrix(dtm))
term_freq_df <- data.frame(Term = names(term_freq), Frequency = term_freq)

# 计算TF-IDF
dtm_tfidf <- weightTfIdf(dtm)
tfidf_matrix <- as.matrix(dtm_tfidf)
tfidf_df <- data.frame(Term = colnames(tfidf_matrix), TFIDF = colSums(tfidf_matrix))

# 合并词频和TF-IDF数据
merged_data <- merge(term_freq_df, tfidf_df, by = "Term")

# 可视化词频和TF-IDF的年度变化趋势
# 首先，我们需要按年度计算词频和TF-IDF

# 创建一个数据框来存储年度词频和TF-IDF
yearly_data <- data.frame(Year = integer(), Term = character(), Frequency = numeric(), TFIDF = numeric(), stringsAsFactors = FALSE)

# 检查并移除 documents$Year 列中的 NA 值
valid_docs <- !is.na(documents$Year)
documents <- documents[valid_docs, ]
dtm <- dtm[valid_docs, ]
dtm_tfidf <- dtm_tfidf[valid_docs, ]
# 计算每个年度的词频和TF-IDF
for (year in unique(documents$Year)) {
  yearly_dtm <- dtm[documents$Year == year, ]
  yearly_tfidf <- dtm_tfidf[documents$Year == year, ]
  
  yearly_freq <- colSums(as.matrix(yearly_dtm))
  yearly_tfidf_values <- colSums(as.matrix(yearly_tfidf))
  
  year_terms <- data.frame(
    Year = year,
    Term = names(yearly_freq),
    Frequency = yearly_freq,
    TFIDF = yearly_tfidf_values
  )
  
  yearly_data <- rbind(yearly_data, year_terms)
}

# 可视化
# 选择一些常见词进行可视化
common_terms <- c("arerobics", "art", "athlete", "balance", "championship")  # 替换为你感兴趣的词

# 创建柱状图
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = Frequency, fill = Term)) +
  geom_bar(stat = "identity", position = "dodge") +  # 使用柱状图
  facet_wrap(~ Term, scales = "free_y") +  # 按 Term 分类，y轴独立
  labs(title = "词频的年度变化趋势", x = "年份", y = "词频") +
  theme_minimal() +
  theme(legend.position = "none")  # 隐藏图例

# 创建柱状图
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = TFIDF, fill = Term)) +
  geom_bar(stat = "identity", position = "dodge") +  # 使用柱状图
  facet_wrap(~ Term, scales = "free_y") +  # 按 Term 分类，y轴独立
  labs(title = "TF-IDF的年度变化趋势", x = "年份", y = "TF-IDF") +
  theme_minimal() +
  theme(legend.position = "none")  # 隐藏图例

# 词频变化趋势
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = Frequency, color = Term)) +
  geom_line() +
  labs(title = "词频的年度变化趋势", x = "年份", y = "词频") +
  theme_minimal()


# TF-IDF变化趋势
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = TFIDF, color = Term)) +
  geom_line() +
  labs(title = "TF-IDF的年度变化趋势", x = "年份", y = "TF-IDF") +
  theme_minimal()

# 创建柱状图
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = Frequency, fill = Term)) +
  geom_bar(stat = "identity", position = "dodge") +  # 使用柱状图
  facet_wrap(~ Term, scales = "free_y") +  # 按 Term 分类，y轴独立
  labs(title = "词频的年度变化趋势", x = "年份", y = "词频") +
  theme_minimal() +
  theme(legend.position = "none")  # 隐藏图例

# 创建柱状图
ggplot(yearly_data %>% filter(Term %in% common_terms), aes(x = Year, y = TFIDF, fill = Term)) +
  geom_bar(stat = "identity", position = "dodge") +  # 使用柱状图
  facet_wrap(~ Term, scales = "free_y") +  # 按 Term 分类，y轴独立
  labs(title = "TF-IDF的年度变化趋势", x = "年份", y = "TF-IDF") +
  theme_minimal() +
  theme(legend.position = "none")  # 隐藏图例

# 假设 yearly_data 是一个包含 Year, Term, Frequency 和 TFIDF 列的数据框
# 先将数据转换为长格式，以便于绘图
long_data <- yearly_data %>%
  filter(Term %in% common_terms) %>%
  pivot_longer(cols = c(Frequency, TFIDF), names_to = "Metric", values_to = "Value")

# 创建柱状图
ggplot(long_data, aes(x = Year, y = Value, fill = Term)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free_y") +  # 按照 Metric 分类，y轴独立
  labs(title = "词频和TF-IDF的年度变化趋势", x = "年份", y = "值") +
  theme_minimal() +
  theme(legend.position = "bottom")  # 修改图例位置

##################################分割线###############################
# 主题词共现关系作图代码摘录

# 提取主题词分布
#topic_word_dist <- colSums(as.matrix(dtm))

# 构建共现关系矩阵
#co_occur_matrix <- topic_word_dist %*% t(topic_word_dist)

# 将共现关系矩阵转换为网络图对象
#co_occur_graph <- graph_from_adjacency_matrix(co_occur_matrix, weighted = TRUE, mode = "undirected")

# 使用 ggraph 绘制网络图
#output_dir = getwd()  # 确保设置输出目录
#png(file = paste(output_dir, "/co_occur_graph.png", sep = ""), width = 3200, height = 2400, res = 300)
#ggraph(co_occur_graph, layout = "fr") + 
 # geom_edge_link(aes(width = weight), alpha = 0.5) + 
 # geom_node_point(size = 3) + 
 # geom_node_text(aes(label = name), repel = TRUE) +
 # theme_void()
#dev.off()


# 提取词频
freq <- colSums(as.matrix(dtm))

# 创建数据框
wf <- data.frame(word = names(freq), freq = freq)

# 可视化词频
p <- ggplot(subset(wf, freq > 6), aes(word, freq))    
p <- p + geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "词频分布", x = "词", y = "频率")
print(p)

# 检查词汇表维度
lda_vocab <- colnames(dtm)
phi <- posterior(lda_model)$terms
theta <- posterior(lda_model)$topics

if (ncol(phi) != length(lda_vocab)) {
  print("警告：phi 矩阵的列数和词汇表的长度不匹配！")
} else {
  print("phi 矩阵的列数和词汇表的长度匹配。")
}

# 创建 JSON 数据用于 LDAvis 可视化
json_lda <- createJSON(
  phi = phi,
  theta = theta,
  doc.length = rowSums(as.matrix(dtm)),
  vocab = colnames(dtm),
  term.frequency = colSums(as.matrix(dtm))
)

# 可视化 LDA 结果
serVis(json_lda)


# 绘制主题强度雷达图
set.seed(123)
topic_intensities <- data.frame(
  Topic = paste("Topic", 1:best_num_topics),
  Intensity = runif(best_num_topics, 0, 1)
)

closed_data <- topic_intensities %>% 
  add_row(Topic = topic_intensities$Topic[1], Intensity = topic_intensities$Intensity[1])

# 计算每个主题对应的角度
angles <- seq(0, 2 * pi, length.out = best_num_topics + 1)
closed_data$angle <- angles

# 使用 ggplot2 绘制雷达图
ggplot(closed_data, aes(x = angle, y = Intensity)) +
  geom_polygon(fill = "lightblue", alpha = 0.5) +
  geom_line(color = "blue") +
  geom_point(color = "blue", size = 2) +
  scale_x_continuous(
    limits = c(0, 2 * pi),
    breaks = angles[1:best_num_topics],
    labels = topic_intensities$Topic
  ) +
  coord_polar(start = 0) +
  labs(
    title = "主题强度雷达图",
    x = NULL,
    y = "主题强度"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    plot.title = element_text(hjust = 0.5, size = 16),
    panel.grid.major = element_line(color = "gray", linetype = "dashed"),
    panel.grid.minor = element_blank()
  )
dev.off()

# 文档-主题分布概率计算
doc_topic_dist <- posterior(lda_best)$topics

# 各主题的热图
output_dir <- getwd()
png(file = paste(output_dir, "/topic_heatmap.png", sep = ""), width = 800, height = 600)
heatmap(doc_topic_dist, Rowv = NA, Colv = NA, col = cm.colors(256), scale = "none",
        xlab = "Term", ylab = "Topic",
        main = "Topic-Word Distribution Heatmap")
dev.off()

# 各主题热度时间趋势图
document_metadata <- data.frame(Year = sample(2015:2023, nrow(doc_topic_dist), replace = TRUE))
doc_topic_time <- cbind(document_metadata, doc_topic_dist)
doc_topic_time <- gather(doc_topic_time, key = "Topic", value = "Probability", -Year)
doc_topic_time$Topic <- gsub("Topic_", "", doc_topic_time$Topic)
doc_topic_time$Topic <- as.numeric(doc_topic_time$Topic)

png(file = paste(output_dir, "/topic_time_trend.png", sep = ""), width = 800, height = 600)
ggplot(doc_topic_time, aes(x = Year, y = Probability, color = factor(Topic))) +
  geom_line() +
  labs(title = "Topic Probability Over Time",
       x = "Year", y = "Probability") +
  theme_minimal()
dev.off()

##################################分割线###############################
# 可视化每个主题新颖度的变化折线图

# 1. 假设已经有了每个文档对应的年份信息
# 这里简单模拟年份信息，实际使用时需要替换为真实数据
doc_years <- sample(2015:2024, nrow(dtm), replace = TRUE)

# 2. 获取每个文档的主题分布
doc_topics <- lda_model@gamma

# 3. 按年份分组计算每年的主题分布
yearly_topic_dist <- lapply(sort(unique(doc_years)), function(year) {
  year_docs <- doc_topics[doc_years == year, ]
  colMeans(year_docs)
})
names(yearly_topic_dist) <- sort(unique(doc_years))

# 4. 计算主题新颖度（使用 Jensen - Shannon 散度）
library(proxy)

# 定义 Jensen - Shannon 散度函数
js_divergence <- function(p, q) {
  m <- 0.5 * (p + q)
  kl_pm <- sum(p * log(p / m), na.rm = TRUE)
  kl_qm <- sum(q * log(q / m), na.rm = TRUE)
  0.5 * (kl_pm + kl_qm)
}

# 计算每年相对于前一年的主题新颖度
topic_novelty <- numeric(length(yearly_topic_dist) - 1)
for (i in 2:length(yearly_topic_dist)) {
  prev_year_dist <- yearly_topic_dist[[i - 1]]
  current_year_dist <- yearly_topic_dist[[i]]
  topic_novelty[i - 1] <- js_divergence(prev_year_dist, current_year_dist)
}

# 输出结果
year_novelty_df <- data.frame(
  Year = names(yearly_topic_dist)[-1],
  Topic_Novelty = topic_novelty
)
print(year_novelty_df)

# 可视化主题新颖度随年份的变化
ggplot(year_novelty_df, aes(x = as.numeric(as.character(Year)), y = Topic_Novelty)) +
  geom_line() +
  geom_point() +
  labs(x = "Year", y = "Topic Novelty", title = "Topic Novelty by Year") +
  theme_minimal()


# 以2年时间片计算主题新颖度数值

# 这里简单模拟年份信息，实际使用时需要替换为真实数据
doc_years <- sample(2015:2024, nrow(dtm), replace = TRUE)

# 2. 获取每个文档的主题分布
doc_topics <- posterior(lda_best)$topics

# 3. 获取每个文档的主题归属
doc_topic_assign <- apply(doc_topics, 1, which.max)

# 4. 确定两年的时间片
start_years <- seq(2015, 2024, by = 2)
time_slices <- paste(start_years, start_years + 1, sep = "-")

# 5. 按时间片和主题分组计算每个主题在各时间片的词项分布
time_slice_topic_term_dist <- list()
num_topics <- ncol(doc_topics)

for (i in seq_along(start_years)) {
  start_year <- start_years[i]
  end_year <- start_year + 1
  time_slice_docs <- dtm[doc_years >= start_year & doc_years <= end_year, ]
  time_slice_topic_term_dist[[time_slices[i]]] <- list()
  for (topic in 1:num_topics) {
    topic_docs <- time_slice_docs[doc_topic_assign[doc_years >= start_year & doc_years <= end_year] == topic, ]
    if (nrow(topic_docs) > 0) {
      topic_term_dist <- apply(topic_docs, 2, sum) / sum(apply(topic_docs, 2, sum))
    } else {
      topic_term_dist <- rep(0, ncol(dtm))
    }
    time_slice_topic_term_dist[[time_slices[i]]][[topic]] <- topic_term_dist
  }
}

# 6. 定义 Jensen - Shannon 散度函数
js_divergence <- function(p, q) {
  m <- 0.5 * (p + q)
  kl_pm <- sum(p * log(p / m), na.rm = TRUE)
  kl_qm <- sum(q * log(q / m), na.rm = TRUE)
  0.5 * (kl_pm + kl_qm)
}

# 7. 计算各主题在每个时间片相对于前一个时间片的新颖度
topic_time_slice_novelty <- matrix(NA, nrow = length(time_slices) - 1, ncol = num_topics)
rownames(topic_time_slice_novelty) <- time_slices[-1]
colnames(topic_time_slice_novelty) <- paste("Topic", 1:num_topics)

for (topic in 1:num_topics) {
  for (i in 2:length(time_slices)) {
    prev_time_slice_dist <- time_slice_topic_term_dist[[time_slices[i - 1]]][[topic]]
    current_time_slice_dist <- time_slice_topic_term_dist[[time_slices[i]]][[topic]]
    topic_time_slice_novelty[i - 1, topic] <- js_divergence(prev_time_slice_dist, current_time_slice_dist)
  }
}

# 8. 输出结果
topic_time_slice_novelty_df <- as.data.frame(topic_time_slice_novelty)
print(topic_time_slice_novelty_df)

# 9. 可视化（以第一个主题为例，可换不同主题，修改数字1即可）

ggplot(data.frame(Time_Slice = rownames(topic_time_slice_novelty_df),
                  Novelty = topic_time_slice_novelty_df[, 1]),
       aes(x = Time_Slice, y = Novelty, group = 1)) +
  geom_line() +
  geom_point() +
  labs(title = "Topic 1 Novelty by Two - Year Time Slices",
       x = "Two - Year Time Slice",
       y = "Topic Novelty") +
  theme_minimal()


library(ggplot2)
library(ggthemes)
library(scales)
library(gridExtra)

# 您的数据
data <- data.frame(
  Topics = 2:30,
  JSD = c(0.4711575, 0.5437349, 0.5409028, 0.5809258, 0.6272898, 0.6650219, 0.6743255, 
          0.6815364, 0.6790667, 0.6978835, 0.7244651, 0.7351762, 0.7358839, 0.7455129, 
          0.7541737, 0.7749927, 0.7851273, 0.7895387, 0.7904511, 0.7978826, 0.8122340, 
          0.8198474, 0.8237945, 0.8202143, 0.8211785, 0.8333279, 0.8320392, 0.8354510, 0.8334889),
  Overlap_Score = c(0.6797369, 0.6477796, 0.6489702, 0.6325408, 0.6145187, 0.6005927, 0.5972554, 
                    0.5946942, 0.5955689, 0.5889686, 0.5798900, 0.5763103, 0.5760754, 0.5728975, 
                    0.5700690, 0.5633826, 0.5601841, 0.5588032, 0.5585185, 0.5562098, 0.5518051, 
                    0.5494966, 0.5483074, 0.5493859, 0.5490950, 0.5454561, 0.5458398, 0.5448252, 0.5454083)
)

# 创建双y轴图表
# 第一个y轴：JSD，第二个y轴：Overlap_Score
p_dual <- ggplot(data, aes(x = Topics)) +
  # 添加JSD线（左侧y轴）
  geom_line(aes(y = JSD, color = "JSD (主题区分度)"), 
            linewidth = 1.5, alpha = 0.8) +
  geom_point(aes(y = JSD, color = "JSD (主题区分度)"), 
             size = 3, shape = 19) +
  
  # 添加Overlap_Score线（右侧y轴，需要缩放）
  geom_line(aes(y = Overlap_Score * 1.5, color = "重叠度分数"), 
            linewidth = 1.5, alpha = 0.8, linetype = "dashed") +
  geom_point(aes(y = Overlap_Score * 1.5, color = "重叠度分数"), 
             size = 3, shape = 17) +
  
  # 缩放右侧y轴
  scale_y_continuous(
    name = "JSD值 (越大越好)",
    sec.axis = sec_axis(~./1.5, name = "重叠度分数 (越小越好)",
                        breaks = seq(0.5, 0.7, 0.05))
  ) +
  
  # 颜色和主题
  scale_color_manual(
    name = "指标",
    values = c("JSD (主题区分度)" = "#2E86AB", "重叠度分数" = "#A23B72")
  ) +
  
  # 主题设置
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold", 
                              margin = margin(b = 20)),
    plot.subtitle = element_text(hjust = 0.5, size = 14, color = "gray40",
                                 margin = margin(b = 15)),
    axis.title = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(color = "#2E86AB", margin = margin(r = 10)),
    axis.title.y.right = element_text(color = "#A23B72", margin = margin(l = 10)),
    axis.text = element_text(size = 10),
    axis.text.y = element_text(color = "#2E86AB"),
    axis.text.y.right = element_text(color = "#A23B72"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    legend.box = "horizontal",
    legend.margin = margin(t = 10),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
    panel.grid.minor = element_line(color = "gray95", linewidth = 0.25),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  
  # 标签
  labs(
    title = "主题模型JSD值与重叠度分数对比分析",
    subtitle = "JSD衡量主题区分度（越大越好），重叠度分数衡量主题相似性（越小越好）",
    x = "主题数",
    caption = "数据来源：LDA主题模型分析 | 可视化：ggplot2"
  ) +
  
  # 添加参考线
  geom_vline(xintercept = c(5, 12, 18, 22), 
             color = "gray70", linewidth = 0.5, linetype = "dashed", alpha = 0.6) +
  
  # 添加文本标注
  annotate("text", x = 5, y = 0.9, label = "AIC推荐", 
           color = "gray40", size = 3.5, angle = 90, hjust = 1) +
  annotate("text", x = 12, y = 0.9, label = "中位数", 
           color = "gray40", size = 3.5, angle = 90, hjust = 1) +
  annotate("text", x = 18, y = 0.9, label = "JSD拐点", 
           color = "gray40", size = 3.5, angle = 90, hjust = 1) +
  annotate("text", x = 22, y = 0.9, label = "似然推荐", 
           color = "gray40", size = 3.5, angle = 90, hjust = 1) +
  
  # 添加关键点标注
  geom_label(data = data[data$Topics == 5, ], 
             aes(x = Topics, y = JSD, label = "5"), 
             fill = "#2E86AB", color = "white", size = 3, nudge_y = 0.02) +
  geom_label(data = data[data$Topics == 12, ], 
             aes(x = Topics, y = JSD, label = "12"), 
             fill = "#2E86AB", color = "white", size = 3, nudge_y = 0.02) +
  geom_label(data = data[data$Topics == 18, ], 
             aes(x = Topics, y = JSD, label = "18"), 
             fill = "#2E86AB", color = "white", size = 3, nudge_y = 0.02) +
  geom_label(data = data[data$Topics == 22, ], 
             aes(x = Topics, y = JSD, label = "22"), 
             fill = "#2E86AB", color = "white", size = 3, nudge_y = 0.02)

# 显示图表
print(p_dual)

# 保存图表
ggsave("topic_analysis_dual_axis.png", plot = p_dual, width = 12, height = 8, dpi = 300)

# 安装并加载plotly
if (!require("plotly")) install.packages("plotly")
library(plotly)

# 创建交互式图表
p_interactive <- plot_ly(data = data, x = ~Topics) %>%
  # 添加JSD轨迹
  add_trace(y = ~JSD, 
            name = "JSD (Topic Distinctiveness)",
            type = 'scatter', 
            mode = 'lines+markers',
            line = list(color = '#2E86AB', width = 3),
            marker = list(color = '#2E86AB', size = 8, 
                          line = list(color = 'white', width = 1)),
            hovertemplate = paste(
              "<b>Number of Topics:</b> %{x}<br>",
              "<b>JSD:</b> %{y:.4f}<br>",
              "<extra></extra>"
            )) %>%
  
  # 添加重叠度分数轨迹
  add_trace(y = ~Overlap_Score, 
            name = "Overlap Score",
            yaxis = "y2",
            type = 'scatter', 
            mode = 'lines+markers',
            line = list(color = '#A23B72', width = 3, dash = 'dash'),
            marker = list(color = '#A23B72', size = 8, symbol = 'diamond',
                          line = list(color = 'white', width = 1)),
            hovertemplate = paste(
              "<b>Number of Topics:</b> %{x}<br>",
              "<b>Overlap Score:</b> %{y:.4f}<br>",
              "<extra></extra>"
            )) %>%
  
  # 布局设置
  layout(
    title = list(
      text = "<b>Interactive Analysis of Topic Model Performance Metrics</b>",
      font = list(size = 24, family = "Arial")
    ),
    xaxis = list(
      title = "<b>Number of Topics</b>",
      tickmode = "linear",
      dtick = 2,
      gridcolor = 'lightgray',
      zerolinecolor = 'lightgray'
    ),
    yaxis = list(
      title = list(
        text = "<b>JSD值</b>",
        font = list(color = '#2E86AB')
      ),
      tickfont = list(color = '#2E86AB'),
      gridcolor = 'lightgray',
      zerolinecolor = 'lightgray'
    ),
    yaxis2 = list(
      title = list(
        text = "<b>Overlap Score</b>",
        font = list(color = '#A23B72')
      ),
      tickfont = list(color = '#A23B72'),
      overlaying = "y",
      side = "right",
      gridcolor = 'lightgray',
      zerolinecolor = 'lightgray'
    ),
    hovermode = "x unified",
    legend = list(
      x = 0.5,
      y = -0.2,
      xanchor = "center",
      yanchor = "top",
      orientation = "h"
    ),
    margin = list(t = 80, b = 100),
    plot_bgcolor = 'white',
    paper_bgcolor = 'white',
    shapes = list(
      list(type = "line", 
           x0 = 5, x1 = 5, 
           y0 = 0, y1 = 1, 
           yref = "paper",
           line = list(color = "gray", width = 1, dash = "dash")),
      list(type = "line", 
           x0 = 12, x1 = 12, 
           y0 = 0, y1 = 1, 
           yref = "paper",
           line = list(color = "gray", width = 1, dash = "dash")),
      list(type = "line", 
           x0 = 18, x1 = 18, 
           y0 = 0, y1 = 1, 
           yref = "paper",
           line = list(color = "gray", width = 1, dash = "dash")),
      list(type = "line", 
           x0 = 22, x1 = 22, 
           y0 = 0, y1 = 1, 
           yref = "paper",
           line = list(color = "gray", width = 1, dash = "dash"))
    ),
    annotations = list(
      list(x = 5, y = 1.05, yref = "paper",
           text = "AIC-optimal number of topics", showarrow = FALSE,
           font = list(size = 10, color = "gray40")),
      list(x = 12, y = 1.05, yref = "paper",
           text = "Median-based selection", showarrow = FALSE,
           font = list(size = 10, color = "gray40")),
      list(x = 18, y = 1.05, yref = "paper",
           text = "Elbow point of JSD curve", showarrow = FALSE,
           font = list(size = 10, color = "gray40")),
      list(x = 22, y = 1.05, yref = "paper",
           text = "Likelihood-ratio test recommendation", showarrow = FALSE,
           font = list(size = 10, color = "gray40"))
    )
  )

# 显示交互式图表
p_interactive

# 保存为HTML文件
htmlwidgets::saveWidget(p_interactive, "topic_analysis_interactive.html")











