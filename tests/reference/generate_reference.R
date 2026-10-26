# Reference values for the stats pipeline, computed independently in base R.
#
# Usage (from repo root):
#   Rscript tests/reference/generate_reference.R
# Writes tests/reference/reference_values.json, which test_reference_values.py
# compares against the Python implementation. Only base R + jsonlite are used.

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
d <- read.csv(file.path(here, "fixture.csv"), stringsAsFactors = FALSE)
d$group <- factor(d$group, levels = c("control", "ifs"))
d$sex <- factor(d$sex, levels = c("female", "male"))

ref <- list(
  meta = list(
    r_version = R.version.string,
    generated = format(Sys.time(), "%Y-%m-%d"),
    fixture = "fixture.csv"
  )
)

# ---------------------------------------------------------------------------
# One-sample t-test and Wilcoxon signed-rank on di vs 0, within each group;
# Holm across groups (per test)
# ---------------------------------------------------------------------------
one <- list()
for (g in levels(d$group)) {
  x <- d$di[d$group == g]
  tt <- t.test(x, mu = 0)
  wt <- wilcox.test(x, mu = 0, exact = TRUE)
  n <- length(x)
  one[[g]] <- list(
    n = n,
    t = unname(tt$statistic),
    t_p = tt$p.value,
    ci_low = tt$conf.int[1],
    ci_high = tt$conf.int[2],
    # R reports V = sum of positive ranks; SciPy reports min(V, n(n+1)/2 - V)
    w_v = unname(wt$statistic),
    w_min = min(unname(wt$statistic), n * (n + 1) / 2 - unname(wt$statistic)),
    w_p = wt$p.value
  )
}
t_adj <- p.adjust(sapply(one, `[[`, "t_p"), "holm")
w_adj <- p.adjust(sapply(one, `[[`, "w_p"), "holm")
for (g in names(one)) {
  one[[g]]$t_p_holm <- unname(t_adj[g])
  one[[g]]$w_p_holm <- unname(w_adj[g])
}
ref$one_sample <- one

# ---------------------------------------------------------------------------
# Two-way ANOVA y ~ group * sex
# Type II: each main effect adjusted for the other main effect, not the interaction
# Type III: sum-to-zero contrasts, each term adjusted for all others
# ---------------------------------------------------------------------------
full <- lm(y ~ group * sex, data = d)
mse <- sum(residuals(full)^2) / full$df.residual
a_gs <- anova(lm(y ~ group + sex + group:sex, data = d))  # group first: SS(sex | group)
a_sg <- anova(lm(y ~ sex + group + sex:group, data = d))  # sex first: SS(group | sex)
row <- function(ss, df) list(sum_sq = ss, df = df, F = (ss / df) / mse,
                             p = pf((ss / df) / mse, df, full$df.residual, lower.tail = FALSE))
type2 <- list(
  group = row(a_sg["group", "Sum Sq"], a_sg["group", "Df"]),
  sex = row(a_gs["sex", "Sum Sq"], a_gs["sex", "Df"]),
  interaction = row(a_gs["group:sex", "Sum Sq"], a_gs["group:sex", "Df"]),
  df_resid = full$df.residual
)

full_sum <- lm(y ~ group * sex, data = d, contrasts = list(group = "contr.sum", sex = "contr.sum"))
d3 <- drop1(full_sum, . ~ ., test = "F")
type3 <- list(
  group = list(sum_sq = d3["group", "Sum of Sq"], df = d3["group", "Df"], F = d3["group", "F value"], p = d3["group", "Pr(>F)"]),
  sex = list(sum_sq = d3["sex", "Sum of Sq"], df = d3["sex", "Df"], F = d3["sex", "F value"], p = d3["sex", "Pr(>F)"]),
  interaction = list(sum_sq = d3["group:sex", "Sum of Sq"], df = d3["group:sex", "Df"],
                     F = d3["group:sex", "F value"], p = d3["group:sex", "Pr(>F)"]),
  df_resid = full_sum$df.residual
)
ref$anova <- list(type2 = type2, type3 = type3)

# ---------------------------------------------------------------------------
# Posthoc on cell means with pooled MSE from the full model; Sidak over k pairs
# Pairs mirror configs/stats.yaml (level index 0/1 in the orders above)
# ---------------------------------------------------------------------------
pairs <- list(
  group_within_sex_0 = list(a = c("control", "female"), b = c("ifs", "female")),
  sex_within_group_0 = list(a = c("control", "female"), b = c("control", "male")),
  sex_within_group_1 = list(a = c("ifs", "female"), b = c("ifs", "male")),
  group_within_sex_1 = list(a = c("control", "male"), b = c("ifs", "male"))
)
k <- length(pairs)
post <- list()
for (nm in names(pairs)) {
  pa <- pairs[[nm]]$a
  pb <- pairs[[nm]]$b
  ya <- d$y[d$group == pa[1] & d$sex == pa[2]]
  yb <- d$y[d$group == pb[1] & d$sex == pb[2]]
  diff <- mean(ya) - mean(yb)
  se <- sqrt(mse * (1 / length(ya) + 1 / length(yb)))
  tval <- diff / se
  p <- 2 * pt(abs(tval), full$df.residual, lower.tail = FALSE)
  post[[nm]] <- list(diff = diff, se = se, t = tval, df = full$df.residual,
                     p = p, p_sidak = 1 - (1 - p)^k)
}
ref$posthoc <- post

# ---------------------------------------------------------------------------
# Correlations (pooled): Pearson; Spearman with asymptotic t (exact = FALSE),
# plus R's default exact p for documentation
# ---------------------------------------------------------------------------
pc <- cor.test(d$x2, d$y, method = "pearson")
sc_t <- cor.test(d$x2, d$y, method = "spearman", exact = FALSE)
sc_e <- cor.test(d$x2, d$y, method = "spearman", exact = TRUE)
ref$correlations <- list(
  pearson = list(r = unname(pc$estimate), p = pc$p.value, n = nrow(d)),
  spearman = list(rho = unname(sc_t$estimate), p_asymptotic_t = sc_t$p.value, p_exact = sc_e$p.value, n = nrow(d))
)

# ---------------------------------------------------------------------------
# OLS y ~ x2 with HC3 standard errors (written out; no sandwich package)
# ---------------------------------------------------------------------------
fit <- lm(y ~ x2, data = d)
X <- model.matrix(fit)
e <- residuals(fit)
h <- hatvalues(fit)
bread <- solve(crossprod(X))
meat <- crossprod(X * (e / (1 - h)))
V <- bread %*% meat %*% bread
se_hc3 <- sqrt(diag(V))[["x2"]]
beta <- coef(fit)[["x2"]]
ref$ols_hc3 <- list(
  beta = beta, se = se_hc3, t = beta / se_hc3, df_resid = fit$df.residual,
  p_t = 2 * pt(abs(beta / se_hc3), fit$df.residual, lower.tail = FALSE),
  p_normal = 2 * pnorm(abs(beta / se_hc3), lower.tail = FALSE)
)

# ---------------------------------------------------------------------------
# Multiple-comparison adjustments on a fixed p-value vector
# ---------------------------------------------------------------------------
p_in <- c(0.001, 0.008, 0.012, 0.03, 0.041, 0.2, 0.5, 0.9)
ref$p_adjust <- list(
  input = p_in,
  holm = p.adjust(p_in, "holm"),
  bonferroni = p.adjust(p_in, "bonferroni"),
  fdr_bh = p.adjust(p_in, "BH"),
  fdr_by = p.adjust(p_in, "BY"),
  sidak = 1 - (1 - p_in)^length(p_in)
)

writeLines(toJSON(ref, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(here, "reference_values.json"))
cat("Wrote", file.path(here, "reference_values.json"), "\n")
