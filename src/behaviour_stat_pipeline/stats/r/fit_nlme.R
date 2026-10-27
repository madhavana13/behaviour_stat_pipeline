# Linear and linear mixed models for behaviour_stat_pipeline (optional R backend).
#
# Called by stats/endpoints/linear_models.py as:
#   Rscript fit_nlme.R <spec.json> <data.csv> <out_dir>
# Uses only nlme (recommended package, ships with R) and jsonlite.
#
# One model per endpoint:
#   random effects given  -> nlme::lme   (REML by default)
#   no random effects     -> nlme::gls   (ANCOVA / heteroscedastic linear model)
#   variance_by given     -> weights = varIdent(~ 1 | <variance_by>)  (unequal residual variances)
# Inference: Type III (marginal) F-tests with sum-to-zero contrasts and centred
# covariates; contrasts of equally weighted cell means at covariate means.
# Writes anova.csv, coefficients.csv, contrasts.csv, random_effects.csv, model_info.csv.

suppressPackageStartupMessages({
  library(nlme)
  library(jsonlite)
})
options(contrasts = c("contr.sum", "contr.poly"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: Rscript fit_nlme.R <spec.json> <data.csv> <out_dir>")
spec <- fromJSON(args[1], simplifyVector = TRUE)
d0 <- read.csv(args[2], stringsAsFactors = FALSE, check.names = FALSE)
out_dir <- args[3]

r_version <- paste(R.version$major, R.version$minor, sep = ".")
nlme_version <- as.character(packageVersion("nlme"))
id_col <- spec$id_col
# jsonlite reads an empty JSON array as list(); coerce to character vectors
covariates <- as.character(unlist(spec$covariates))
variance_by <- as.character(unlist(spec$variance_by))
has_random <- !is.null(spec$random) && nzchar(spec$random)
method <- if (is.null(spec$method)) "REML" else spec$method

for (f in names(spec$factor_levels)) d0[[f]] <- factor(d0[[f]], levels = spec$factor_levels[[f]])

anova_rows <- list(); coef_rows <- list(); contrast_rows <- list(); re_rows <- list(); info_rows <- list()

bind <- function(lst) if (length(lst)) do.call(rbind, lst) else data.frame()

fit_one <- function(dd, fixed, weights) {
  if (has_random) {
    lme(fixed, random = as.formula(spec$random), weights = weights, data = dd, method = method,
        control = lmeControl(maxIter = 200, msMaxIter = 200, opt = "optim"))
  } else {
    gls(fixed, weights = weights, data = dd, method = method)
  }
}

for (ep in spec$endpoints) {
  dd <- d0
  dd$.y <- dd[[ep]]
  # endpoint-specific covariate columns, e.g. baseline -> baseline__<endpoint>
  for (cv in names(spec$per_endpoint_columns)) {
    dd[[cv]] <- dd[[sprintf(spec$per_endpoint_columns[[cv]], ep)]]
  }
  fixed <- as.formula(paste(".y ~", spec$fixed))
  needed <- unique(c(".y", all.vars(fixed), variance_by, id_col,
                     if (has_random) all.vars(as.formula(spec$random)) else character(0)))
  dd <- dd[complete.cases(dd[, needed, drop = FALSE]), , drop = FALSE]
  dd <- droplevels(dd)
  # Centre covariates at the mean over animals, so main effects are at the average covariate
  for (cv in covariates) {
    dd[[cv]] <- dd[[cv]] - mean(dd[[cv]][!duplicated(dd[[id_col]])])
  }

  weights <- if (length(variance_by)) varIdent(form = as.formula(paste("~ 1 |", paste(variance_by, collapse = " * ")))) else NULL
  fit <- tryCatch(fit_one(dd, fixed, weights), error = function(e) e)
  base_info <- data.frame(endpoint = ep, n_obs = nrow(dd), n_ids = length(unique(dd[[id_col]])),
                          engine = if (has_random) "nlme::lme" else "nlme::gls", method = method,
                          variance_by = paste(variance_by, collapse = "*"),
                          r_version = r_version, nlme_version = nlme_version, stringsAsFactors = FALSE)
  if (inherits(fit, "error")) {
    info_rows[[ep]] <- cbind(base_info, status = paste("failed:", conditionMessage(fit)),
                             logLik = NA, AIC = NA, BIC = NA, sigma = NA, variance_ratios = "",
                             hetero_lr = NA, hetero_p = NA, shapiro_p = NA)
    next
  }

  N <- fit$dims$N
  p <- if (has_random) length(fixef(fit)) else fit$dims$p
  b <- if (has_random) fixef(fit) else coef(fit)
  V <- vcov(fit)

  # Type III (marginal) F-tests
  a <- anova(fit, type = "marginal")
  den <- if ("denDF" %in% names(a)) a$denDF else rep(N - p, nrow(a))
  anova_rows[[ep]] <- data.frame(endpoint = ep, term = rownames(a), num_df = a$numDF, den_df = den,
                                 F = a$`F-value`, p_value = a$`p-value`, stringsAsFactors = FALSE)

  tt <- summary(fit)$tTable
  coef_df <- if ("DF" %in% colnames(tt)) tt[, "DF"] else rep(N - p, nrow(tt))
  coef_rows[[ep]] <- data.frame(endpoint = ep, coefficient = rownames(tt), estimate = tt[, "Value"],
                                se = tt[, "Std.Error"], df = coef_df, t = tt[, "t-value"],
                                p_value = tt[, "p-value"], stringsAsFactors = FALSE)

  if (has_random) {
    vc <- VarCorr(fit)
    sd_col <- suppressWarnings(as.numeric(vc[, "StdDev"]))
    re_rows[[ep]] <- data.frame(endpoint = ep, component = rownames(vc), std_dev = sd_col,
                                stringsAsFactors = FALSE)
  }

  # Heteroscedasticity: variance ratios and LR test against equal variances
  ratios <- ""; het_lr <- NA; het_p <- NA
  if (length(variance_by)) {
    r <- coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
    ratios <- paste(sprintf("%s=%.10g", names(r), r), collapse = ";")
    fit0 <- tryCatch(update(fit, weights = NULL), error = function(e) NULL)
    if (!is.null(fit0)) {
      lr <- anova(fit0, fit)
      het_lr <- lr$L.Ratio[2]
      het_p <- lr$`p-value`[2]
    }
  }
  res_n <- resid(fit, type = "normalized")
  shap <- if (length(res_n) >= 3 && length(res_n) <= 5000) shapiro.test(res_n)$p.value else NA

  info_rows[[ep]] <- cbind(base_info, status = "ok", logLik = as.numeric(logLik(fit)), AIC = AIC(fit),
                           BIC = BIC(fit), sigma = fit$sigma, variance_ratios = ratios,
                           hetero_lr = het_lr, hetero_p = het_p, shapiro_p = shap)

  # Contrasts of equally weighted cell means (covariates at their centred mean, 0)
  fac_vars <- all.vars(fixed)[sapply(all.vars(fixed), function(v) v != ".y" && is.factor(dd[[v]]))]
  if (length(spec$contrasts) && length(fac_vars)) {
    grid <- expand.grid(lapply(setNames(fac_vars, fac_vars), function(v) levels(dd[[v]])),
                        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    for (v in fac_vars) grid[[v]] <- factor(grid[[v]], levels = levels(dd[[v]]))
    for (cv in setdiff(all.vars(fixed), c(".y", fac_vars))) grid[[cv]] <- 0
    X <- model.matrix(delete.response(terms(fixed)), grid)
    if (!identical(colnames(X), names(b))) stop("contrast design columns do not match model coefficients")
    cons <- spec$contrasts
    for (k in seq_len(nrow(cons))) {
      fct <- cons$factor[k]
      by <- unlist(cons$by[k])
      if (is.null(by)) by <- character(0)
      by_keys <- if (length(by)) unique(grid[, by, drop = FALSE]) else data.frame(row.names = 1)
      for (r_i in seq_len(nrow(by_keys))) {
        in_by <- rep(TRUE, nrow(grid))
        for (bv in by) in_by <- in_by & grid[[bv]] == by_keys[[bv]][r_i]
        lv <- levels(dd[[fct]])
        means <- lapply(lv, function(l) colMeans(X[in_by & grid[[fct]] == l, , drop = FALSE]))
        for (i in seq_along(lv)) for (j in seq_along(lv)) if (i < j) {
          L <- means[[i]] - means[[j]]
          est <- sum(L * b)
          se <- sqrt(as.numeric(t(L) %*% V %*% L))
          # lme: smallest denominator df among the coefficients involved (conservative)
          df <- if (has_random) min(fit$fixDF$X[abs(L) > 1e-12]) else N - p
          tval <- est / se
          by_label <- if (length(by)) paste(sprintf("%s=%s", by, sapply(by, function(bv) as.character(by_keys[[bv]][r_i]))), collapse = ";") else ""
          contrast_rows[[length(contrast_rows) + 1]] <- data.frame(
            endpoint = ep, contrast = if (length(by)) paste(fct, "within", paste(by, collapse = "+")) else fct,
            by_level = by_label, level_a = lv[i], level_b = lv[j], estimate = est, se = se, df = df,
            t = tval, p_value = 2 * pt(abs(tval), df, lower.tail = FALSE), stringsAsFactors = FALSE)
        }
      }
    }
  }
}

write.csv(bind(anova_rows), file.path(out_dir, "anova.csv"), row.names = FALSE)
write.csv(bind(coef_rows), file.path(out_dir, "coefficients.csv"), row.names = FALSE)
write.csv(bind(contrast_rows), file.path(out_dir, "contrasts.csv"), row.names = FALSE)
write.csv(bind(re_rows), file.path(out_dir, "random_effects.csv"), row.names = FALSE)
write.csv(bind(info_rows), file.path(out_dir, "model_info.csv"), row.names = FALSE)
