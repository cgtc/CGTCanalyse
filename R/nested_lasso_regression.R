#' Nested Cross-Validated LASSO Regression with tidymodels
#'
#' Performs nested cross-validation for LASSO regression using `glmnet`
#' and the `tidymodels` framework. After evaluating performance on outer
#' folds, the function fits a final model to the entire dataset and extracts
#' non-zero coefficients as feature importances.
#'
#' @param data A data frame containing the predictors and outcome variable.
#' @param target_variable A string specifying the name of the outcome variable.
#' @param seed Integer seed for reproducibility. Default is `123`.
#' @param penalty_range A numeric vector of length 2 specifying the range of log10-penalty values to test (e.g., `c(-4, 0)` tests penalties from `1e-4` to `1`). Default is `c(-4, 0)`.
#' @param levels Integer number of levels (grid points) to use for the penalty tuning grid. Default is `30`.
#' @param outer_folds Number of outer cross-validation folds. Default is `5`.
#' @param outer_repeats Number of repeats for outer cross-validation. Default is `1`.
#' @param inner_folds Number of inner cross-validation folds for hyperparameter tuning. Default is `5`.
#' @param final_tune_folds Number of folds used to tune the final model on the full dataset. Default is `5`.
#' @param metric Performance metric used for model selection (e.g. `"rmse"`, `"rsq"`). Default is `"rmse"`.
#' @param strata Optional column name (unquoted) used for stratification in cross-validation.
#' @param group Optional column name (unquoted) used for grouped cross-validation (e.g. for repeated measures or grouped samples).
#'
#' @return A list with three elements:
#' \describe{
#'   \item{`outer_performance`}{A data frame containing performance metrics from each outer fold.}
#'   \item{`final_model`}{A `workflow` object representing the final fitted model.}
#'   \item{`feature_importances`}{A data frame of non-zero coefficients from the final model, including variable names and their estimated coefficients.}
#' }
#'
#' @examples
#' \dontrun{
#' results <- nested_lasso_regression(
#'   data = my_data,
#'   target_variable = "outcome",
#'   strata = group_column
#' )
#' results$outer_performance
#' results$feature_importances
#' }
#'
#' @export
nested_lasso_regression <- function(data,
                                     target_variable,
                                     seed = 123,
                                     penalty_range = c(-4, 0),  # log10 scale
                                     levels = 30,
                                     outer_folds = 5,
                                     outer_repeats = 1,
                                     inner_folds = 5,
                                     final_tune_folds = 5,
                                     metric = "rmse",
                                     strata = NULL,
                                     group = NULL) {

  set.seed(seed)

  message("Creating outer cross-validation splits...")
  if (!is.null(group)) {
    outer_cv <- group_vfold_cv(data, group = {{group}}, v = outer_folds, repeats = outer_repeats)
  } else if (!is.null(strata)) {
    outer_cv <- vfold_cv(data, v = outer_folds, strata = {{strata}}, repeats = outer_repeats)
  } else {
    outer_cv <- vfold_cv(data, v = outer_folds, repeats = outer_repeats)
  }

  outer_results <- map_dfr(outer_cv$splits, function(split) {
    message(glue::glue("Processing outer fold: {split$id}"))

    train_data <- analysis(split)
    test_data <- assessment(split)

    message("  Creating inner cross-validation splits...")
    if (!is.null(group)) {
      inner_cv <- group_vfold_cv(train_data, group = {{group}}, v = inner_folds)
    } else if (!is.null(strata)) {
      inner_cv <- vfold_cv(train_data, v = inner_folds, strata = {{strata}})
    } else {
      inner_cv <- vfold_cv(train_data, v = inner_folds)
    }

    message("  Preparing recipe and model specification...")
    rec <- recipe(as.formula(paste(target_variable, "~ .")), data = train_data)

    if (!is.null(strata) && strata != target_variable) {
      rec <- rec |> step_rm(!!rlang::sym(strata))
    }

    rec <- rec |>
      step_zv(all_predictors()) |>
      step_normalize(all_predictors()) |>
      step_naomit(all_predictors(), all_outcomes())

    lasso_spec <- linear_reg(
      penalty = tune(),
      mixture = 1  # LASSO
    ) |>
      set_engine("glmnet") |>
      set_mode("regression")

    wf <- workflow() |> add_recipe(rec) |> add_model(lasso_spec)

    grid <- tibble(penalty = 10^seq(penalty_range[1], penalty_range[2], length.out = levels))

    message("  Tuning hyperparameters on inner folds...")
    tuned <- tune_grid(
      wf,
      resamples = inner_cv,
      grid = grid,
      control = control_grid(save_pred = TRUE)
    )

    best <- select_best(tuned, metric = metric)
    final_wf <- finalize_workflow(wf, best)

    message("  Fitting best model to outer test fold...")
    final_fit <- fit(final_wf, data = test_data)
    pred <- predict(final_fit, test_data) |> bind_cols(test_data[target_variable])

    metrics <- metric_set(rmse, rsq)(pred, truth = !!sym(target_variable), estimate = .pred)
    metrics$fold <- split$id
    metrics
  })

  message("Training final model on full dataset...")

  final_recipe <- recipe(as.formula(paste(target_variable, "~ .")), data = data)

  if (!is.null(strata) && strata != target_variable) {
    final_recipe <- final_recipe |> step_rm(!!rlang::sym(strata))
  }

  final_recipe <- final_recipe |>
    step_zv(all_predictors()) |>
    step_normalize(all_predictors()) |>
    step_naomit(all_predictors(), all_outcomes())

  lasso_final <- linear_reg(
    penalty = tune(),
    mixture = 1
  ) |>
    set_engine("glmnet") |>
    set_mode("regression")

  wf_final <- workflow() |>
    add_recipe(final_recipe) |>
    add_model(lasso_final)

  grid <- tibble(penalty = 10^seq(penalty_range[1], penalty_range[2], length.out = levels))

  message("Performing final hyperparameter tuning on full dataset...")
  if (!is.null(group)) {
    cv_full <- group_vfold_cv(data, group = {{group}}, v = final_tune_folds)
  } else if (!is.null(strata)) {
    cv_full <- vfold_cv(data, v = final_tune_folds, strata = {{strata}})
  } else {
    cv_full <- vfold_cv(data, v = final_tune_folds)
  }

  tuned_full <- tune_grid(
    wf_final,
    resamples = cv_full,
    grid = grid,
    control = control_grid(save_pred = TRUE)
  )

  best_full <- select_best(tuned_full, metric = metric)
  final_wf <- finalize_workflow(wf_final, best_full)

  message("Fitting final model with best hyperparameters...")
  final_model <- fit(final_wf, data = data)

  # Extract non-zero coefficients as importance
  vip <- tidy(final_model) |>
    filter(term != "(Intercept)", estimate != 0) |>
    arrange(desc(abs(estimate))) |>
    rename(feature = term, importance = estimate)

  message("Nested CV complete.")

  return(list(
    outer_performance = outer_results,
    final_model = final_model,
    feature_importances = vip
  ))
}
