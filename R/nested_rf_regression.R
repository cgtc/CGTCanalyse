#' Nested Random Forest Regression with Cross-Validation
#'
#' Performs nested cross-validated hyperparameter tuning and model evaluation for
#' random forest regression using `ranger` via the `tidymodels` framework.
#'
#' @param data A data frame or tibble containing predictors and the target variable.
#' @param target_variable A string indicating the name of the target variable column.
#' @param seed An integer seed for reproducibility. Default is 123.
#' @param trees Number of trees to use in the random forest. Default is 500.
#' @param mtry_range A numeric vector of length 2 specifying the range for `mtry` hyperparameter. Default is `c(1, 10)`.
#' @param min_n_range A numeric vector of length 2 specifying the range for `min_n` hyperparameter. Default is `c(2, 10)`.
#' @param levels Number of levels in the regular tuning grid. Default is 5.
#' @param outer_folds Number of outer cross-validation folds. Default is 5.
#' @param inner_folds Number of inner cross-validation folds for nested tuning. Default is 5.
#' @param final_tune_folds Number of folds for tuning on the full dataset before final model fitting. Default is 5.
#' @param metric Metric to use for model evaluation and tuning. Default is `"rmse"`.
#' @param strata Optional column name (string) to use for stratified sampling in resampling. Default is `NULL`.
#' @param group Optional column name (string) to use for grouped resampling (e.g., for repeated measures). Default is `NULL`.
#'
#' @return A list with three elements:
#' \describe{
#'   \item{`outer_performance`}{A tibble containing RMSE and R-squared scores from outer folds.}
#'   \item{`final_model`}{A fitted random forest workflow on the full dataset using the best parameters.}
#'   \item{`feature_importances`}{A tibble of variable importances sorted in descending order.}
#' }
#'
#' @details
#' This function implements nested resampling for unbiased model evaluation and then
#' fits a final model using the full dataset and optimal hyperparameters. It optionally
#' handles stratified or grouped sampling in both inner and outer CV, and can exclude
#' stratification variables from predictor space automatically when appropriate.
#'
#' The function uses the `ranger` engine and assumes
#' numeric predictors and response. Non-numeric columns should be preprocessed or removed.
#'
#' @import tidymodels
#' @importFrom dplyr arrange bind_cols
#' @importFrom purrr map_dfr
#' @importFrom glue glue
#' @importFrom rlang sym
#' @importFrom tibble tibble
#' @importFrom yardstick metric_set rmse rsq
#'
#' @examples
#' \dontrun{
#' results <- nested_rf_regression(
#'   data = my_data,
#'   target_variable = "yield",
#'   strata = "yield_quartile",
#'   trees = 200,
#'   outer_folds = 5,
#'   inner_folds = 3,
#'   final_tune_folds = 5,
#'   metric = "rmse"
#' )
#' results$outer_performance
#' results$feature_importances
#' }
#'
#' @export
nested_rf_regression <- function(data,
                                 target_variable,
                                 seed = 123,
                                 trees = 500,
                                 mtry_range = c(1, 10),
                                 min_n_range = c(2, 10),
                                 levels = 5,
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

    # If strata is set and not the target variable, remove it from predictors
    if (!is.null(strata) && strata != target_variable) {
      rec <- rec |> step_rm(!!rlang::sym(strata))
    }

    rec <- rec |>
      step_zv(all_predictors()) |>
      step_normalize(all_predictors()) |>
      step_naomit(all_predictors(), all_outcomes())

    rf_spec <- rand_forest(
      mtry = tune(),
      trees = trees,
      min_n = tune()
    ) |>
      set_engine("ranger") |>
      set_mode("regression")

    wf <- workflow() |> add_recipe(rec) |> add_model(rf_spec)

    grid <- grid_regular(
      mtry(range = mtry_range),
      min_n(range = min_n_range),
      levels = levels
    )

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

  rf_final <- rand_forest(
    mtry = tune(),
    trees = trees,
    min_n = tune()
  ) |>
    set_engine("ranger", importance = "permutation") |>
    set_mode("regression")

  wf_final <- workflow() |>
    add_recipe(final_recipe) |>
    add_model(rf_final)

  grid <- grid_regular(
    mtry(range = mtry_range),
    min_n(range = min_n_range),
    levels = levels
  )

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

  vi <- final_model$fit$fit$fit$variable.importance
  feature_importances <- tibble(feature = names(vi), importance = vi) |> arrange(desc(importance))

  message("Nested CV complete.")

  return(list(
    outer_performance = outer_results,
    final_model = final_model,
    feature_importances = feature_importances
  ))
}
