#' Random Forest Classification Model Tuning
#'
#' This function performs tuning of a Random Forest classification model on a given dataset.
#' It includes data preprocessing, parameter tuning, and model evaluation.
#'
#' @param data A data frame containing the dataset. It should include both predictors and the target variable.
#' @param target_variable A string representing the name of the target variable in the dataset.
#' @param seed An integer to set the random seed for reproducibility.
#' @param prop A numeric value between 0 and 1 indicating the proportion of data to be used for training.
#' @param strata A string representing the name of the stratification variable for splitting the data.
#' @param trees An integer indicating the number of trees to grow in the Random Forest model.
#' @param mtry_range A numeric vector of length two specifying the range of values for the number of variables to be randomly sampled at each split.
#' @param min_n_range A numeric vector of length two specifying the range of values for the minimum number of observations required at each node.
#' @param levels An integer indicating the number of levels in the grid for tuning parameters.
#' @param v_folds An integer indicating the number of folds to be used in cross-validation.
#' @param metric A string specifying the metric to be used for model evaluation (e.g., "accuracy").
#'
#' @return A list containing:
#' \describe{
#'   \item{best_parameters}{A data frame of the best parameters selected based on the tuning results.}
#'   \item{model_metrics}{A data frame of the metrics collected from the final model evaluation.}
#' }
#'
#' @export
#'
rf_classification <- function(data,
                              target_variable,
                              seed,
                              prop,
                              strata,
                              trees,
                              mtry_range,
                              min_n_range,
                              levels,
                              v_folds,
                              metric,
                              group = NULL) {

  # Set seed for reproducibility
  set.seed(seed)

  # Split the data into training and testing sets, either by group or regular split
  if (!is.null(group)) {
    # Use group_initial_split if group is provided
    data_split <- group_initial_split(data, prop = prop, group = group)
  } else {
    # Use regular initial_split
    data_split <- initial_split(data, prop = prop, strata = strata)
  }

  train_data <- training(data_split)
  test_data <- testing(data_split)

  # Create a formula for the recipe
  formula <- as.formula(paste(target_variable, "~ ."))

  # Define a recipe for preprocessing
  data_recipe <- recipe(formula, data = train_data) |>
    step_zv(all_predictors()) |>
    step_normalize(all_predictors()) |>   # Normalise predictors
    step_naomit(all_predictors(), all_outcomes())  # Remove rows with missing values

  # Define the Random Forest model with tunable parameters
  rf_spec <- rand_forest(
    mtry = tune(),
    trees = trees,
    min_n = tune()
  ) |>
    set_engine("ranger", importance = "permutation", local.importance = TRUE) |>
    set_mode("classification")

  # Define a workflow
  rf_workflow <- workflow() |>
    add_recipe(data_recipe) |>
    add_model(rf_spec)

  # Define a grid for tuning mtry and min_n
  grid <- grid_regular(
    mtry(range = mtry_range),
    min_n(range = min_n_range),
    levels = levels
  )

  # Define cross-validation for tuning
  cv_folds <- vfold_cv(train_data, v = v_folds, strata = strata)

  # Tune the model
  tune_results <- tune_grid(
    rf_workflow,
    resamples = cv_folds,
    grid = grid,
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

  # Select the best parameters
  best_params <- select_best(tune_results, metric = metric)

  # Finalize the workflow with the best parameters
  final_workflow <- finalize_workflow(rf_workflow, best_params)

  # Fit the final model on the test set
  final_fit <- last_fit(final_workflow, data_split)

  # Collect and display the metrics
  final_fit_metrics <- final_fit |>
    collect_metrics()

  return(list(
    best_parameters = best_params,
    model_metrics = final_fit_metrics,
    final_fit = final_fit,
    train_data = train_data
  ))
}
