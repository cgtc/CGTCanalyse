#' Get Local Importances from a Ranger Model
#'
#' This function extracts and summarizes the local variable importance scores
#' from a ranger model fitted using the parsnip package.
#'
#' @param fit_object A fitted model object from a parsnip workflow, containing the ranger engine.
#' @param train_data The training data used to fit the model. Should contain the target variable.
#' @param target_variable The name of the outcome variable in the training data (as a string).
#'
#' @return A tibble summarizing the local variable importances for each class, including the mean, median, and standard deviation of the importance scores.
#' @export
#'
get_local_importances <- function(fit_object, train_data, target_variable) {
  # Step 1: Extract the fitted ranger model from the final fit
  ranger_fit <- fit_object |>
    extract_fit_parsnip() |>
    extract_fit_engine()  # Extract the underlying ranger model

  # Step 2: Access local importances from the ranger model
  local_importances <- as.data.frame(ranger_fit$variable.importance.local)  # Local importances

  # Step 3: Add class labels to local importances (assuming 'target_variable' is the outcome variable)
  local_importances$Class <- train_data[[target_variable]]  # Add class labels

  # Step 4: Reshape the data for easier plotting (long format)
  local_importances_long <- local_importances |>
    pivot_longer(cols = -Class, names_to = "Feature", values_to = "Importance")

  # Step 5: Summarize local importances by class
  local_importances_summary <- local_importances_long |>
    group_by(Class, Feature) |>
    summarise(
      Mean_Importance = mean(Importance, na.rm = TRUE),
      Median_Importance = median(Importance, na.rm = TRUE),
      SD_Importance = sd(Importance, na.rm = TRUE),
      .groups = "drop"
    )

  return(list(
    local_importances_long = local_importances_long,
    local_importances_summary = local_importances_summary
  ))
}
