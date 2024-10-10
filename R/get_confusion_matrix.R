#' Generate Confusion Matrix with Percentages for a Fitted Model
#'
#' This function generates a confusion matrix from a fitted model's predictions and calculates
#' row-wise percentages. The output is a tidy data frame that includes the counts and percentages
#' of each cell in the confusion matrix.
#'
#' @param fit_object A fitted model object from a parsnip workflow containing predictions.
#' @param target_variable The name of the outcome variable in the dataset (as a string).
#'
#' @return A tidy data frame containing the true class, predicted class, count of observations,
#' and the percentage of observations for each cell in the confusion matrix.
#' @export
#'
get_confusion_matrix <- function(fit_object, target_variable) {
  # Assuming stage_final_fit is already defined and contains predictions
  stage_predictions <- fit_object |>
    collect_predictions()

  # Generate the confusion matrix
  stage_confusion_matrix <- conf_mat(stage_predictions, truth = target_variable, estimate = .pred_class)

  # Extract the confusion matrix data
  cm_data <- as.matrix(stage_confusion_matrix$table)

  # Calculate percentages for each cell in the confusion matrix
  cm_percents <- prop.table(cm_data, margin = 1) * 100  # calculate row-wise percentages

  # Create row and column names for the confusion matrix
  row_names <- rownames(cm_data)
  col_names <- colnames(cm_data)

  # Reshape the confusion matrix data into a tidy format
  cm_df <- expand.grid(True_Class = row_names, Predicted_Class = col_names)
  cm_df$Count <- as.vector(cm_data)
  cm_df$Percent <- as.vector(cm_percents)

  # Reorder True_Class and Predicted_Class based on stage_order
  cm_df$True_Class <- factor(cm_df$True_Class)
  cm_df$Predicted_Class <- factor(cm_df$Predicted_Class)

  return(confusion_matrix_df = cm_df)
}
