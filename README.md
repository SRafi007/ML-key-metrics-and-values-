# Some key metrics and values that help  evaluate and improve  Machine Learning models:

## 1. Accuracy
**Definition**: Accuracy represents the percentage of correct predictions made by the model compared to the total predictions.
**When to Use**: Accuracy is useful for balanced datasets (where the number of samples for each class is roughly equal). In imbalanced datasets, accuracy can be misleading.
**Example**: In a binary classification task (e.g., spam vs. non-spam), if your model predicts 90 out of 100 emails correctly, your accuracy is 90%.
## 2. Loss
**Definition**: Loss is a measure of how well the model's predictions match the actual outcomes. Lower loss indicates a better model.
**Types**: Common types include:
**Mean Squared Error (MSE)**: Used in regression tasks (predicting continuous values).
**Cross-Entropy Loss**: Used in classification tasks (predicting discrete categories).
**Goal**: The aim is to minimize the loss over time as the model learns from the data.
## 3. Epochs
**Definition**: An epoch is one complete pass through the entire training dataset. Typically, models are trained for multiple epochs to allow the model to adjust its weights and improve.
**Caution**: Too few epochs can lead to underfitting (the model doesn’t learn enough), while too many epochs can lead to overfitting (the model learns too much noise and performs poorly on new data).
**Optimal Value**: You'll need to find a balance where the model performs well on both training and unseen data (validation set).
## 4. Loss Curve (Training and Validation Loss)
**Training Loss**: The loss calculated on the training data.
**Validation Loss**: The loss calculated on the validation data (unseen during training).
What to Look for:
**Overfitting**: If training loss keeps decreasing while validation loss starts increasing, your model is overfitting.
**Good Generalization**: If both training and validation losses decrease together, the model is learning well.
## 5. Precision, Recall, and F1-Score
**Precision**: The proportion of true positive predictions among all positive predictions made by the model.
Important when false positives (incorrectly classifying a negative as a positive) are costly.
**Recall**: The proportion of true positives among all actual positives.
Important when missing true positives (false negatives) is costly (e.g., medical diagnoses).
**F1-Score**: The harmonic mean of precision and recall, used when you want a balance between the two.
## 6. Confusion Matrix
**Definition**: A table that shows the true positives, true negatives, false positives, and false negatives for classification models.
Helps You Understand: Where your model is making mistakes and which classes are being confused with each other.
## 7. Learning Rate
**Definition**: Controls how much to change the model in response to the estimated error each time the model weights are updated.
**Caution**: A learning rate that's too high can cause the model to converge too quickly to a suboptimal solution. A learning rate that's too low can make the training process slow.
## 8. Bias and Variance
**Bias**: Refers to the error introduced by approximating a real-world problem with a simplified model.
**High Bias**: Leads to underfitting (the model is too simple to capture the data’s complexity).
**Variance**: Refers to the model’s sensitivity to small fluctuations in the training data.
**High Variance**: Leads to overfitting (the model captures noise along with the signal).
## 9. Gradient Descent
**Definition**: An optimization algorithm used to minimize the loss function by adjusting model parameters (weights).
**Types**:

 - Batch Gradient Descent: Uses the entire dataset to compute the   
   gradient. Stochastic
 - Gradient Descent (SGD): Uses one training    example at a time.
   Mini-batch Gradient
 - Descent: Uses small batches of    data for each update.

## 10. Regularization (L1, L2)
**Definition**: Regularization techniques prevent overfitting by adding a penalty for larger weights.
**L1 Regularization (Lasso)**: Shrinks some weights to zero, which can help in feature selection.
**L2 Regularization (Ridge)**: Distributes the penalty more evenly and discourages very large weights.


## Key Takeaways:
Accuracy alone isn’t always sufficient, especially with imbalanced datasets.
Loss provides a more direct indication of how well the model is performing.
Epochs determine how long the model should train; balance is important to avoid overfitting/underfitting.
Precision, recall, and F1-score are crucial for understanding performance in classification tasks, especially when the dataset is imbalanced.
Focus on these values as  training and evaluating  models, and always use validation or test data to ensure the model generalizes well to unseen data.
