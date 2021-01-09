import numpy as np


class Explainer:
    def __init__(self, plot_name):
        self.plot_name = plot_name

        if self.plot_name == 'EPLR':
            print('\n\u2022 Explains the prediction of a linear type regressor\n'+
                  '\u2022 Explains the effect of each feature on the target.\n'
                  '\u2022 The features x0, x1, x2... are features' + '\n' +
                  'in base_features[] with indexes 0, 1, 2,... respectively.'
                  '\n\u2022 Feature at the top contributes the most\n' +
                  'and the one at the bottom, the least.')

        elif self.plot_name == 'EPLRW':
            print("This show weights plot depicts the contribution of each of the features towards prediction of the"
                  "target feature in the form of weights.")

        elif self.plot_name == 'EPTR':
            print(
                '\n\u2022 Explains the prediction of a tree type regressor\n'
                '\n\u2022 Explains the effect of each feature on the target.\nThe features x0, x1, x2... are features in base_features[] with indexes 0, 1, 2,... respectively.\n'
                '\n\u2022 Feature at the top contributes the most and the one at the bottom, the least.\n\n')

        elif self.plot_name == 'EFRF':
            print(
                "\n\n\u2022 This show weights plot shown below depicts the contribution of each of the features towards prediction of the"
                " target feature \nin the form of weights by shuffling the values of each feature in the dataset to determine the importance of each feature.\n"
                
                "\n\u2022 If the changes values of a feature affects the prediction to a greater extent, the feature is considered more important, and vice"
                "versa.\n"
                
                "\n\u2022 When we see that 'Feature X' has the highest score with 'x', it means, when we permute the "
                "displacement feature,\nit will change the accuracy of the model as big as 'x'. The value after "
                "the plus-minus sign is the uncertainty value.\n\n")

        elif self.plot_name == 'ESW':
            print("\n\n\u2022 This show weights plot depicts the contribution of each of the features towards prediction of the"
                  " target feature in the form of weights\n\n")

        elif self.plot_name == 'EWX':
            print(
                '\n\n\u2022 This xgboost feature importance plot explains the prediction of the xgboost regressor in the form of '
                'weights.')

        elif self.plot_name == 'EWL':
            print(
                '\n\n\u2022 This LightGBM feature importance plot explains the prediction of the LightGBM regressor in the form '
                'of weights.')

        elif self.plot_name == 'EWC':
            print(
                '\n\n\u2022 This CatBoost feature importance plot explains the prediction of the CatBoost regressor in the form '
                'of weights.')

        elif self.plot_name == 'EPLC':
            print(
                '\n\n\u2022 Explains the prediction of a linear type classifier\n' + 'Explains the effect of each feature on the '
                                                                          'target.\nThe features x0, x1, x2... are '
                                                                          'features' + '\n' +
                'in base_features[] with indexes 0, 1, 2,... respectively. Feature at the top contributes the most\n' +
                'and the one at the bottom, the least.')

        elif self.plot_name == 'EPTCD':
            print(
                '\n\n\u2022 Explains the prediction of a Decision tree Classifier.\n' + 'Explains the effect of each feature on the target.\nThe features x0, x1, x2... are features' + '\n' +
                'in base_features[] with indexes 0, 1, 2,... respectively. Feature at the top contributes the most\n' +
                'and the one at the bottom, the least.')

        elif self.plot_name == 'EPTCR':
            print(
                '\n\n\u2022 Explains the prediction of a Random Forest tree Classifier.\n' + 'Explains the effect of each feature on the target.\nThe features x0, x1, x2... are features' + '\n' +
                'in base_features[] with indexes 0, 1, 2,... respectively. Feature at the top contributes the most\n' +
                'and the one at the bottom, the least.')

        elif self.plot_name == 'PDPB':
            print('\n\n\u2022 This pdp isolate plot explains the variation of confidence in prediction w.r.t a particular '
                  'feature.\n' + '\n\u2022 The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.\n'
                  '\n\u2022 A blue shaded area indicates level of confidence\n\n')

        
        elif self.plot_name == 'SDP':
            print("\n\u2022 The x-axis represents the model's output. In this case, the units are sale price.\n"
                  
                  "\n\u2022 The plot is centered on the x-axis at explainer.expected_value. All SHAP values are relative to the"
                  " model's expected value like a linear model's effects are relative to the intercept.\n\n"
                  
                  "\n\u2022 The y-axis lists the model's features. By default, the features are ordered by descending importance. The importance"
                  " is calculated over the observations plotted.\n\n"
                  
                  "\n\u2022 This is usually different than the importance ordering for the entire dataset. In addition to feature importance ordering, the"
                  " decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.\n\n"
                  
                  "\n\u2022 Each observation's prediction is represented by a colored line. At the top of the plot, each line strikes the x-axis at"
                  " its corresponding observation's predicted value.\n\n"
                  
                  "\n\u2022 This value determines the color of the line on a spectrum. Moving from the bottom of the plot to the top, SHAP values for each"
                  " feature are added to the model's base value.\n\n"
                  
                  "\n\u2022 This shows how each feature contributes to the overall prediction. At the bottom of the plot, the observations converge at"
                  " explainer.expected_value.\n\n")

        elif self.plot_name == 'SSP':
            print('\n\n\u2022 Create a SHAP beeswarm plot, colored by feature values when they are provided. \n'
                  
                  '\n\u2022 This plot depicts the feature importance of each feature w.r.t to its SHAP value.\n'
                  
                  '\n\u2022 The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.\n'
                  
                  '\n\u2022 Color shows whether that variable is high (in red) or low (in blue) for that observation.\n\n')

        elif self.plot_name == 'MODP':
            print('\n\n\u2022 The plot shows plainly that several interactions drive this prediction’s score higher.')

        
        elif self.plot_name == 'YBRP' or self.plot_name == 'residuals':
            print("\n\n\u2022 A residual is a difference between the target and predicted values, i.e. the error of the prediction.n\n"
                  
                  "\n\u2022 The ResidualsPlot Visualizer shows the difference between residuals on the vertical axis and the dependent variable on the horizontal axis,\nallowing you to detect regions within the target that may be susceptible to more or less error.\n\n")

        elif self.plot_name == 'YBPEP' or self.plot_name == 'error':
            print("\n\n\u2022 The Prediction Error Visualizer visualizes prediction errors as a scatter-plot of the predicted and"
                  "\nactual values. We can then visualize the line of best fit and compare it to the 45º line.\n\n")
        
        
        elif self.plot_name == 'rfe':
            print("\n\n\u2022 The RFECV plot depicts the performance of the training model corresponding to the number of features selected for training."
                  
                  "\n\n\u2022 The performance can be judged on various metrics like 'accuracy', 'f1', etc. for classification, 'r2', 'max_error', etc."
                  "\n for regression, and so on.")
            
        elif self.plot_name == 'learning':
            print("\n\n\u2022 Learning Curve: Line plot of learning (y-axis) over experience (x-axis)."
                  
                  "\n\n\u2022 There are three common dynamics that you are likely to observe in learning curves; they are:"

                  "\n\t\t\u2022 Underfit."
                  "\n\t\t\u2022 Overfit."
                  "\n\t\t\u2022 Good Fit."
                 
                  "\n\n\u2022 An underfit model can be identified from the learning curve of the training loss only."
                  "\nIt may show a flat line or noisy values of relatively high loss, indicating that the model was unable to learn the training dataset at all."
                  
                  "\n\n\u2022 A plot of learning curves shows overfitting if the plot of training loss continues to decrease with experience OR"
                  "\nif the plot of validation loss decreases to a point and begins increasing again."
                  
                  "\n\n\u2022 A good fit is identified by a training and validation loss that decreases to a point of stability with a minimal gap between"
                  "\nthe two final loss values.\n\n")
        
        elif self.plot_name == 'vc':
            print("\n\n\u2022 A Validation Curve is an important diagnostic tool that shows the sensitivity between to changes in a Machine Learning model’s"
                  "\naccuracy with change in some parameter of the model."
                  
                  "\n\n\u2022 Ideally, we would want both the validation curve and the training curve to look as similar as possible."
                  
                  "\n\n\u2022 If both scores are low, the model is likely to be underfitting. This means either the model is too simple or it is informed"
                  "\nby too few features. It could also be the case that the model is regularized too much."
                  
                  "\n\n\u2022 If the training curve reaches a high score relatively quickly and the validation curve is lagging behind, the model is overfitting."
                  "\nThis means the model is very complex and there is too little data; or it could simply mean there is too little data.")
        
        elif self.plot_name == 'manifold':
            print("\n\n\u2022 Manifold learning is an approach to non-linear dimensionality reduction. Algorithms for this task are based on the idea that"
                  "\nthe dimensionality of many data sets is only artificially high."
                  
                  "\n\n\u2022 t-SNE Manifold converts affinities of data points to probabilities. The affinities in the original space are represented by"
                  "Gaussian joint probabilities and the affinities in the embedded space are represented by Student’s t-distributions.")
        
        elif self.plot_name == 'parameter':
            print('')
        
        elif self.plot_name == 'feature':
            print("\n\n\u2022 Explains the importance of various features in the process of prediction.")
        
        elif self.plot_name == 'tree':
            print("\n\n\u2022 Plot the decision-tree structure for the trained model. Works with ensemble-type classifiers only.")
        
        
        
        elif self.plot_name == 'auc':
            print("\n\n\u2022 An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all"
                  "\nclassification thresholds. This curve plots two parameters: True Positive Rate vs. False Positive Rate")
        
        elif self.plot_name == 'threshold':
            print('')
        
        elif self.plot_name == 'pr':
            print('\n\n\u2022 The plot depicts the precision vs. recall variation. Recall is provided as the x-axis and precision is provided as the y-axis.')
        
        elif self.plot_name == 'confusion_matrix':
            print("\n\n\u2022 A 2-D array comparing predicted category labels to the true label. For binary classification, these are the True Positive, True"
                  "Negative,\nFalse Positive and False Negative categories.")
        
        elif self.plot_name == 'class_report':
            print("\n\n\u2022 A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions"
                  "\nare True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to"
                  "\npredict the metrics of a classification report")
        
        elif self.plot_name == 'boundary':
            print("\n\n\u2022 A decision boundary or decision surface is a hypersurface that partitions the underlying vector space into two sets, one for"
                  "\neach class. The classifier will classify all the points on one side of the decision boundary as belonging to one class and all those"
                  "\non the other side as belonging to the other class.")
        
        elif self.plot_name == 'calibration':
            print("\n\n\u2022 Plots the deviation from ideal predictions(dotted lines). X-axis is the average probablity that result belongs to positive"
                  "\nclass and y-axis is the fraction of actual positives. ")
        
        elif self.plot_name == 'dimension':
            print('')
        
        elif self.plot_name == 'lift':
            print("\n\n\u2022 A lift curve shows the ratio of a model prediction to a random guess"
                  "\nFor example, Lift = (Expected Response In A Specific Lot Of 10,000 data points using Predictive Model) / (Expected Response In A Random Lot"
                  "\nof 10,000 data points Without Using Predictive Model)"
                  
                  "\n\n\u2022 In short, lift is a measure of the effectiveness of a predictive model calculated as the ratio between the results obtained with and"
                  "without the predictive model.")
        
        elif self.plot_name == 'gain':
            print("\n\n\u2022 The y-axis shows the percentage of positive responses. This is a percentage of the total possible positive responses."
                  "\n\n\u2022 The x-axis shows the percentage of customers contacted, which is a fraction of the 100,000 total customers."
                  "\n\n\u2022 Baseline (overall response rate): If we contact X% of customers then we will receive X% of the total positive responses.")
        
        
        
        
##==============================================Evaluation metrics explainer section===============================================================================        
        
        
        elif self.plot_name == 'Classification Prerequisite':
            print('True Positive:\n'

                  'Interpretation: You predicted positive and itâ€™s true.\n'

                  'You predicted that a woman is pregnant and she actually is.\n'

                  'True Negative:\n'

                  'Interpretation: You predicted negative and itâ€™s true.\n'

                  'You predicted that a man is not pregnant and he actually is not.\n'

                  'False Positive: (Type 1 Error)\n'

                  'Interpretation: You predicted positive and itâ€™s false.\n'

                  'You predicted that a man is pregnant but he actually is not.\n'

                  'False Negative: (Type 2 Error)\n'

                  'Interpretation: You predicted negative and itâ€™s false.\n'

                  'You predicted that a woman is not pregnant but she actually is.\n\n')
        elif self.plot_name == 'Accuracy':
            print('Accuracy is the fraction of predictions our model got right.\n' +
                  'Accuracy = Number of correct predictions/Total number of predictions\n' +
                  'Accuracy=(TP+TN)/(TP+TN+FP+FN)\n' +
                  'This is good metric only for balanced data set\n' +
                  'The best value is 1 and the worst value is 0')
        elif self.plot_name == 'Gini Score':
            print(
                'Gini Index, also known as Gini impurity, calculates the amount of probability of a specific feature that is classified incorrectly when selected randomly. \n' +
                'Gini index varies between values 0 and 1 \n' +
                'where 0 expresses the purity of classification, \n' +
                'This Score mainly used in Tree Based Classifier (EX: DecisionTreeClassifier)\n' +
                'And 1 indicates the random distribution of elements across various classes. \n' +
                'The value of 0.5 of the Gini Index shows an equal distribution of elements over some classes')
        elif self.plot_name == 'Precision':
            print('The precision is the ratio ``tp / (tp + fp)`` \n' +
                  '``tp`` is the number of true positives and ``fp`` the number of false positives. \n' +
                  'The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. \n' +

                  'To reduce  type 1 error (False Positive Rate)\n' +

                  'Also called as  Positive Prediction Value\n' +

                  'Precision= TP/(TP+FP)\n' +

                  'EX: Spam Detection\n' +

                  'FN -> Spam mail in Inbox \n' +

                  'FP -> Inbox mail in spam box -> High Priority\n' +
                  'Precision is used in dataset concerns more about False Positive\n'+
                  'The best value is 1 and the worst value is 0')
        elif self.plot_name == 'Recall':
            print('The recall is the ratio ``tp / (tp + fn)``\n' +
                  ' where ``tp`` is the number of true positives and ``fn`` the number of false negatives.\n' +
                  ' The recall is intuitively the ability of the classifier to find all the positive samples.\n' +
                  ' To reduce type 2 error (False Negative Rate)\n' +

                  'Also called as Sensitivity\n' +

                  'Recall = TP/(TP+FN)\n' +

                  'EX: Cancer Detection\n' +

                  'FP -> Person without cancer predicted as YES\n' +

                  'FN -> Person with cancer predicted as NO -> Higher Priority\n' +
            'Recall is used in dataset concerns more about False Negative \n'+

                  'The best value is 1 and the worst value is 0.')
        elif self.plot_name == 'F1 Score':
            print('Also known as balanced F-score or F-measure or Harmonic Mean\n' +
                  'The F1 score can be interpreted as a weighted average of the precision and recall \n' +
                  'The relative contribution of precision and recall to the F1 score are equal\n' +
                  'The formula for the F1 score is::\n' +
                  'F1 = 2 * (precision * recall) / (precision + recall)\n' +
                  'EX: Bank predicting whether amount is transacted or not\n' +
                  'Both FP and FN is important'
                'F1 Score is used in dataset concerns both False Positive and Negative \n'+

                  'F1 score reaches its best value at 1 and worst score at 0.\n')
        elif self.plot_name == 'F-Beta Score':
            print('The F-beta score is the weighted harmonic mean of precision and recall\n' +
                  'The `beta` parameter determines the weight of recall in the combined score.\n' +
                  '``beta < 1`` lends more weight to precision\n' +
                  '``beta > 1`` favors recall \n' +
                  '(``beta -> 0`` considers only precision, ``beta -> +inf``only recall).\n' +
                  'F-Beta Score is used for datasets where we can set Importance for both False Positive and Negative according to the Model \n' +
                  'Its optimal value at 1 and its worst value at 0.\n')
        elif self.plot_name == 'AUC Score':
            print('Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.\n' +
                  'It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) \n' +
                  'for a number of different candidate threshold values between 0.0 and 1.0.\n' +

                  'In other words The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 â€“ FPR)\n' +
                  'Its optimal value at 1 and its worst value at 0.\n' +
                  'Any Model with value below 0.5 is considered as bad model\n')
        elif self.plot_name == 'Classification Report':
            print('Build a text report showing the main classification metrics.\n' +
                  'Precision\n' +
                  'Recall\n' +
                  'F1-Score\n')
        elif self.plot_name == 'Matthews CorrCoef':
            print('Compute the Matthews correlation coefficient (MCC)\n' +
                  'It takes into account true and false positives and negatives and is generally regarded as a balanced measure\n' +
                  'which can be used even if the classes are of very different sizes\n' +
                  'The statistic is also known as the phi coefficient.\n' +

                  'MCC=((TP * TN) - (FP * FN))/SQRT[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]\n' +
            'MCC is considered for datasets where the balanced score is concerned (same score even swapping predicted and actual values)\n'+
                  'A coefficient of +1 represents a perfect prediction, 0  an average random prediction and -1 an inverse prediction.\n')
        elif self.plot_name == 'Hamming Loss':
            print('The Hamming loss is the fraction of labels that are incorrectly predicted.\n' +
                  'Hamming Loss=1-Accuracy\n' +
                  'HAMMING LOSS = 1/N * SUMMATION( Y XOR X)\n'
                  'Hamming loss is considered for pointing out Loss seperately in dataset\n'+
                  'Optimal Value is 0 and Worst value is 1')
        elif self.plot_name == 'Log Loss':
            print('Log Loss defined as the negative\n' +
                  'log-likelihood of a logistic model that returns ``y_pred`` probabilities\n' +
                  'for its training data ``y_true``\n' +
                  'Formula:\n' +
                  '-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))'
                  'The log_loss function computes log loss given a list of ground-truth labels and a probability matrix\n' +
                  'as returned by an estimatorâ€™s predict_proba method.\n' +
                  'Optimal Value is 0 and Worst value is 1')
        elif self.plot_name == 'Zero One Loss':
            print('Zero-one loss is a common loss function used with classification learning.\n' +
                  'It assigns 0 to loss for a correct classification and 1 for an incorrect classification.' +
                  'Optimal Value is 0 and Worst value is 1')
        elif self.plot_name == 'Cohen Kappa Score':
            print('This function computes Cohen\'s kappa [1]\n' +
                  'A score that expresses the level of agreement between two annotators on a classification problem.\n' +
                  '.. math::\n' +
                  '   kappa = (p_o - p_e) / (1 - p_e)\n' +
                  '`p_o` is the empirical probability of agreement\n' +
                  '`p_e` is estimated using a per-annotator empirical prior over the class labels [2]\n' +
                  ' This measure is intended to compare labelings by different human annotators, not a classifier versus a ground truth.\n' +

                  'The kappa score is a number between -1 and 1.\n' +
                  'Scores above 0.8 are generally considered good agreement; zero or lower means no agreement (practically random labels).')
        elif self.plot_name == 'Explained Variance Score':
            print('Variance is a measure of how far observed values differ from the average of predicted values, \n' +
                  'i.e., their difference from the predicted value mean.\n' +
                  'Explained_Variance(y,y\') = 1 -(var{y - y\'}/var{y})\n' +
                  'where y -  Estimated Target Output\n' +
                  'y\' = Correct Output\n' +
                  'Simply Explained_Variance= 1-(Var(Error) - Var(Correct Output)) '
                  'Best possible score is 1.0, lower values are worse.')
        elif self.plot_name == 'R2 Score':
            print('R^2 (coefficient of determination) regression score function.\n' +
                  'Similar to accuracy score in Classification\n' +
                  'It gives no information about Prediction Error\n' +
                  '1-(variance of Model Error/variance of Worst Possible Error)\n' +
                  'Best possible score is 1.0 and it can be negative (because the\n' +
                  'model can be arbitrarily worse).')
        elif self.plot_name == 'Mean Absolute Error':
            print(
                'Mean Absolute Error (MAE) is a measure of errors between paired observations expressing the same phenomenon.\n' +
                'MAE is calculated as: (SUMMATION(|Y - X|))/N\n' +
                'where Y is Predicted Value, X is Actual Value and N is Total Data Points\n' +
                'It is thus an arithmetic average of the absolute errors\n' +
                'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Mean Absolute Percentage Error':
            print(
                'The mean absolute percentage error (MAPE) is the mean or average of the absolute percentage errors of forecasts.\n' +
                'Error is defined as actual or observed value minus the forecasted value. \n' +
                'Percentage errors are summed without regard to sign to compute MAPE')
        elif self.plot_name == 'Median Absolute Error':
            print('Median absolute error output is non-negative floating point.\n' +
                  'It is similar to Mean Absolute Error only changing factor is Median instead of Mean\n' +
                  'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Mean Squared Error':
            print('Mean squared Error measures the average of the squares of the errors\n' +
                  'the average squared difference between the estimated values and the actual value.\n' +
                  'MSE = 1/n (Summation(y - y\'))\n' +
                  'where y is the predicted value and y\' is actual value\n' +
                  'Example : Take the sample value as 0.1 and 10\n' +
                  '0.1->(0.1)^2 = 0.01 (samples near the model -> more priority ->came closer)\n' +
                  ' 10 ->10^2 = 100 (samples far the model -> low priority ->gone far)\n' +
                  'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Root Mean Squared Error':
            print('Rooted value of Mean squared Error\n' +
                  'where the Mean squared Error measures the average of the squares of the errors\n' +
                  'the average squared difference between the estimated values and the actual value.\n' +
                  'MSE changes to Degree 2(because of squaring)\n' +
                  'RMSE is used to get back the degree to 1\n' +
                  'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Mean Squared Log Error':
            print(
                'Mean squared logarithmic error (MSLE) can be interpreted as a measure of the ratio between the true and predicted values.\n' +
                'Mean squared logarithmic error is, as the name suggests, a variation of the Mean Squared Error.\n'
                'Used to Scale Larger data to smaller \n' +
                'EX: Population count (in billions) scaled using MSLE and take logarithmic data\n' +
                'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Root Mean Squared Log Error':
            print('Square root of Mean Squared Log Error\n' +
                  'MSLE changes to Degree 2(because of squaring)\n' +
                  'RMSLE is used to get back the degree to 1\n' +
                  'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Max Error':
            print('max_error metric calculates the maximum residual error.\n' +
                  'Max Error is used to find the worst case prediction of model\n' +
                  'Best Value is 0 and Worst Value has no limit')
        elif self.plot_name == 'Silhouette Score':
            print('Compute the mean Silhouette Coefficient of all samples.\n' +
                  'The Silhouette Coefficient is calculated using the mean intra-cluster distance (``a``) and the mean nearest-cluster distance (``b``) for each sample.\n' +
                  'The Silhouette Coefficient for a sample is ``(b - a) / max(a,b)``.\n' +
                  'The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.\n' +
                  'Negative values generally indicate that a sample has been assigned to the wrong cluster, \n' +
                  'as a different cluster is more similar.')
        elif self.plot_name == 'Silhouette Sample':
            print(
                'The Silhouette Coefficient is calculated using the mean intra-cluster distance (``a``) and the mean nearest-cluster distance (``b``) for each sample.\n' +
                'The Silhouette Coefficient for a sample is ``(b - a) / max(a,b)``.\n' +
                'The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.\n' +
                'Negative values generally indicate that a sample has been assigned to the wrong cluster, \n' +
                'as a different cluster is more similar.')
        elif self.plot_name == 'contingency_matrix':
            print(
                'Contingency Matrix is a matrix which provides a basic picture of the interrelation between two variables and can help find interactions between them.\n' +
                'Similar to confusion matrix\n' +
                'Takes Unique Actual values  and Unique Predicted values to form matrix')
        elif self.plot_name == 'Mutual Info Score':
            print('The Mutual Information is a measure of the similarity between two labels of the same data.\n' +
                  'Where :math:`|U_i|` is the number of the samples in cluster \n' +
                  ':math:`U_i` and :math:`|V_j|` is the number of the samples in cluster\n' +
                  ':math:`V_j`, the Mutual Information between clusterings\n' +
                  ':math:`U` and :math:`V` is given as:\n' +
                  '.. math::\n' +
                  '    MI(U,V)=\\sum_{i=1}^{|U|} \\sum_{j=1}^{|V|} \\frac{|U_i\\cap V_j|}{N}\n' +
                  '   \\log\\frac{N|U_i \\cap V_j|}{|U_i||V_j|}\n' +
                  'No Mutual Information gives 0 and value gets increased with increase in similarity of two labels')
        elif self.plot_name == 'Normalized Mutual Info Score':
            print(
                'Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation).')
        elif self.plot_name == 'Adjusted Mutual Info Score':
            print(
                'Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance.\n' +
                'AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]\n' +
                'The AMI returns a value of 1 when the two partitions are identical (ie perfectly matched).\n' +
                'Random partitions (independent labellings) have an expected AMI around 0 on average hence can be negative.')
        elif self.plot_name == 'Adjusted Rand Score':
            print(
                'The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and \n' +
                'counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.\n' +
                'ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)\n' +
                'Though the Rand Index may only yield a value between 0 and +1\n' +
                'the adjusted Rand index can yield negative values if the index is less than the expected index.')
        elif self.plot_name == 'Fowlkes Mallows Score':
            print(
                'The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of the precision and recall::\n' +
                'FMI = TP / sqrt((TP + FP) * (TP + FN))\n' +
                'Where ``TP`` is the number of True Positive  (i.e.the number of pair of points that belongs in the same clusters in both ``labels_true`` and``labels_pred``)\n' +
                '``FP`` is the number of False Positive ** (i.e.the number of pair of points that belongs in the same clusters in``labels_true`` and not in ``labels_pred``)\n' +
                'and ``FN`` is the number of FalseNegative ** (i.e the number of pair of points that belongs in thesame clusters in ``labels_pred`` and not in ``labels_True``)\n' +
                'The score ranges from 0 to 1.\n' +
                'high value indicates a good similarity between two clusters.')
        elif self.plot_name == 'Homogeneity Score':
            print(
                'A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.\n' +
                'Homogeneity is a measure of the ratio of samples of a single class pertaining to a single cluster. The fewer different classes included in one cluster, the better.\n' +
                'The lower bound should be 0.0 and the upper bound should be 1.0 (higher is better)')
        elif self.plot_name == 'Completeness Score':
            print(
                'A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.')
        elif self.plot_name == 'V Measure Score':
            print('The V-measure is the harmonic mean between homogeneity and completeness::\n' +
                  'v = (1 + beta) * homogeneity * completeness\n' +
                  '    / (beta * homogeneity + completeness)')
        elif self.plot_name == 'homogeneity_completeness_v_measure':
            print('Compute the homogeneity and completeness and V-Measure scores at once.')
        elif self.plot_name == 'Davies Bouldin Score':
            print(
                'The score is defined as the average similarity measure of each cluster with its most similar cluster, \n' +
                'where similarity is the ratio of within-cluster distances to between-cluster distances.\n' +
                'The minimum score is zero, with lower values indicating better clustering.\n' +
                'The minimum score is zero, with lower values indicating better clustering.'
            )
        elif self.plot_name == 'Calinski Harabasz Score':
            print(
                'It is also known as the Variance Ratio Criterion\n' +
                'The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.\n' +
                'The higher the score , the better the performances.')
        
        
        
        
        

        else:
           raise Exception('Invalid interpretation keyword.')
# if __name__ == '__main__':
#     ex = Explainer('EPTR')
