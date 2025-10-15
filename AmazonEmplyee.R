## Logistic Regression


library(tidymodels)
submission_ex <- vroom("~/Downloads/AmazonEmployeeAccess/amazon-employee-access-challenge/sampleSubmission.csv")

#Define Model
logRegModel <- logistic_reg() %>%
  set_engine("glm")

#Create workflow and fit
logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=amazonTrain)

#Make Predictions
amazon_predictions <- predict(logReg_workflow, new_data = amazonTest, type = "prob")
#amazon_predictions <- predict(logReg_workflow, new_data = amazonTest, type = "class")


#Prepare for Kaggle
amazon_results <- amazonTest %>%
  bind_cols(amazon_predictions)

kaggle_submission_amazon <- amazon_results %>%
  select(id, .pred_1) %>%
  rename(ACTION=.pred_1) %>%
  #rename(ACTION=.pred_class)
  mutate(ACTION=as.numeric(format(ACTION))) 


vroom_write(x=kaggle_submission_amazon, file="./LogisticRegression2.csv", delim=",")



## Penalized Logistic Regression


library(embed)
library(tidymodels)

my_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate(across(everything(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = "ACTION")

my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 4) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazonTrain, v = 4, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- tune::select_best(CV_results, metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
amazon_predictions <- final_wf %>%
  predict(new_data = amazonTest, type="class")

amazon_results <- amazonTest %>%
  bind_cols(amazon_predictions)

kaggle_submission_amazon <- amazon_results %>%
  select(id, .pred_class) %>% #change ".pred_class" to ".pred_1" if using "prob" instead of "class"
  #rename(ACTION=.pred_1) 
  rename(ACTION=.pred_class)%>%
  mutate(ACTION=as.numeric(format(ACTION))) 


vroom_write(x=kaggle_submission_amazon, file="./PenalizedLogReg1.csv", delim=",")