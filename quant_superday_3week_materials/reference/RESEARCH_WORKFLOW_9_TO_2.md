# Dataset-interview workflow: 09:00 → 14:00

## 09:00–09:20 — understand before modelling
- unit of observation
- target / action to optimize
- shape, types, missingness, duplicates
- time/order/group structure
- quick target + numeric plots
- leakage risks
- decide validation design

## 09:20–10:00 — hypotheses + baseline
- clean obvious issues
- simple feature transformations
- establish naive/statistical baseline
- identify 2–4 promising hypotheses

## First interviewer checkpoint
Be ready to say:
1. what the dataset appears to contain;
2. the main data-quality/leakage issue;
3. what simple baseline you have;
4. what relationships look promising;
5. how you will validate;
6. what you will do next.

## 10:15–12:00 — serious modelling
- simple linear/regularized model
- one nonlinear sklearn model
- limited hyperparameter search
- CV appropriate to data structure
- begin residual/bucket/regime diagnostics

## 12:00–13:00 — lunch
Take an actual break if possible.

## 13:00–13:35 — robustness / economic translation
- error by bucket/regime/time
- feature importance
- threshold/ranking/strategy if relevant
- costs/turnover where relevant
- frozen OOS test

## 13:35 — stop adding models
Choose the story.

## 13:35–14:00 — presentation
Aim for 4–6 slides:
1. objective + dataset
2. EDA/data issues
3. method + validation
4. main results
5. robustness/strategy
6. limitations + next steps
