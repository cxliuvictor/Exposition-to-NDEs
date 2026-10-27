# Research Project / Mock Scoring Rubric (100)

## Problem framing — 10
- unit of observation and target/action clear
- prediction timestamp / information set clear
- appropriate metric(s)

## Data audit and cleaning — 15
- missing/duplicates/types/constants/outliers
- leakage explicitly audited
- merge/cardinality handled correctly

## EDA and hypotheses — 15
- plots answer questions, not decoration
- transformations motivated
- no giant unfocused EDA dump

## Validation design — 20
- split matches data structure
- preprocessing fit on train only
- baseline established
- OOS test protected
- CV/fold stability considered

## Modelling — 15
- simple model before complex model
- tuning restrained
- model choice justified
- residual / bucket diagnostics

## Decision / strategy translation — 10
- threshold/ranking/portfolio rule validated
- costs/turnover/tail behavior where relevant

## Robustness — 5
- time/regime/bucket sensitivity
- shift / concentration / failure modes

## Presentation — 10
- concise story
- decisive figures/tables
- limitations acknowledged
- able to defend choices

### Automatic major penalties
- future-data leakage
- test-set tuning
- misaligned financial return target
- accidental many-to-many merge
- reporting only training performance
