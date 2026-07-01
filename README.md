# Exposition-to-NDEs
This is a documentation of learning about the fundamentals for, the most recent progress of, and the applications surrounding neural differential equations.

There are currently some issues (seemingly insurmountable) with KaTeX rendering on this repo. For proper rendering, download the pertinent files onto your VSCode (with the appropriate Markdown extensions) or another IDE.


REMARK: use NDEs as training data generator (e.g. input: irregularly sampled; trained NDE: outputs regularly sampled data faithful to the original data; some interpretable model: trained on this regular data)

The summer project (UROP) concerns itself with benchmarking neural SDE-based models with traditional volatility models (GBM, Heston, GARCH-type, etc) in volatility modelling and/or return prediction. Classical models are good in the sense
1. interpretable param
2. param are easy to calibrate given just a small amount of data
3. rigorous mathematical foundation and decades of underlying research
Their biggest shortcomings, however, are their limited expressivity (vulnerable to complex regime change) and lack of non-ad hoc framework for model selection. (Also Knightian uncertainty?)

On the other hand, ML-based approaches are expressive and, by design, data-driven (adaptive). Nevertheless, their black-box nature, data-hunger, and empiricism (lack of both standardized benchmarks and theoretical guarantees) are characteristics awaiting further reconciliation. Neural DEs is a step towards achieving this goal.
<img width="569" height="409" alt="image" src="https://github.com/user-attachments/assets/ac18470e-6555-4798-b17a-fe5a2cc0774a" />


The central question to ask across all applied scenarios in quantitative finance is whether modelling capacity, which often comes with the black box cost, has greater significance than interpretability. This underpins the competition between ML-based and classical approaches.

