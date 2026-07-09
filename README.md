# LLMs for GSPs

Code and data for "Improving Conservation Efficiency: Accelerating Groundwater Sustainability Plan Reviews Using Large Language Models"

## Repo structure

```
GSP_Drafts/
  Rubrics/       scoring rubric CSVs for all 65 basins
  *.pdf          trial GSP PDFs (Big Valley, Butte, ECC, Fillmore, Sonoma, SLO)

code/
  run_*.py                 eval scripts for each model
  make_*_fig.py            figure generation scripts
  variance_experiment.py   repeated-inference variance check
  GSP_All.ipynb            main analysis notebook

results/
  results_*.csv                    per-question model outputs, all 9 models
  checkpoint_*.json                resumable checkpoints
  gpt41ft_allgsps_per_gsp_metrics.csv   per-GSP accuracy + AUC for all 62 GSPs
  variance_metrics_by_gsp.csv           run-to-run variability of the best model

images/
  roc_prc_comparison.png
  binary_accuracy_{overall,by_gsp}.png
  class_accuracy_{overall,by_gsp}.png
  class_recall_comparison.png
  confusion_matrices.png
```

## Results summary

| Model | Binary Acc. | ROC AUC | PRC AUC |
|---|---|---|---|
| GPT-4o (base) | 73.9% | 0.711 | 0.590 |
| GPT-3.5 FT | 71.4% | 0.693 | 0.631 |
| GPT-4o FT | 76.8% | 0.770 | 0.696 |
| GPT-4.1 (base) | 73.0% | 0.759 | 0.639 |
| GPT-5.5 (base) | 71.8% | 0.723 | 0.632 |
| o3 | 75.5% | 0.762 | 0.636 |
| Claude Sonnet 4.6 | 75.5% | 0.749 | 0.639 |
| Claude Opus 4.7 (vision) | 73.9% | 0.736 | 0.616 |
| **GPT-4.1 FT** | **75.9%** | **0.773** | **0.711** |

Trial GSPs (n=241 after excluding NotApplicable rows). Binary = Yes vs. {Somewhat + No}.

GPT-4.1 FT applied to all 62 California GSPs: **75.6% accuracy** (57 non-trial GSPs), mean ROC AUC = 0.767.

## Notes

Install dependencies with `pip install -r requirements.txt`. API keys are not included: the scripts read `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from the environment, and the fine-tuned model can be overridden with the `GSP_FT_MODEL` variable. Large embedding caches, model weights, and most GSP PDFs are excluded from this repo via `.gitignore`. See the paper for the full data pipeline.

## License

Code is released under the MIT License (see `LICENSE`). The Groundwater Sustainability Plans and their scoring rubrics are public records from the California Department of Water Resources; the rubric annotations in this repository are our own.
