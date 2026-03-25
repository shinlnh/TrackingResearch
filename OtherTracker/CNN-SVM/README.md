CNN-SVM tracker status

Original release status:

- No original runnable source code is present under `OtherTracker/CNN-SVM`.
- Embedded OTB results exist locally under `otb/otb-toolkit/results/OPE/*_CNN-SVM.mat`.
- Embedded result files only expose `res`, `len`, and `type`; they do not contain `fps` or per-frame timing.
- The original project page only ships paper, bibtex, supplementary, poster, and benchmark result zip. It does not provide a code package.

Approximate local reimplementation:

- `run_cnnsfm_approx_otb.py` benchmarks a report-oriented approximation named `CNN-SVM-Approx`.
- `cnnsfm_approx.py` implements the approximate tracker with:
  - ImageNet-pretrained AlexNet `fc6` features
  - online linear hinge classifier via `SGDClassifier`
  - target-specific saliency by backpropagating positive classifier dimensions
  - recent saliency-template matching for localization
- `eval_cnnsfm_approx_otb.py` evaluates the exported OTB txt results with success AUC and precision@20.

Important caveat:

- `CNN-SVM-Approx` is intended for internal comparison and reporting only.
- It is not the original paper code, and the online SVM solver / Bayesian filtering are simplified for practicality.
