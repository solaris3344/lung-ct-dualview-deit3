# V3.6 method freeze / test registration (written before TEST)

- **Development selection**: V3.4 20 epoch / three seeds (42,52,62), 558 train+valid images, 508 image-similarity groups, best exploratory accuracy 539/558 (96.59%). **Do not claim patient-disjoint CV.**
- **Reason for final refit**: source V3.4 training saved per-fold predictions by default but not weights. No real 288-image inference can be made from OOF arrays.
- **Final model**: DeiT3 Base, timm `deit3_base_patch16_224.fb_in22k_ft_in1k`, pretrained then fine-tuned 20 epoch on all DEV 558, 3 seeds. V3.4-style dual-view global + cross-fitted selected geometric crop. This is a deployment refit, not exactly the 15 heldout fold estimators.
- **Region-selector**: 7 ImageNet frozen feature views (global plus 6 geometric crops), logistic regressions C=.03 with standardization and balanced weights; 3-fold stratified-group out-of-fold selection for DEV training examples; six predictors refit on full DEV for TEST view selection.
- **Training**: ImageNet normalization, RGB conversion, bicubic resize 224×224; augmentation max 7° affine / 3% translation / scale 0.97–1.03; AdamW wd .02; LR backbone 3e-5/head 3e-4; cosine annealing over 20 epochs; global/region weighted CE .8/.2; CE class weights sqrt(n/(4*n_class)); batch2, accumulation4; epoch 1 backbone frozen; AMP; gradient clipping 2.0; gradient checkpointing.
- **Final probability**: arithmetic mean over seed models for global softmax and region softmax; `0.85 * P_global + 0.15 * P_region` (NO adjustments on TEST).
- **288 source TEST**: test exactly once. Report four-class top-1 accuracy, macro-F1, balanced accuracy, confusion, Wilson 95% accuracy interval, class recalls, candidate visual duplicates.
- **External ADC/SCC**: only explicitly curated, patient-labelled ADC/SCC BT PNG/JPG; per-patient average probabilities and conditional ADC/SCC scores, not four-class external validation. Note imaging and windowing shift.
- **No TTA**, no normalization fit on TEST, no threshold chosen on TEST. No independent hospital/patient validation unless provenance proves it.
