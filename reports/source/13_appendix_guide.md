# *Appendix II: User Guide to the Application

The application is started with one command, `streamlit run app/dguard_app.py`, and opens in a web browser. The first load takes about twenty seconds while the six specialists and the calibration are loaded; after that, one assessment takes about a second. Every result is computed live by the trained models. The sidebar switches between seven views, and any view can be opened directly with a link such as `?view=research`.

**Patient view.** A patient registers (name, age, sex, skin tone), opens a case for one spot, and submits evidence: either their own photographs or, for demonstration, a real held-out MILK10k lesion that the model never saw in training (its true diagnosis is shown to the user but hidden from the system). Two switches make it easy to see the system ask for evidence: leaving "include the dermoscopy photo" unticked lets the system request it when needed, and unticking "include patient details" makes it ask for them with a small form. The result appears as a coloured badge (HIGH, LOW, ACQUIRE or DEFER), a plain-language message, a calibrated confidence bar, any warning signs found in the patient's own words, and an expandable "why this result" explanation. If the photograph failed the quality check, a box to retake it appears. The messages panel shows the full conversation with filters, coloured tags for each kind of message, one-click questions and a counter of unanswered questions; the conversation can be cleared. Under "My data", a case or the whole patient record can be deleted.

**Doctor view.** The review queue lists every case that needs a human, most urgent first: deferred cases to review, LOW answers to confirm, and HIGH answers awaiting sign-off. For each case, six tabs show the assessment (images, the doctor-facing summary, calibrated confidence, probability of malignancy, ignorance, conflict and the differential), the explanation, the specialist team and the gatekeeper with every safety check marked as passed or fired, similar confirmed past cases and the written report, the conversation with quick-reply templates, and the timeline of every event. The doctor confirms, overrides with their own diagnosis, or asks to see the patient in person, and can add an instruction and a follow-up date.

![Doctor view: the prioritised review queue and the evidence for one case.](reports/screens/doctor.png)

**Compare systems view.** "One lesion, every system" runs a chosen held-out lesion through each specialist alone, every fusion rule, and every decision system (always answer, confidence threshold, learned rejector and DERMA-GUARD), and shows what each said, whether it was right, its confidence, the probability it gave the true diagnosis, whether it answered or abstained, and the weights it gave each specialist. "Test-set comparison" shows the same systems over all 787 test lesions, including the fusion rules' validation and test scores, the oracle, the subset study and each specialist alone. "Evaluation methods explained" describes every metric used in the project.

**Stress Lab.** A lesion's photograph can be damaged (blur, darkness, noise, JPEG compression, occlusion, or removal of the patient details) and assessed side by side with the original through the full pipeline. The example below shows the central finding of DERMA-STRESS: blurring makes the model *more* confident (ignorance falls from 0.69 to 0.54 and confidence rises to 91%), and only the quality check stops it from giving a HIGH answer.

![Stress Lab: the same lesion, original and blurred. Blur lowers ignorance and raises confidence; the quality gate fires and the answer is downgraded to LOW.](reports/screens/stress.png)

**Research view.** Every experimental result: headline numbers with confidence intervals, paired tests, the fusion tables and what the learned gate relies on, the decision policy against baselines at three costs, DERMA-STRESS, both external tests, the re-evaluation of the earlier prototype's models, the warning-sign analysis and the skin-tone breakdown.


**Architecture view.** The pipeline drawn as a diagram, and a table linking each component to its source file.


**System view.** Platform statistics, the audit log and a button to reset the demonstration database, which asks for confirmation and deletes every patient, case, message and uploaded photograph (never the dataset images).


**REST API.** The same workflow is available programmatically with `uvicorn src.api:app --port 8000`; interactive documentation is served at /docs. The endpoints are listed in Table 5.4.

# *Appendix III: Reproducing the Results

Every number in this report can be regenerated from the repository. The MILK10k images must be downloaded from the ISIC Archive into `milk10k/`, the trained network's checkpoint placed in `runs/milk10k_resnet50_seed42/checkpoints/`, and, for the external tests, PAD-UFES-20 and Derm7pt obtained from their authors. The commands below were run on a laptop CPU (Windows 11, Python 3.13); training the network itself is done once on Kaggle with `notebooks/kaggle_train_dguard.py`.

```
pip install -r requirements.txt
python scripts/milk_native_infer.py            # trained network on validation and test
python scripts/build_specialists.py            # all specialists, embeddings, image statistics
python scripts/run_team.py                     # 7 fusion rules, cross-fitted; subset study
python scripts/run_team.py --method adaptive_em --out outputs/milk_team_em --no-subsets
for c in 2 3 5 10; do
  python scripts/run_eade_milk.py --native outputs/milk_team --missed-cost $c --out outputs/eade_team/cost_$c
done
python scripts/run_eade_milk.py --native outputs/milk_team_em --missed-cost 5 --out outputs/eade_team_em/cost_5
python scripts/analyze_milk.py                 # confidence intervals, tests, rejector, fairness
python scripts/run_stress_milk.py --out outputs/stress_milk/report_team.json
python scripts/run_shift_pad.py                # PAD-UFES-20
python scripts/run_shift_derm7pt.py --data dataset/release_v0
python scripts/legacy_pad_models.py            # the earlier prototype's seven models
python scripts/analyze_pad_symptoms.py         # warning signs on the PAD-UFES-20 questionnaire
python scripts/make_figures.py                 # every figure
pytest tests -q                                # 230 tests
```

The five-specialist results in Table 8.5 come from the same scripts with `--no-s6`. Additional training seeds are processed with `python scripts/run_seeds.py runs/<seed folder> ...`, which repeats the whole pipeline and reports the mean and standard deviation of every headline number.

Table: Where each result in Chapter 8 is produced.
| Result | Script | Output file |
|---|---|---|
| Specialists, fusion rules, subsets, gate weights | run_team.py | outputs/milk_team/team_report.json |
| Decision policy, baselines, tiers, costs | run_eade_milk.py | outputs/eade_team/cost_*/report.json |
| Confidence intervals, paired tests, rejector, skin tone | analyze_milk.py | outputs/analysis/analysis.json |
| DERMA-STRESS | run_stress_milk.py | outputs/stress_milk/report_team.json |
| PAD-UFES-20 shift | run_shift_pad.py | outputs/shift_pad/report.json |
| Derm7pt external test | run_shift_derm7pt.py | outputs/shift_derm7pt/report.json |
| Earlier prototype's models | legacy_pad_models.py | outputs/legacy_pad/report.json |
| Warning signs | analyze_pad_symptoms.py | outputs/analysis/pad_symptoms.json |
