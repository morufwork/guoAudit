"""Close the Introduction/Methods/Results audit using frozen analysis outputs.

Run once on the reporting-completion V2 sources, then run scripts 54 and 55.
Original documents and renders are preserved before any manuscript mutation.
"""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
main_path = ROOT / 'manuscript/FINAL_MANUSCRIPT_V2.md'
supp_path = ROOT / 'manuscript/SUPPLEMENTARY_INFORMATION_V2.md'
m, s = main_path.read_text(), supp_path.read_text()
assert 'Supplementary Table S45.' not in s, 'Coverage revision already applied'


def replace_paragraph(text, prefix, replacement):
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines[hits[0]] = replacement
    return '\n'.join(lines) + '\n'


def insert_after(text, prefix, addition):
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines.insert(hits[0] + 1, '\n' + addition)
    return '\n'.join(lines) + '\n'


def read_table(name):
    return pd.read_csv(ROOT / 'tables' / f'{name}.csv')


negative = read_table('table7_negative_sampling_robustness')
seeds = read_table('negative_sampling_seed_sensitivity_summary')
novelty = read_table('tableS23_figure3_data')
architecture = read_table('table6_architecture_ablation')
tests = read_table('statistical_test_family')
operating = read_table('threshold_and_operating_point_metrics')


def swap_auc(train, test, design='A'):
    row = negative.loc[
        negative.experiment.str.startswith(design)
        & negative.trained_on.eq(train) & negative.evaluated_on.eq(test), 'roc_auc']
    return f'{row.item():.3f}'


def seed_summary(model):
    row = seeds.loc[seeds.split.eq('both_unseen') & seeds.model.eq(model)].iloc[0]
    return f'{row.roc_auc_mean:.3f} ± {row.roc_auc_std:.3f}'


# Preserve the current delivery, including its pagination, for comparison.
backup = ROOT / 'revision/coverage_before'
backup.mkdir(exist_ok=False)
originals = [main_path, supp_path]
originals += [ROOT / 'manuscript' / f'{name}.docx'
              for name in ['final_manuscrip_v2', 'supplementary_v2']]
originals += [ROOT / 'revision/rendered' / f'{name}.{ext}'
              for name in ['final_manuscrip_v2', 'supplementary_v2']
              for ext in ['pdf', 'txt']]
hashes = {}
for path in originals:
    shutil.copy2(path, backup / path.name)
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(backup / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')

# Methods: identify the two distribution-shift directions and their separate
# relationship to consistent-negative revalidation.
m = m.replace('Supplementary Sections S1–S22.', 'Supplementary Sections S1–S23.')
m = insert_after(m, 'For R3, negative construction occurred',
    'Negative-set effects were evaluated using complementary designs. In the distribution-shift experiment, positive pairs were partitioned once into 80% training and 20% test subsets (seed 42), and each negative set was partitioned separately using the same proportions and seed. An ESM-2–MLP trained on N0 was evaluated after replacing test negatives with N1, N2, N3, or N5; in the reverse direction, models trained on each negative set were evaluated on the common N0 test set. Positive partitions were shared across these comparisons. These designs assess sensitivity to the training or evaluation negative definition. Separate R0/R3 revalidation used the same negative definition in both training and evaluation. Full distribution-shift results are reported in Supplementary Section S23 and Table S45.')
m = m.replace('Primary discrimination metrics were ROC-AUC and PR-AUC;',
              'Primary discrimination metrics were ROC-AUC and PR-AUC, with PR-AUC calculated as average precision;')
m = replace_paragraph(m, 'The multi-species specificity analysis applied',
    'The multi-species specificity analysis applied analogous topology-only, sequence, and degree-matching comparisons to a separately constructed dataset comprising three species. It tested whether degree-associated separability and the complete Guo sequence-model collapse reproduced together. Following accession-level taxonomy resolution, the analyses were repeated within C. elegans, D. melanogaster, and E. coli, with degrees and matching strata recomputed within each species and replacement endpoints restricted to that species. Each replication used an independently reconstructed 80/20 random pair split (seed 42). A same-species indicator assessed the contribution of taxonomic pairing to pooled discrimination. Construction, exclusions, realized balance, and matching sensitivity are detailed in Supplementary Sections S16 and S20 and Tables S21–S22 and S25–S33. Reproducibility information and provenance are provided in Supplementary Sections S17–S18.')

m = replace_paragraph(m, "**Figure 7.", '**Figure 7. Degree-only and sequence-model sensitivity to the negative definition.** (A) Models trained on N0 are evaluated after replacing test negatives. (B) Models trained on each negative set are evaluated on the common N0 test set. Degree-only logistic regression reproduces strong N0 discrimination and approaches chance under degree-matched sets. Sequence-model results use ESM-2–MLP. Numerical results are reported in Supplementary Tables S34 and S45; length-only specificity controls are reported separately in Table S34.')

# Results: provide PR-AUC alongside the matched novelty trajectory.
pr = [novelty.loc[novelty.Regime.eq(r) & novelty.Model.eq('MLP'), 'PR-AUC'].item()
      for r in ['R1', 'R2', 'R3']]
m = m.replace('Within the genuinely matched R1→R2→R3 curve', 'Within the matched R1→R2→R3 curve')
m = m.replace('MCC declined from 0.305 to 0.191 and 0.129.',
              f'MCC declined from 0.305 to 0.191 and 0.129. PR-AUC followed the same direction, decreasing from {pr[0]:.3f} at R1 to {pr[1]:.3f} at R2 and {pr[2]:.3f} at R3 (Table S23).')

granularity = ('In Guo, refining the degree-matching rule also left the N0-trained degree-only logistic classifier near chance: ROC-AUC was 0.502 with decile matching and 0.499 with either exact-degree or nearest-degree matching (Table S7). Thus, its loss of discrimination was not dependent on the use of coarse degree strata.')
m = insert_after(m, 'Additional controls localized', granularity)

m = replace_paragraph(m, 'Two complementary designs examined',
    'Sequence-model discrimination depended on the negative definition in both directions of the distribution-shift experiment. With N0 training and replacement of test negatives, ESM-2–MLP ROC-AUC decreased from '
    f'{swap_auc("n0", "n0")} on N0 to {swap_auc("n0", "n1")} on random unobserved pairs (N1), {swap_auc("n0", "n3")} on length-matched pairs (N3), {swap_auc("n0", "n2")} on degree-matched pairs (N2), and {swap_auc("n0", "n5")} on jointly matched pairs (N5). In the reverse direction, training on N1 or N3 and evaluating on the common N0 test set yielded ROC-AUC {swap_auc("n1", "n0", "B")} and {swap_auc("n3", "n0", "B")}, respectively (Table S45). Random resampling and length matching therefore reduced N0-trained discrimination, while degree-based matching produced the lowest point estimates. These comparisons combine negative-set control with a change between the training and evaluation distributions.'
    '\n\nWhen each negative definition was instead used consistently during both training and evaluation, Guo R0/N2 ROC-AUC was 0.455 for AAC+CTD and 0.463 for ESM-2 mean; R3/N2 values were 0.517 for both (Table S17). The R0-versus-R3 gap and the ESM-2-versus-AAC+CTD advantage apparent under N0 therefore did not persist under degree matching.')
pr_n2 = architecture.loc[architecture.negative_set.eq('n2'), 'pr_auc']
m = m.replace('Across five training seeds, means remained between 0.500 and 0.525, with overlapping variability.',
    f'PR-AUC ranged from {pr_n2.min():.3f} to {pr_n2.max():.3f}, close to the test-set positive prevalence of 0.500 (Table S4). Across five training seeds, mean ROC-AUC remained between 0.500 and 0.525, with overlapping variability.')

boot_p = tests.loc[tests.test_id.eq('Q1-roc_auc-n0-siamese_mlp-bootstrap'), 'p_value_adjusted'].item()
jack_p = tests.loc[tests.test_id.eq('Q1-roc_auc-n0-siamese_mlp-jackknife'), 'p_value_adjusted'].item()
m = replace_paragraph(m, 'Component-aware inference supported',
    'Component-aware inference supported selected N0 above-chance results, with conclusions dependent on the metric and resampling method. For example, the N0 Siamese ROC-AUC of 0.651 exceeded chance after bootstrap correction '
    f'(adjusted P = {boot_p:.3f}), but not after jackknife correction (adjusted P = {jack_p:.3f}; Table S19). The degree-only controls in Section 3.4 provide algorithmic evidence of topology sufficiency and were not included in this 44-test family. N0 and N2 strict test graphs contained only 19 and 5 usable connected components, respectively, with most pairs in one giant component. Bootstrap and jackknife intervals were consequently wide and sometimes divergent. After Benjamini–Hochberg correction, no tested architecture had evidence of above-chance discrimination under N2 with either method.')
m = replace_paragraph(m, 'The N0–N2 point-estimate difference was large',
    'The N0–N2 point-estimate difference was large and directionally consistent across architectures and seeds, but the corrected cluster-aware comparison did not reach conventional significance. This limits the inferential conclusion without establishing the absence of a difference. Across 30 N2 negative-generation seeds, R3 ROC-AUC was '
    f'{seed_summary("logistic_regression")} for ESM-2 logistic regression, {seed_summary("mlp")} for ESM-2–MLP, and {seed_summary("degree_only_lr")} for the degree-only control (mean ± SD; Table S8). Together with the five training-seed analyses, these results show algorithmic stability on the same underlying graph; they do not add independent biological replicates. Supplementary Sections S8 and S14–S15 and Tables S8 and S18–S20 report the full seed and component-aware analyses.')
m = m.replace('Leave-one-component-out results are now reported', 'Leave-one-component-out results are reported')

# Threshold and operating-point summaries refer to different saved fits.
op = operating.loc[operating.negative_set.eq('n2') & operating.model.eq('mlp')].iloc[0]
precision = operating.loc[operating.negative_set.eq('n2'), 'precision_at_recall_0.5']
threshold_text = (
    'Improvements in probability reliability did not consistently improve threshold performance. For the R3/N2 MLP at the prespecified threshold of 0.5, MCC was '
    f'{op["mcc_at_0.5_raw"]:.4f} before calibration, {op["mcc_at_0.5_platt"]:.4f} after Platt scaling, and {op["mcc_at_0.5_isotonic"]:.4f} after isotonic regression (Table S13). In the separate architecture experiment, precision at recall 0.5 ranged from {precision.min():.3f} to {precision.max():.3f} across R3/N2 models, close to the positive prevalence of 0.500. These operating points describe the saved test curves and do not validate prospective thresholds.')
m = insert_after(m, 'Under R3/N2, isotonic-calibrated', threshold_text)

# Supplement: make the outcomes explicit where their procedures are described.
s = insert_after(s, 'N6 and N7 each contained',
    'The complete ESM-2–MLP comparison across N0, N1, N2, N3, and N5, including both training/evaluation directions, is reported in Section S23 and Table S45. These distribution-shift comparisons are distinct from the consistent-negative R0/R3 revalidation in Table S17.')
s = replace_paragraph(s, 'Thirty independently generated negative sets',
    'Thirty independently generated N2 negative sets were evaluated with the model-training seed and split fixed, separating sensitivity to decoy selection from sensitivity to model initialization. Under R3, mean ROC-AUC was '
    f'{seed_summary("logistic_regression")} for ESM-2 logistic regression, {seed_summary("mlp")} for ESM-2–MLP, and {seed_summary("degree_only_lr")} for the degree-only logistic control (mean ± SD; Table S8). These estimates support stability of the aggregate near-chance pattern, although individual sequence-model draws varied. The runs reuse the same biological benchmark and do not increase its independent biological sample size.')
s = insert_after(s, 'Table S12 contains component-aware', threshold_text)
s = s.replace('Table S13 now compares', 'Table S13 compares')
s = insert_after(s, 'Inference used connected-component',
    f'For N0 Siamese ROC-AUC, the adjusted bootstrap P value was {boot_p:.3f}, whereas the adjusted jackknife P value was {jack_p:.3f}. This example illustrates the dependence of above-chance conclusions on the inferential method. The 44-test family concerns the six sequence architectures and their specified contrasts; it does not test the degree-only controls in Table S34. Those controls provide descriptive and algorithmic evidence of topology-associated separability.')
s = insert_after(s, '## S10 Exact and nearest degree matching', granularity)

# Include every recorded metric, retaining all ten rows and both N0 baselines.
rename = {'experiment': 'Design', 'trained_on': 'Training set', 'evaluated_on': 'Test set',
          'roc_auc': 'ROC-AUC', 'pr_auc': 'PR-AUC', 'mcc': 'MCC', 'f1': 'F1',
          'balanced_accuracy': 'Balanced accuracy', 'accuracy': 'Accuracy',
          'precision': 'Precision', 'recall': 'Recall', 'specificity': 'Specificity',
          'brier_score': 'Brier score', 'ece': 'ECE'}
columns = ['Design', 'Training set', 'Test set', 'ROC-AUC', 'PR-AUC', 'MCC', 'F1',
           'Balanced accuracy', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'Brier score', 'ECE']
printed = negative.rename(columns=rename)[columns].copy()
printed['Design'] = printed.Design.map({'A_train_n0_eval_swap': 'A', 'B_train_each_eval_n0': 'B'})
assert len(printed) == 10 and printed.notna().all().all()
table_path = 'tables/tableS45_negative_set_sequence_comparison.csv'
printed.to_csv(ROOT / table_path, index=False)
note = ('Frozen mean-pooled ESM-2 representations, combined symmetric pair features, and the MLP classifier were used throughout. Design A trains on N0 and replaces the test negatives; design B trains on each negative set and evaluates on the common N0 test set. Positive train/test partitions are shared across sets (80/20; seed 42); each negative set is split separately with the same proportions and seed. N0 denotes original benchmark negatives; N1, random unobserved pairs; N2, degree-matched pairs; N3, length-matched pairs; and N5, jointly degree–length-matched pairs. The N0/N0 baseline is repeated to retain each design\'s reference. All metrics are descriptive point estimates. PR-AUC is average precision; threshold metrics use 0.5; Brier score and ECE use raw probabilities. These results assess changes in the training or evaluation negative definition, not consistently trained N1/N3 performance. Source: tables/table7_negative_sampling_robustness.csv.')
s += ('\n## S23 Sequence-model sensitivity to the negative definition\n\n'
      'Random resampling and length matching reduced sequence-model discrimination relative to the original benchmark negatives. With N0 training, ESM-2–MLP ROC-AUC was '
      f'{swap_auc("n0", "n1")} on N1 and {swap_auc("n0", "n3")} on N3, compared with {swap_auc("n0", "n0")} on N0, {swap_auc("n0", "n2")} on N2, and {swap_auc("n0", "n5")} on N5. Training on N1 or N3 and evaluating on the common N0 test set yielded {swap_auc("n1", "n0", "B")} and {swap_auc("n3", "n0", "B")}, respectively. Both directions show dependence on the negative definition; neither isolates a causal effect of degree or length. The consistent-negative revalidation is reported separately in Table S17.\n\n'
      '**Supplementary Table S45. Sequence-model performance across negative-set definitions and training/evaluation directions.**\n\n'
      f'[CSV:{table_path}]\n\n{note}\n')

main_path.write_text(m)
supp_path.write_text(s)
manifest_path = ROOT / 'revision/reporting_completion_tables.json'
manifest = json.loads(manifest_path.read_text())
manifest.append([45, 'Sequence-model performance across negative-set definitions and training/evaluation directions', table_path, note])
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print('Updated both V2 sources and added the complete 10-row Table S45; originals preserved in', backup)
