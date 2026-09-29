"""Close manuscript reporting gaps from frozen outputs; no model training.
Run once against the pre-completeness V2 Markdown files. Backups are required.
"""
from pathlib import Path
import json, re, hashlib, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/manuscript-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
main=Path('manuscript/FINAL_MANUSCRIPT_V2.md')
supp=Path('manuscript/SUPPLEMENTARY_INFORMATION_V2.md')
m=main.read_text(); s=supp.read_text()
assert 'S21 Additional benchmark controls' not in s, 'Already revised'

def replace_para(text,prefix,new):
 lines=text.splitlines(); hits=[i for i,line in enumerate(lines) if line.startswith(prefix)]
 assert len(hits)==1,(prefix,hits)
 lines[hits[0]]=new
 return '\n'.join(lines)+'\n'

def add_after(text,prefix,new):
 lines=text.splitlines(); hits=[i for i,line in enumerate(lines) if line.startswith(prefix)]
 assert len(hits)==1,(prefix,hits)
 lines.insert(hits[0]+1,'\n'+new+'\n')
 return '\n'.join(lines)+'\n'

# Post-audit chart: counts and distributions are recomputed from canonical inputs.
pairs=pd.read_csv('data/processed/ppi_pairs_clean.tsv',sep='\t',header=None,names=['a','b','label'])
proteins=pd.read_csv('data/processed/proteins_with_groups.tsv',sep='\t')
assert len(pairs)==11164 and not pairs.duplicated(['a','b']).any()
assert (pairs.a<pairs.b).all() and set(pairs.label)=={0,1}
assert set(pairs.a)|set(pairs.b)==set(proteins.protein_id)
pos=pairs[pairs.label==1]; degrees=pd.concat([pos.a,pos.b]).value_counts().reindex(proteins.protein_id,fill_value=0)
lengths=proteins.sequence.str.len()
assert (degrees==0).sum()==280 and (lengths>1022).sum()==268
assert pairs.label.value_counts().to_dict()=={0:5583,1:5581}
audit=proteins[['protein_id']].copy();audit['length']=lengths;audit['positive_degree']=degrees.to_numpy()
audit.to_csv('tables/yeast_postaudit_protein_distributions.csv',index=False)
classes=pd.DataFrame({'Label':['Positive','Constructed negative'],'Pairs':[5581,5583]})
classes['Percent']=classes.Pairs/len(pairs)*100
classes.to_csv('tables/yeast_postaudit_class_counts.csv',index=False)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(10.8,3.3),layout='constrained')
axes[0].bar(['Positive','Constructed\nnegative'],classes.Pairs,color=['#337ca0','#d78142'])
for i,row in classes.iterrows(): axes[0].text(i,row.Pairs+110,f'{row.Pairs:,}\n({row.Percent:.2f}%)',ha='center',fontsize=8)
axes[0].set_ylim(0,7200);axes[0].set_ylabel('Unique pairs');axes[0].set_title('A  Post-audit class balance',loc='left')
bins=np.geomspace(lengths.min(),lengths.max()+1,24)
axes[1].hist(lengths,bins=bins,color='#337ca0',edgecolor='white',linewidth=.4)
axes[1].set_ylim(0,370);axes[1].set_xscale('log');axes[1].set_xlabel('Sequence length (residues; log scale)');axes[1].set_ylabel('Proteins')
axes[1].axvline(1022,color='#d78142',ls='--',lw=1.3);axes[1].set_title('B  Sequence lengths (n = 2,497)',loc='left')
axes[1].text(.97,.94,'268 proteins >1,022 residues\n(10.73%)',transform=axes[1].transAxes,ha='right',va='top',fontsize=8)
counts=degrees.value_counts().sort_index();axes[2].bar(counts.index,counts.values,color=['#d78142' if x==0 else '#337ca0' for x in counts.index])
axes[2].set_xlabel('Observed positive-network degree');axes[2].set_ylabel('Proteins');axes[2].set_xticks([0,3,6,9,12,15]);axes[2].set_title('C  Degree across all proteins',loc='left')
axes[2].text(.97,.94,'280 proteins with degree 0',transform=axes[2].transAxes,ha='right',va='top',fontsize=8)
fig.savefig('figures/figure_yeast_postaudit.png',dpi=300);fig.savefig('figures/figure_yeast_postaudit.pdf');plt.close(fig)

# Insert Figure 3; shift subsequent main figure labels and references, not asset filenames.
def renumber(text):
 text=re.sub(r'\b(Figure[s]?) (\d+)( and (\d+))?',lambda z:z[1]+' '+str(int(z[2])+(int(z[2])>=3))+((' and '+str(int(z[4])+(int(z[4])>=3))) if z[4] else ''),text)
 return text
m=renumber(m);s=renumber(s)
m=m.replace('### 3.1 Performance declines with protein novelty','### 3.1 Post-audit dataset and protein-novelty performance')
chartpara='The audited yeast dataset contained 11,164 unique undirected pairs: 5,581 positives (49.99%) and 5,583 constructed negatives (50.01%). Duplicate removal retained all 2,497 protein identifiers. Of these, 280 had no observed positive partner, whereas 268 sequences exceeded the 1,022-residue embedding limit. Figure 3 summarizes the post-audit class, length, and degree distributions. Degree zero denotes absence from the observed positive graph, not evidence of biological isolation.'
caption='**Figure 3. Yeast benchmark distributions after the integrity audit.** (A) Class counts after removal of 24 duplicate pairs. (B) Sequence-length distribution across all 2,497 protein identifiers; the dashed line marks the 1,022-residue embedding limit. The horizontal axis is logarithmic, and bars show counts in logarithmically spaced bins. (C) Positive-network degree across the same protein universe, including 280 zero-degree proteins. Each protein contributes once to panels B–C; these distributions differ from the endpoint-occurrence-weighted summaries used in the negative-set audit. Source counts and protein-level values are provided in Supplementary Table S44 and its accompanying data files.'
m=add_after(m,'### 3.1 Post-audit',chartpara+'\n\n[FIGURE:figures/figure_yeast_postaudit.pdf]\n\n'+caption)

# Correct methods to reflect actual analysis scope and documented saved protocols.
m=m.replace('Handcrafted features comprised 20-dimensional amino-acid composition, 147-dimensional composition/transition/distribution descriptors, and eight global physicochemical summaries calculated with Biopython ProtParam.', 'The principal handcrafted representation concatenated 20-dimensional amino-acid composition (AAC) and 147-dimensional composition/transition/distribution (CTD) descriptors. Eight Biopython ProtParam physicochemical summaries were evaluated separately in the conventional five-fold baseline and were not part of AAC+CTD. That baseline used stratified pair-level folds (seed 42); fold standard deviations describe split variability rather than independent biological replication (Supplementary Table S43).')
m=m.replace("N7 added exclusion of candidates belonging to the fixed endpoint's R3-20 MMseqs2 cluster.","N7 added within-pair sequence-cluster exclusion to N5's joint degree–length matching, using the saved 20%-identity cluster assignment. This negative-pair compatibility constraint is distinct from the versioned R3-20 train/test split control.")
m=m.replace('No class weighting was used because all comparison datasets were balanced or nearly balanced.','No class weighting was used. Class counts are reported for each realized dataset; exact-matching omissions can introduce imbalance, particularly in the species-specific analyses.')
m=m.replace('Calibration details and operating points are reported in Supplementary Section S12 and Supplementary Tables S12–S13.','Calibration models reserved 20% of the R3 training pairs for fitting the mappings and were therefore fitted separately from the architecture comparison. Raw and calibrated metrics are compared within the same fitted model. Precision–recall operating points are descriptive summaries of the saved architecture-model test curves at prespecified targets of 0.5; they are not prospectively validated thresholds. Calibration details and operating points appear in Supplementary Section S12 and Tables S12–S13 and S38–S39.')
m=m.replace('Calibration can improve probability reliability or threshold performance, but it cannot restore ranking discrimination when ROC-AUC is approximately 0.5.', 'Calibration can change probability reliability and threshold performance. Isotonic mappings introduce ties, and an unconstrained fitted sigmoid can reverse score ordering; neither change establishes additional interaction information.')
m=m.replace('Full hypotheses, corrections, component analyses, and sensitivity results appear in Supplementary Sections S14–S15 and Supplementary Tables S18–S20.','Full hypotheses, corrections, component analyses, and sensitivity results appear in Supplementary Sections S14–S15 and Tables S18–S20 and S40–S42.')
m=m.replace('Full topology and incremental-feature results are in Supplementary Tables S6–S7 and S9.','Full topology and incremental-feature results are in Supplementary Tables S34–S35; degree-distribution and matching diagnostics are in Tables S5–S7.')
m=m.replace('Corresponding numerical results are reported in Supplementary Tables S6–S7.','Corresponding numerical results are reported in Supplementary Table S34.')
m=m.replace('Numeric results are in Supplementary Tables S8 and S14.','Numeric results are in Supplementary Table S17; generation-seed robustness is reported separately in Table S8.')
m=m.replace('Corresponding values are in Supplementary Tables S3, S6, and S21–S22.','Corresponding values are in Supplementary Tables S23, S34, and S21–S22.')
m=m.replace('Detailed outputs are mapped to Supplementary Sections S1–S18.','Detailed outputs are mapped to Supplementary Sections S1–S22.')

controls='Additional controls localized this dependence more precisely. Under N0 training with test-negative replacement, length-only logistic regression and random forest yielded ROC-AUC 0.540–0.551 on N0 and 0.501–0.506 on N2; degree-only N5 values were 0.501–0.504 (Table S34). Adding degree to ESM-2 pair features gave logistic-regression ROC-AUC 0.928 on N0, 0.488 on N2, and 0.489 on N5 (Table S35). These comparisons support a topology-associated interpretation within the tested benchmark, without isolating a unique causal feature.'
m=add_after(m,'An incremental model combining',controls)
n6='Changing the substitution rule did not restore discrimination. Both N2 and the two-sided N6 set contained 5,581 negatives; the N0-trained ESM-2–MLP yielded ROC-AUC 0.467 on N2 and 0.482 on N6, while degree-only logistic regression yielded 0.502 and 0.503 (Table S36). N7 retained 5,581 degree–length-matched pairs after excluding within-pair cluster matches and produced ROC-AUC 0.494 under N0 training, compared with 0.494 for N5. Training on N7 and testing against N0 yielded 0.478 (Table S37). These are distribution-shift sensitivity analyses, not consistently trained N6/N7 evaluations or evidence of evolutionary non-homology.'
m=add_after(m,'Architecture complexity did not restore',n6)
m=add_after(m,'The N0–N2 point-estimate difference was large', 'The paired Siamese-versus-logistic-regression contrast was 0.047 ROC-AUC under N0 (bootstrap-adjusted P = 0.088; jackknife-adjusted P = 0.999) and 0.012 under N2 (0.261 and 0.999, respectively; Table S40). Across architectures, N0–N2 differences ranged from 0.077 to 0.139, with adjusted P values of 0.130–0.492 for the bootstrap and 0.674–0.999 for the jackknife (Table S41). Leave-one-component-out results are now reported for each model and negative set (Table S42); removing the giant component leaves only 25 N0 pairs or eight N2 pairs, limiting interpretation of the remaining estimates.')

sensitivity='''### 3.9 Representation and degree-source sensitivity

The conventional five-fold physicochemical-only baselines yielded mean ROC-AUC 0.588–0.617 across four classifiers (Table S43). These features were evaluated separately from AAC+CTD and do not account for the reported AAC+CTD results. In the pair-fusion experiment, the learned symmetric operator achieved ROC-AUC 0.921 at R0 but 0.651 at R3. Fixed symmetric operators ranged from 0.658 to 0.763 at R0 and 0.588 to 0.621 at R3; plain concatenation yielded 0.735 and 0.598 (Table S15). The learned operator changes model capacity as well as fusion, so its advantage cannot be attributed to symmetry alone.

Truncation sensitivity was small at the aggregate level. ESM-2–MLP ROC-AUC ranged from 0.720 to 0.734 at R0 and 0.614 to 0.617 at R3 across N-terminal retention, C-terminal retention, full-sequence processing, and segment averaging. The analysis also reports affected and unaffected test-pair subsets (Table S16); stable aggregate results do not establish that truncation is harmless for every protein. The limited ProtT5 checkpoint yielded ROC-AUC 0.528 under R0/N0 and 0.453 under R3/N0, compared with 0.509 and 0.494 under N2 (Table S17). Its weak N0 baseline limits its value as an independent replication of the ESM-2 representation effect.

Restricting matching degrees to training positives did not restore R0/N2 discrimination: logistic-regression ROC-AUC changed from 0.413 with full-graph degrees to 0.440 with training-only degrees, and MLP ROC-AUC changed from 0.463 to 0.482 (Table S9). The corresponding N0 values were unchanged at 0.744 and 0.754. This sensitivity concerns the saved R0 experiment; it does not supply prospective degree information for unseen R3 proteins.

### 3.10 Calibration improves some probability estimates without recovering discrimination

Calibration effects depended on the model and negative set. In the dedicated R3/N0 calibration experiment, the MLP Brier score decreased from 0.344 before calibration to 0.253 after Platt scaling and 0.257 after isotonic regression; ECE decreased from 0.323 to 0.126 and 0.105. Random-forest Brier score instead increased from 0.234 to 0.239 and 0.244, showing that calibration did not uniformly improve held-out reliability (Table S38). These models used a separate calibration holdout and should not be numerically equated with the architecture experiment.

Under R3/N2, isotonic-calibrated MLP probabilities had Brier score 0.250 and ECE 0.005 but ROC-AUC 0.502. Near-base-rate probabilities can therefore appear well calibrated while offering little discrimination. Platt scaling reversed the MLP ordering in this condition (ROC-AUC 0.538 to 0.462), reinforcing the need to examine discrimination separately from probability reliability. Table S38 reports raw and calibrated slope/intercept estimates and threshold metrics from the same fitted models; Table S13 separates these threshold comparisons from descriptive operating points based on the architecture predictions.

ECE also depended on binning. For the R3/N2 random forest, equal-width ECE ranged from 0.002 to 0.044 over 5–20 bins, whereas equal-frequency ECE ranged from 0.051 to 0.093 (Table S39). A single small ECE is consequently insufficient evidence of reliable probability estimates in this small, dependent test set. The component-aware intervals in Table S12 apply to raw probabilities; the calibrated comparisons in Table S38 are descriptive point estimates.

'''
m=m.replace('## 4 Discussion',sensitivity+'## 4 Discussion')
m=add_after(m,'The negative-topology findings are more consequential', 'The sensitivity analyses narrow alternative explanations without eliminating all benchmark confounding. Two-sided substitution and within-pair cluster exclusion did not restore N0-trained discrimination, and training-only degree matching retained the R0/N2 failure. Fusion capacity substantially changed the random-split result, whereas truncation strategies had little aggregate effect. These observations support evaluating representation and preprocessing choices within controlled benchmark designs. The weak N0 performance of the limited ProtT5 checkpoint prevents treating that cross-check as a comprehensive test of a second PLM family.')
m=add_after(m,'The formal inference also warrants restraint.', 'Calibration addresses a different property from interaction discrimination. Validation-fitted mappings improved some Brier and ECE estimates, but improvements were not uniform and N2 probabilities could concentrate near the class prevalence while remaining uninformative. The dependence of ECE on binning, together with small effective test samples, argues for reporting discrimination, probability reliability, and threshold performance separately. None of these calibration results validates a deployment threshold in a naturally imbalanced interactome.')
# Fix overinterpretation of the consistently trained score-inversion experiment.
m=m.replace('this is consistent with reversal of an N0-associated ordering under distribution shift.', 'these consistently trained evaluations show modest reversed class ordering, but do not establish that an N0-trained ordering was transferred to N2 or N5.')
m=m.replace('suggesting that predictors trained on the original construction retained a benchmark-specific direction that changed sign under the controlled distribution.', 'showing reversed class ordering in the consistently trained controlled-negative experiment. These predictions do not identify the mechanism responsible for reversal or demonstrate transfer of an N0-trained ordering.')
m=add_after(m,'The fourth conclusion concerns the meaning of robustness.', 'The supplementary sensitivity analyses support a similarly bounded interpretation. Altering endpoint substitution, degree-source exposure, or truncation did not restore the tested Guo discrimination, while learned fusion substantially affected the random-split result. Calibration improved selected probability metrics without establishing useful N2 ranking. The limited ProtT5 result and single-split species analyses constrain the scope of these observations rather than establishing universal representation rankings.')

# Supplementary scope and missing numerical tables.
s=s.replace('ProtParam features include length-associated and global physicochemical summaries.', 'Eight ProtParam physicochemical features were evaluated separately in the conventional stratified five-fold benchmark (Table S43). The main AAC+CTD representation excludes these features.')
s=s.replace("N7 adds exclusion of candidates in the fixed endpoint's R3-20 MMseqs2 cluster.", "N7 adds exclusion of within-pair cluster matches to N5 degree–length matching using the saved 20%-identity cluster assignment (2,191 clusters). This is distinct from a versioned train/test cluster-disjointness claim.")
s=add_after(s,'For every positive pair, the algorithm selected', 'N6 and N7 each contained 5,581 realized negative pairs. The saved N7 audit recorded zero within-pair matches under its cluster assignment and no additional skipped positive pairs due to cluster-exclusion exhaustion. N6 results and both N7 train/test directions are provided in Tables S36–S37. These sensitivity outputs were reused without rerunning negative generation or model fitting.')
s=add_after(s,'Primary degree values were derived', 'In the saved R0 analysis, N2 logistic-regression ROC-AUC was 0.413 using full-graph degree and 0.440 using training-only degree; MLP values were 0.463 and 0.482. Neither restriction restored discrimination. Table S9 does not represent an R2 or R3 degree-source replication.')
s=add_after(s,'Calibration was assessed using', 'Table S12 contains component-aware uncertainty for the raw probabilities from the calibration experiment. Table S38 adds the complete within-fit raw/Platt/isotonic comparisons; calibration mappings used a 20% holdout from training, never test labels. Table S13 now compares raw and calibrated MCC from those same fitted models. Its precision–recall operating points come from the separate architecture experiment and are descriptive test-curve summaries. They do not measure the effect of calibration and are not validated deployment thresholds. Table S39 reports both ECE binning strategies at 5, 10, 15, and 20 bins.')
s=s.replace('Calibration can change probability reliability and fixed-threshold performance but cannot restore ranking discrimination when ROC-AUC is near 0.5.', 'Calibration can change probability reliability and threshold performance; isotonic ties and an unconstrained sigmoid slope can also change the reported ranking. These transformations do not establish new interaction information.')
s=add_after(s,'Under N0, architectures produced modest differences.', 'Table S15 shows a learned-symmetric R0 ROC-AUC of 0.921 versus 0.651 at R3; learned capacity and fusion are not isolated effects. Across truncation strategies, aggregate R3 ROC-AUC varied only from 0.614 to 0.617 (Table S16). ProtT5 was weak even under N0 (R0 0.528; R3 0.453), limiting its ability to confirm a representation-specific collapse (Table S17).')
s=add_after(s,'The large N0–N2 point-estimate difference was directionally', 'Tables S40–S41 provide the paired architecture and unpaired negative-set contrasts, including effect sizes, 95% intervals, and both raw and adjusted P values. They retain the original separate 44-test bootstrap and jackknife correction families; no subset-specific correction was recomputed.')
s=add_after(s,'Giant-component exclusion and leave-one-component-out', 'Table S42 summarizes the minimum and maximum AUC over all deleted components for each model and negative set, with the number of deletions and remaining-pair range. The full component-level values remain in tables/leave_one_component_out.csv. This analysis is distinct from the giant-component-only comparison in Table S20.')
s=add_after(s,'This is evidence that degree-associated negative-set bias', 'The pooled residual discrimination is additionally confounded by cross-species pairing: 45.1% of reconstructed N2 negatives joined different species. The same-species indicator reached ROC-AUC 0.729. Tables S25–S33 and Section S20 provide the within-species controls; pooled discrimination alone cannot establish molecular interaction compatibility.')

# Compact reporting tables, all derived from existing outputs.
tables=[]
def emit(n,title,df,note):
 path=f'tables/tableS{n}_reporting_completion.csv';df.to_csv(path,index=False)
 tables.append((n,title,path,note))
metric={'roc_auc':'ROC-AUC','pr_auc':'PR-AUC','mcc':'MCC'}
for n,file,ids,title in [
 (34,'topology_only_baseline',['feature_source','model','experiment','trained_on','evaluated_on'],'Topology-only and length-only controls'),
 (35,'incremental_sequence_degree',['feature_set','trained_on','evaluated_on'],'Incremental sequence and degree features'),
 (36,'two_sided_negative_check',['representation','trained_on','evaluated_on'],'One-sided and two-sided degree-matched negative controls'),
 (37,'homology_controlled_negative',['experiment','trained_on','evaluated_on'],'Within-pair cluster-excluded negative sensitivity')]:
 df=pd.read_csv('tables/'+file+'.csv')
 emit(n,title,df[ids+list(metric)].rename(columns=metric), 'ROC-AUC, PR-AUC and MCC are descriptive point estimates. Training and evaluation negative definitions are shown separately. A denotes N0 training with test-negative replacement; B denotes training on each negative set and evaluation against N0. N7 adds saved 20%-identity within-pair cluster exclusion to joint degree–length matching. Source: tables/'+file+'.csv.')
cal=pd.read_csv('tables/table8_calibration.csv')
emit(38,'Raw and calibrated performance from the same fitted models',cal[['negative_set','model','calibration_method','roc_auc','mcc','brier_score','ece','calibration_slope','calibration_intercept']], 'R3 calibration experiment: N0 fit/calibration/test counts 4,562/1,141/462; N2 counts 4,590/1,148/446. None denotes raw probabilities. ECE uses 10 equal-width bins. Estimates are descriptive; raw-probability cluster-aware intervals are in Table S12. Source: tables/table8_calibration.csv.')
emit(39,'ECE sensitivity to bin number and binning strategy',pd.read_csv('tables/ece_bin_sensitivity.csv'), 'Raw predictions from the calibration experiment. Equal-width and equal-frequency bins are evaluated at 5, 10, 15 and 20 bins. Source: tables/ece_bin_sensitivity.csv.')
tests=pd.read_csv('tables/statistical_test_family.csv')
for n,code,title in [(40,'Q2','Paired Siamese-versus-logistic-regression ROC-AUC contrasts'),(41,'Q3','Unpaired N0-minus-N2 ROC-AUC contrasts')]:
 df=tests[tests.test_id.str.startswith(code)]
 emit(n,title,df[['negative_set','comparison','method','point','ci_low','ci_high','p_value_raw','p_value_adjusted']], 'Point is the ROC-AUC difference; intervals are 95% component-aware intervals. P values are two-sided. BH correction retains the complete 44-test family separately for each method. Source: tables/statistical_test_family.csv.')
loco=pd.read_csv('tables/leave_one_component_out.csv')
summary=loco.groupby(['negative_set','model']).agg(deletions=('component_id','count'),auc_min=('loo_metric','min'),auc_max=('loo_metric','max'),remaining_min=('n_rows_remaining','min'),remaining_max=('n_rows_remaining','max')).reset_index()
emit(42,'Leave-one-component-out sensitivity summary',summary,'Each deletion removes one complete test-graph component. Minima and maxima are sensitivity ranges, not confidence intervals. Full deletion-level results: tables/leave_one_component_out.csv.')
baseline=pd.read_csv('tables/table2_conventional_benchmark.csv')
emit(43,'Conventional stratified five-fold handcrafted-feature benchmark',baseline[['representation','model','roc_auc_mean','roc_auc_std','pr_auc_mean','pr_auc_std','mcc_mean','mcc_std']], 'Pair-level five-fold cross-validation on the canonical yeast dataset, seed 42. Standard deviations are across folds, not independent biological replicates. Physicochemical features are a separate representation. Source: tables/table2_conventional_benchmark.csv.')
summary=pd.DataFrame({'Quantity':['Unique pairs','Positive pairs','Constructed negative pairs','Protein identifiers','Observed-positive-graph nodes','Zero-positive-degree proteins','Sequences >1,022 residues','Minimum sequence length','Median sequence length','Maximum sequence length'],'Value':[len(pairs),5581,5583,len(proteins),int((degrees>0).sum()),int((degrees==0).sum()),int((lengths>1022).sum()),int(lengths.min()),float(lengths.median()),int(lengths.max())]})
emit(44,'Post-audit yeast distribution summary',summary, 'Main Figure 3 is computed from the canonical pair file and protein sequence file. The plotted class counts are in tables/yeast_postaudit_class_counts.csv; all protein lengths and degrees are in tables/yeast_postaudit_protein_distributions.csv. Zero observed degree does not establish biological isolation.')
s+='\n## S21 Additional benchmark controls and complete sensitivity results\n\nThe following tables close the numerical reporting of analyses described in the Methods. They reuse saved outputs; no additional model training or inferential testing was performed.\n\n'
for n,title,path,note in tables:
 if n==44:s+='## S22 Post-audit yeast distributions\n\n'
 s+=f'**Supplementary Table S{n}. {title}.**\n\n[CSV:{path}]\n\n{note}\n\n'

# Final editorial consolidation.
start=m.index('## 6 Conclusion');end=m.index('## Funding',start)
m=m[:start]+'## 6 Conclusion'+'\n\nProtein-novelty control and negative-set control address complementary sources of apparent performance in the Guo yeast benchmark. Discrimination declined along the matched seen–seen to both-unseen curve, whereas additional sequence-similarity exclusion produced no consistent monotonic decline. Original negative pairs carried a strong degree-associated imbalance: degree-only models reproduced high discrimination, and consistently trained sequence models approached chance after degree matching. The substitution, degree-source, representation and truncation analyses support this benchmark-specific interpretation; calibration improved selected probability metrics without recovering informative N2 ranking.\n\nThe evidence remains descriptive and non-causal. Repeated architectures and random seeds reuse the same underlying graph, and the corrected component-aware tests did not establish the N0–N2 contrast statistically. The limited ProtT5 checkpoint does not support a general PLM ranking. In the second benchmark, cross-species pairing provided an additional cue in pooled negatives; the within-species analyses refined the comparison but did not test unseen-protein generalization or establish molecular compatibility.\n\nSequence-based PPI evaluations should therefore report protein and sequence-cluster overlap, negative-set provenance, endpoint-property balance, topology-only controls, sensitivity to alternative negative definitions, and graph-aware uncertainty. Discrimination, calibration and threshold performance should be reported separately. These controls make representation and architecture comparisons more interpretable without implying that the Guo result applies to all PPI datasets or protein language models.\n\n'+m[end:]
m=m.replace('Topology-only classifiers used endpoint-degree summaries without sequence information.','Topology-only classifiers used the sum, absolute difference, minimum and maximum of endpoint positive-network degrees, without sequence information. Length-only controls used the same summaries of sequence length.')
s=s.replace('Eight ProtParam physicochemical features were evaluated separately','The eight ProtParam features were molecular weight, aromaticity, instability index, isoelectric point, GRAVY, and predicted helix, turn and sheet fractions. They were evaluated separately')
s=s.replace('The following tables close the numerical reporting of analyses described in the Methods. They reuse saved outputs; no additional model training or inferential testing was performed.','The following tables report the additional control and sensitivity analyses. LR denotes logistic regression; RF, random forest; MLP, multilayer perceptron; ROC-AUC, area under the receiver operating characteristic curve; PR-AUC, average precision; MCC, Matthews correlation coefficient; ECE, expected calibration error. Repeated row identifiers link panels of wide tables. All values derive from the saved analyses.')
main.write_text(m);supp.write_text(s)
Path('revision/reporting_completion_tables.json').write_text(json.dumps(tables,indent=2))
print('Markdown revised; new tables',len(tables))
