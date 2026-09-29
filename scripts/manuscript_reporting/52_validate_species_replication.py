"""Validate saved species splits, predictions, numerical tables, and DOCX preservation."""
from pathlib import Path
import json
import hashlib
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from docx import Document
from src.evaluation.metrics import classification_metrics
from src.evaluation.degree_distribution import compare_degree_distributions
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/species_stratified_replication'

def main(data_only=False):
    mapping=pd.read_csv(ROOT/'data/external/multi_species_taxonomy/species_assignments.tsv',sep='\t').set_index('protein_id')
    frames=[];checked=0
    for species in ['celegans','dmelanogaster','ecoli']:
        folder=OUT/species;metrics=pd.read_csv(folder/'metrics.csv');assert len(metrics)==20
        balance=pd.read_csv(folder/'balance.csv').set_index('variant')
        stats=pd.read_csv(folder/'protein_stats.tsv',sep='\t',index_col=0)
        pos_tests=[]
        for v in ['n0','n2','n2_exact','n2_nn']:
            train=pd.read_csv(folder/f'{v}_train.tsv',sep='\t');test=pd.read_csv(folder/f'{v}_test.tsv',sep='\t')
            a=set(zip(train.protein_a,train.protein_b));b=set(zip(test.protein_a,test.protein_b));assert not a&b
            original=pd.read_csv(folder/'pairs.tsv',sep='\t')
            negatives=pd.read_csv(folder/f'{v}_negatives.tsv',sep='\t')
            combined=pd.concat([train,test],ignore_index=True)
            def pair_set(df):return set(zip(df.protein_a,df.protein_b))
            assert pair_set(combined[combined.label==1])==pair_set(original[original.label==1])
            assert pair_set(combined[combined.label==0])==pair_set(negatives)
            assert not pair_set(negatives)&pair_set(original[original.label==1])
            assert (negatives.protein_a!=negatives.protein_b).all()
            if v=='n0':assert pair_set(negatives)==pair_set(original[original.label==0])
            positives=original[original.label==1]
            counts=pd.concat([positives.protein_a,positives.protein_b]).value_counts()
            assert np.array_equal(stats.positive_degree,counts.reindex(stats.index,fill_value=0))
            check_balance=compare_degree_distributions(positives,negatives,stats.positive_degree,2000,42)
            for key in ['standardized_mean_difference','ks_statistic','wasserstein_distance']:
                assert np.isclose(check_balance[key],balance.loc[v,key],atol=1e-12,rtol=0)
            assert len(negatives)==balance.loc[v,'n_negatives']
            for df in [train,test]:
                assert (df.protein_a<df.protein_b).all()
                assert df[['protein_a','protein_b']].apply(lambda col:col.map(mapping.species).eq(species).all()).all()
                assert not df.duplicated(['protein_a','protein_b']).any()
            pos_tests.append(test[test.label==1].reset_index(drop=True))
            for row in metrics[metrics.variant==v].itertuples():
                pred=pd.read_csv(folder/f'{v}__{row.representation}__{row.model}'/'predictions.csv')
                pd.testing.assert_frame_equal(pred[['protein_a','protein_b','label']],test)
                assert len(pred)==row.n_test and len(train)==row.n_train
                assert np.isfinite(pred.y_prob).all() and pred.y_prob.between(0,1).all()
                for key,value in classification_metrics(pred.label,pred.y_prob).items():assert np.isclose(value,getattr(row,key),atol=1e-12,rtol=0),(species,v,key)
                checked+=1
        for p in pos_tests[1:]:pd.testing.assert_frame_equal(pos_tests[0],p)
        frames.append(metrics)
    report={'prediction_files_validated':checked,'model_rows':sum(len(f) for f in frames),'checks':['within-species endpoints','canonical unique pairs','no train/test pair overlap','identical positive partitions across variants','saved prediction order and row counts','recomputed all nine model metrics','finite probability range']}
    allpairs=pd.read_csv(ROOT/'data/processed/multi_species/ppi_pairs_clean.tsv',sep='\t',header=None)
    accounting=pd.read_csv(ROOT/'tables/species_pair_accounting.csv');assert accounting[['positive','negative']].to_numpy().sum()==len(allpairs)
    diagnostics=pd.read_csv(ROOT/'tables/pooled_same_species_diagnostic.csv')
    for row in diagnostics.itertuples():
        pred=pd.read_csv(OUT/f'pooled_{row.variant}_same_species_predictions.csv')
        expected=pred.protein_a.map(mapping.species).eq(pred.protein_b.map(mapping.species)).astype(float)
        assert np.array_equal(expected,pred.y_prob)
        for key,value in classification_metrics(pred.label,pred.y_prob).items():
            assert np.isclose(value,getattr(row,key),atol=1e-12,rtol=0)
    report['checks']+=['complete positive/negative split accounting','no generated negative overlaps a known positive','no negative self-pairs','original N0 labels preserved','pooled taxonomy diagnostic recomputed','positive-network degrees and SMD/KS/Wasserstein recomputed']
    if data_only:
        (OUT/'data_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));return
    if (ROOT/'revision/reporting_table_panels.json').exists():
        # The user-authorized reporting revision rebuilds document layout and
        # splits wide tables into panels; validate its current content contract.
        import importlib.util
        spec=importlib.util.spec_from_file_location('reporting_validation',ROOT/'scripts/55_validate_reporting_completion.py')
        validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
        validator.main()
        report['document_validation']=json.loads((ROOT/'revision/reporting_completion_validation.json').read_text())
        (OUT/'validation.json').write_text(json.dumps(report,indent=2))
        print('Species data and current reporting revision passed validation.')
        return
    for file in ['final_manuscrip_v2.docx','supplementary_v2.docx']:
        before=Document(OUT/file.replace('.docx','_before_species.docx'));after=Document(ROOT/'manuscript'/file)
        # Preserve all content except explicitly recorded scientific claim revisions.
        changes=json.loads((OUT/'manuscript_claim_revisions.json').read_text()) if file.startswith('final') else {}
        previous=before.paragraphs;i=0
        for p in after.paragraphs:
            if i<len(previous):
                old=previous[i]
                matches=p.text==changes[old.text] if old.text in changes else p._p.xml==old._p.xml
                if matches:i+=1
        assert i==len(previous),(file,'paragraphs altered')
        assert [t._tbl.xml for t in before.tables]==[t._tbl.xml for t in after.tables[:len(before.tables)]]
        assert len(after.inline_shapes)==len(before.inline_shapes)+(1 if file.startswith('final') else 0)
    main=Document(ROOT/'manuscript/final_manuscrip_v2.docx');sup=Document(ROOT/'manuscript/supplementary_v2.docx')
    assert any(p.text.startswith('3.8 Species-stratified') for p in main.paragraphs)
    main_text='\n'.join(p.text for p in main.paragraphs)
    for species,name in [('celegans','C. elegans'),('dmelanogaster','D. melanogaster'),('ecoli','E. coli')]:
        rows=pd.read_csv(OUT/species/'metrics.csv')
        def auc(v,r,m='logistic_regression'):
            return rows.loc[(rows.variant==v)&(rows.representation==r)&(rows.model==m),'roc_auc'].item()
        assert f"In {name}, degree-only logistic regression changed from ROC-AUC {auc('n0','degree_only'):.3f} under N0 to {auc('n2','degree_only'):.3f} under N2." in main_text
        assert f"ESM-2–MLP from {auc('n0','esm2_mean','mlp'):.3f} to {auc('n2','esm2_mean','mlp'):.3f}." in main_text
    assert len(sup.tables)==33
    # Reporting tables must match source metrics at displayed precision.
    allmetrics=pd.concat(frames,ignore_index=True)
    for table_number,species in zip([27,28,29],['celegans','dmelanogaster','ecoli']):
        display=pd.read_csv(ROOT/f'tables/tableS{table_number}_species_replication.csv')
        expected=allmetrics[allmetrics.species==species]
        assert np.allclose(display[['ROC-AUC','PR-AUC','MCC']],expected[['roc_auc','pr_auc','mcc']],atol=1e-12)
    for table_number in range(25,34):
        display=pd.read_csv(ROOT/f'tables/tableS{table_number}_species_replication.csv')
        t=sup.tables[table_number-1]
        for i,row in enumerate(display.itertuples(index=False,name=None),1):
            for j,value in enumerate(row):
                expected_text=f'{value:.4f}' if isinstance(value,float) else str(value)
                assert t.cell(i,j).text==expected_text
    inputs=['data/processed/multi_species/ppi_pairs_clean.tsv','data/processed/multi_species/proteins_with_groups.tsv','data/external/multi_species_taxonomy/species_assignments.tsv','embeddings/esm2_multi_species/protein_embeddings.h5']
    inputs+=['data/processed/multi_species/features/aac.csv','data/processed/multi_species/features/ctd.csv']
    inputs += [f'scripts/{name}' for name in ['49_species_stratified_replication.py','50_audit_species_assignments.py','51_report_species_replication.py','52_validate_species_replication.py']]
    report['input_sha256']={f:hashlib.file_digest((ROOT/f).open('rb'),'sha256').hexdigest() for f in inputs} if hasattr(hashlib,'file_digest') else {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in inputs}
    report['checks']+=['complete pair accounting','original document content preserved except recorded claim revisions','supplementary table values agree with source metrics']
    (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-only',action='store_true')
    main(parser.parse_args().data_only)
