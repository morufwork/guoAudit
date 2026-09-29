"""Repeat scripts 42/43 within each taxonomically resolved species; save splits/predictions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(key,'1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.data.negative_sampling import (_known_pair_set,compute_protein_stats,sample_matched_negatives,sample_exact_degree_matched_negatives,sample_nearest_degree_matched_negatives)
from src.evaluation.degree_distribution import compare_degree_distributions,pair_associated_degree_values,describe_degree_distribution
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.manifest import write_manifest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/species_stratified_replication'
NAMES={'celegans':'Caenorhabditis elegans','dmelanogaster':'Drosophila melanogaster','ecoli':'Escherichia coli'}
MODELS={'logistic_regression':{'max_iter':2000},'mlp':{'hidden_layer_sizes':[128,64],'max_iter':500,'early_stopping':True}}

@threadpool_limits.wrap(limits=1)
def run(species):
    out=OUT/species;out.mkdir(parents=True,exist_ok=True)
    proteins=pd.read_csv(ROOT/'data/processed/multi_species/proteins_with_groups.tsv',sep='\t')
    mapping=pd.read_csv(ROOT/'data/external/multi_species_taxonomy/species_assignments.tsv',sep='\t')
    assert mapping.protein_id.is_unique and set(mapping.protein_id)==set(proteins.protein_id)
    assert mapping.species.notna().all()
    ids=set(mapping.loc[mapping.species==species,'protein_id'])
    proteins=proteins[proteins.protein_id.isin(ids)].reset_index(drop=True)
    pairs=pd.read_csv(ROOT/'data/processed/multi_species/ppi_pairs_clean.tsv',sep='\t',header=None,names=['protein_a','protein_b','label'])
    pairs=pairs[pairs.protein_a.isin(ids)&pairs.protein_b.isin(ids)].reset_index(drop=True)
    assert not pairs.duplicated(['protein_a','protein_b']).any()
    assert (pairs.protein_a<pairs.protein_b).all()
    proteins.to_csv(out/'proteins.tsv',sep='\t',index=False);pairs.to_csv(out/'pairs.tsv',sep='\t',index=False)
    pos=pairs[pairs.label==1].reset_index(drop=True);neg0=pairs[pairs.label==0].reset_index(drop=True)
    stats=compute_protein_stats(proteins,pos,10);degree=stats.positive_degree;known=_known_pair_set(pos)
    stats.to_csv(out/'protein_stats.tsv',sep='\t')
    variants={'n0':neg0,'n2':sample_matched_negatives(pos,stats,known,['degree_decile'],42),
              'n2_exact':sample_exact_degree_matched_negatives(pos,stats,known,42),
              'n2_nn':sample_nearest_degree_matched_negatives(pos,stats,known,42)}
    audit={'species':species,'organism':NAMES[species],'n_proteins':len(proteins),'n_pairs':len(pairs),'n_positive':len(pos),'n_negative':len(neg0),'zero_positive_degree':int((degree==0).sum())}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    feat=ROOT/'data/processed/multi_species/features'
    features={'aac_ctd':pd.read_csv(feat/'aac.csv').merge(pd.read_csv(feat/'ctd.csv'),on='protein_id'),
              'esm2_mean':load_esm2_features('mean',ROOT/'embeddings/esm2_multi_species')}
    features={k:v[v.protein_id.isin(ids)] for k,v in features.items()}
    for f in features.values():assert set(f.protein_id)==ids
    pt,pv=train_test_split(pos,test_size=.2,random_state=42)
    balances=[];rows=[];splits=[];descriptions=[]
    for variant,neg in variants.items():
        assert len(neg)>1 and not neg.duplicated(['protein_a','protein_b']).any()
        assert not (_known_pair_set(neg)&known)
        assert (set(neg.protein_a)|set(neg.protein_b))<=ids
        neg.to_csv(out/f'{variant}_negatives.tsv',sep='\t',index=False)
        balances.append({'species':species,'variant':variant,'n_negatives':len(neg),'skipped':len(pos)-len(neg) if variant!='n0' else 0,**compare_degree_distributions(pos,neg,degree,2000,42)})
        for label,p in [('positive',pos),('negative',neg)]:descriptions.append({'species':species,'variant':variant,'label':label,**describe_degree_distribution(pair_associated_degree_values(p,degree))})
        nt,nv=train_test_split(neg,test_size=.2,random_state=42)
        train=pd.concat([pt,nt],ignore_index=True);test=pd.concat([pv,nv],ignore_index=True)
        assert not (_known_pair_set(train)&_known_pair_set(test))
        for name,df in [('train',train),('test',test)]:df.to_csv(out/f'{variant}_{name}.tsv',sep='\t',index=False)
        train_ids=set(train.protein_a)|set(train.protein_b);test_ids=set(test.protein_a)|set(test.protein_b)
        splits.append({'species':species,'variant':variant,'train_positive':len(pt),'train_negative':len(nt),'test_positive':len(pv),'test_negative':len(nv),'train_proteins':len(train_ids),'test_proteins':len(test_ids),'overlapping_proteins':len(train_ids&test_ids),'pair_overlap':0})
        for representation in ['degree_only','aac_ctd','esm2_mean']:
            def matrix(df):
                if representation!='degree_only':return build_pair_matrix(df,features[representation],fusion='combined')
                a=df.protein_a.map(degree).to_numpy(float);b=df.protein_b.map(degree).to_numpy(float)
                return np.column_stack([a+b,abs(a-b),np.minimum(a,b),np.maximum(a,b)]),df.label.to_numpy()
            x,y=matrix(train);xt,yt=matrix(test);scaler=StandardScaler();x=scaler.fit_transform(x);xt=scaler.transform(xt)
            configs={'logistic_regression':{}} if representation=='degree_only' else MODELS
            for name,params in configs.items():
                dest=out/f'{variant}__{representation}__{name}';dest.mkdir(exist_ok=True)
                if (dest/'metrics.json').exists(): result=json.loads((dest/'metrics.json').read_text())
                else:
                    model=build_model(name,params,42)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');model.fit(x,y)
                    prob=model.predict_proba(xt)[:,1]
                    pred=test.copy();pred['y_prob']=prob;pred.to_csv(dest/'predictions.csv',index=False)
                    result={'species':species,'variant':variant,'representation':representation,'model':name,'n_train':len(train),'n_test':len(test),**classification_metrics(yt,prob)}
                    (dest/'metrics.json').write_text(json.dumps(result,indent=2))
                    (dest/'fit.json').write_text(json.dumps({'warnings':[str(w.message) for w in caught],'n_iter':np.asarray(model.n_iter_).tolist()},indent=2))
                rows.append(result);print(species,variant,representation,name,round(result['roc_auc'],4),flush=True)
    for name,data in [('metrics',rows),('balance',balances),('splits',splits),('degree_distributions',descriptions)]:pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    write_manifest(out,config={'protocol':'scripts 42/43 within species','seed':42,'test_fraction':.2,'n_boot':2000,'fusion':'combined','models':MODELS},seed=42)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--species',choices=NAMES,required=True);run(parser.parse_args().species)
