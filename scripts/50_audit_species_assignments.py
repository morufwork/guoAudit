"""Resolve taxonomy from cached records and audit within-/cross-species pair membership."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/external/multi_species_taxonomy'

def species(tax,organism=''):
    if pd.isna(tax):
        org=str(organism).upper()
        for name,key in [('CAENORHABDITIS ELEGANS','celegans'),('DROSOPHILA MELANOGASTER','dmelanogaster'),('ESCHERICHIA COLI','ecoli')]:
            if name in org:return key
        return 'other' if pd.notna(organism) and org.strip() else 'unresolved'
    if int(tax)==6239:return 'celegans'
    if int(tax)==7227:return 'dmelanogaster'
    if int(tax) in [562,83333,511145] or str(organism).startswith('Escherichia coli'):return 'ecoli'
    return 'other'

def main():
    proteins=pd.read_csv(ROOT/'data/processed/multi_species/proteins_with_groups.tsv',sep='\t')
    mapping=pd.read_csv(CACHE/'accession_taxonomy.tsv',sep='\t').set_index('protein_id')
    mapping['source']='UniProtKB current accession'
    for filename,source in [('gi_taxonomy.tsv','NCBI protein ESummary'),('archive_taxonomy.tsv','UniSave version 1')]:
        frame=pd.read_csv(CACHE/filename,sep='\t')
        for row in frame.to_dict('records'):
            if (pd.notna(row.get('taxon_id')) or pd.notna(row.get('organism'))) and pd.isna(mapping.loc[row['protein_id'],'taxon_id']):
                for key in ['taxon_id','organism']:
                    if key in row:mapping.loc[row['protein_id'],key]=row[key]
                mapping.loc[row['protein_id'],'source']=source
    mapping['species']=[species(r.taxon_id,r.organism) for r in mapping.itertuples()]
    seqs=proteins.set_index('protein_id').sequence
    sequence_species={}
    for acc,row in mapping[mapping.species!='unresolved'].iterrows():sequence_species.setdefault(seqs[acc],set()).add(row.species)
    for acc,row in mapping[mapping.species=='unresolved'].iterrows():
        candidates=sequence_species.get(seqs[acc],set())
        if len(candidates)==1:
            mapping.loc[acc,'species']=next(iter(candidates));mapping.loc[acc,'source']='Exact sequence match to taxonomically resolved benchmark protein'
    mapping.to_csv(CACHE/'species_assignments.tsv',sep='\t')
    pairs=pd.read_csv(ROOT/'data/processed/multi_species/ppi_pairs_clean.tsv',sep='\t',header=None,names=['protein_a','protein_b','label'])
    pairs['species_a']=pairs.protein_a.map(mapping.species);pairs['species_b']=pairs.protein_b.map(mapping.species)
    assert pairs.species_a.notna().all() and pairs.species_b.notna().all()
    pairs['category']=[a if a==b and a in ['celegans','dmelanogaster','ecoli'] else ('unresolved_endpoint' if 'unresolved' in [a,b] else ('other_organism_endpoint' if 'other' in [a,b] else 'cross_species')) for a,b in zip(pairs.species_a,pairs.species_b)]
    pairs.to_csv(CACHE/'pair_taxonomy_audit.tsv',sep='\t',index=False)
    summary=pairs.groupby(['category','label']).size().unstack(fill_value=0).rename(columns={0:'negative',1:'positive'})
    summary.to_csv(ROOT/'tables/species_pair_accounting.csv')
    pooled_path=ROOT/'results/species_stratified_replication/pooled_n2_reconstructed.tsv'
    if not pooled_path.exists():
        from src.data.negative_sampling import compute_protein_stats,sample_matched_negatives,_known_pair_set
        pos=pairs[pairs.label==1][['protein_a','protein_b','label']].reset_index(drop=True)
        n=sample_matched_negatives(pos,compute_protein_stats(proteins,pos,10),_known_pair_set(pos),['degree_decile'],42)
        pooled_path.parent.mkdir(exist_ok=True,parents=True);n.to_csv(pooled_path,sep='\t',index=False)
    pooled=pd.read_csv(pooled_path,sep='\t')
    a=pooled.protein_a.map(mapping.species);b=pooled.protein_b.map(mapping.species)
    pooled['category']=[x if x==y and x in ['celegans','dmelanogaster','ecoli'] else ('unresolved_endpoint' if 'unresolved' in [x,y] else ('other_organism_endpoint' if 'other' in [x,y] else 'cross_species')) for x,y in zip(a,b)]
    pooled.groupby('category').size().rename('n_pairs').to_csv(ROOT/'tables/pooled_n2_species_accounting.csv')
    from sklearn.model_selection import train_test_split
    from src.evaluation.metrics import classification_metrics
    cols=['protein_a','protein_b','label']
    pos=pairs[pairs.label==1][cols].reset_index(drop=True)
    _,pt=train_test_split(pos,test_size=.2,random_state=42)
    diagnostic=[]
    for variant,neg in [('n0',pairs[pairs.label==0][cols].reset_index(drop=True)),('n2',pooled[cols])]:
        _,nt=train_test_split(neg,test_size=.2,random_state=42)
        test=pd.concat([pt,nt],ignore_index=True)
        sa=test.protein_a.map(mapping.species);sb=test.protein_b.map(mapping.species)
        valid=sa.isin(['celegans','dmelanogaster','ecoli'])&sb.isin(['celegans','dmelanogaster','ecoli'])
        pred=test[valid].copy();pred['y_prob']=sa[valid].eq(sb[valid]).astype(float)
        pred.to_csv(pooled_path.parent/f'pooled_{variant}_same_species_predictions.csv',index=False)
        diagnostic.append({'variant':variant,'n_evaluated':len(pred),'n_excluded':int((~valid).sum()),**classification_metrics(pred.label,pred.y_prob)})
    pd.DataFrame(diagnostic).to_csv(ROOT/'tables/pooled_same_species_diagnostic.csv',index=False)
    print('Pooled N2 pair categories:',pooled.category.value_counts().to_dict())
    print(mapping.species.value_counts().to_string());print(summary.to_string())
    print('Unresolved:',mapping[mapping.species=='unresolved'].index.tolist())
    (CACHE/'provenance.json').write_text(json.dumps({'taxonomy_sources':['https://rest.uniprot.org/uniprotkb/search','https://rest.uniprot.org/unisave/','https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'],'policy':'Use accession taxonomy; archived version 1 for retired UniProt entries; exact sequence transfer only if uniquely assigned among resolved benchmark records; exclude other and unresolved organisms and cross-species pairs. E. coli strain taxa consolidated.','input_proteins':len(proteins),'input_pairs':len(pairs)},indent=2))
if __name__=='__main__':main()
