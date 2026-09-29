"""Cache accession-resolved UniProt taxonomy for species-stratified replication."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data/external/multi_species_taxonomy'

def fetch(item):
    i, ids = item
    path = CACHE/f'batch_{i:03d}.json'
    if path.exists(): return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = requests.get('https://rest.uniprot.org/uniprotkb/search', params={
                'query': ' OR '.join('accession:'+a for a in ids if len(a)==6),
                'format':'json','size':500,'fields':'accession,organism_name,organism_id'},timeout=60)
            r.raise_for_status(); payload=r.json()
            assert 'next' not in r.links
            path.write_text(json.dumps(payload))
            print(f'Cached batch {i}',flush=True)
            return payload
        except Exception:
            if attempt==3: raise
            time.sleep(2*(attempt+1))

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    ids=pd.read_csv(ROOT/'data/processed/multi_species/proteins_with_groups.tsv',sep='\t').protein_id.tolist()
    uniprot_ids=[a for a in ids if not a.startswith('gi:')]
    batches=[(i,uniprot_ids[s:s+100]) for i,s in enumerate(range(0,len(uniprot_ids),100))]
    with ThreadPoolExecutor(max_workers=6) as pool: payloads=list(pool.map(fetch,batches))
    resolved={}
    for payload in payloads:
        for rec in payload['results']:
            if 'organism' not in rec: continue
            org=rec['organism']
            for acc in [rec['primaryAccession']]+rec.get('secondaryAccessions',[]):
                row={'protein_id':acc,'primary_accession':rec['primaryAccession'],'taxon_id':org['taxonId'],'organism':org['scientificName']}
                if acc in resolved: assert resolved[acc]['taxon_id']==row['taxon_id']
                resolved[acc]=row
    rows=[resolved.get(a,{'protein_id':a}) for a in ids]
    out=pd.DataFrame(rows);out.to_csv(CACHE/'accession_taxonomy.tsv',sep='\t',index=False)
    print(out.organism.value_counts(dropna=False).to_string(),flush=True)
    print('Unresolved',int(out.taxon_id.isna().sum()),flush=True)
def fetch_gi():
    ids=pd.read_csv(ROOT/'data/processed/multi_species/proteins_with_groups.tsv',sep='\t').protein_id
    ids=[a[3:] for a in ids if a.startswith('gi:')]
    rows=[]
    for start in range(0,len(ids),150):
        path=CACHE/f'ncbi_{start//150:03d}.json'
        if path.exists(): payload=json.loads(path.read_text())
        else:
            for attempt in range(4):
                r=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi',params={'db':'protein','id':','.join(ids[start:start+150]),'retmode':'json'},timeout=60)
                if r.ok and 'result' in r.json(): break
                time.sleep(2)
            r.raise_for_status();payload=r.json();path.write_text(json.dumps(payload));time.sleep(.4)
        for uid,rec in payload.get('result',{}).items():
            if uid=='uids':continue
            rows.append({'protein_id':'gi:'+uid,'taxon_id':rec.get('taxid'),'title':rec.get('title'),'accession':rec.get('accessionversion'),'error':rec.get('error')})
        print('NCBI batch',start//150,flush=True)
    pd.DataFrame(rows).to_csv(CACHE/'gi_taxonomy.tsv',sep='\t',index=False)
def fetch_archives():
    import re
    p=pd.read_csv(CACHE/'accession_taxonomy.tsv',sep='\t')
    ids=p.loc[p.taxon_id.isna() & p.protein_id.str.len().eq(6),'protein_id'].tolist()
    def one(acc):
        path=CACHE/f'unisave_{acc}.txt'
        if path.exists(): text=path.read_text()
        else:
            for attempt in range(4):
                try:
                    r=requests.get(f'https://rest.uniprot.org/unisave/{acc}',params={'format':'txt','versions':'1'},timeout=40)
                    r.raise_for_status();text=r.text;path.write_text(text);break
                except requests.RequestException:
                    if attempt==3:return {'protein_id':acc,'error':'archive retrieval failed'}
                    time.sleep(2)
        tax=re.search(r'NCBI_TaxID=(\d+)',text)
        org=' '.join(re.findall(r'^OS   (.*)$',text,re.M))
        return {'protein_id':acc,'taxon_id':int(tax.group(1)) if tax else next((v for k,v in {'CAENORHABDITIS ELEGANS':6239,'DROSOPHILA MELANOGASTER':7227,'ESCHERICHIA COLI':562}.items() if k in org.upper()),None),'organism':org,'source':'UniSave version 1'}
    rows=[]
    with ThreadPoolExecutor(max_workers=10) as pool:
        for row in pool.map(one,ids):
            rows.append(row)
            if len(rows)%100==0:print('Archived',len(rows),'of',len(ids),flush=True)
    pd.DataFrame(rows).to_csv(CACHE/'archive_taxonomy.tsv',sep='\t',index=False)
    print('Archive unresolved',sum(pd.isna(r.get('taxon_id')) for r in rows),flush=True)
if __name__=='__main__':
    import sys
    if '--probe' in sys.argv:
        for url in ['https://rest.uniprot.org/unisave/O16257?format=txt&versions=last','https://rest.uniprot.org/uniparc/UPI000007E4EA/databases?format=json']:
            r=requests.get(url,timeout=30);print(r.status_code,r.text[:1800],flush=True)
    elif '--archives' in sys.argv:fetch_archives()
    elif '--gi' in sys.argv:fetch_gi()
    else:main()
