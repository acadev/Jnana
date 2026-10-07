#!/usr/bin/env python3
"""Run a concise text-only membrane hypothesis tournament in native Jnana."""
from __future__ import annotations
import argparse,itertools,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jnana.protognosis.agents.laya_checkpoint_router import checkpoint_router_specs
from jnana.protognosis.core.agent_core import ResearchHypothesis
from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.core.multi_llm_config import LLMConfig
CRITERIA=['novelty','plausibility','testability']
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def semantic(match):
 return match['hypothesis1_id'] if match['overall_winner']=='A' else match['hypothesis2_id'] if match['overall_winner']=='B' else None
def expected(a,b):return 1/(1+10**((b-a)/400))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--archive',required=True);p.add_argument('--state',required=True);p.add_argument('--report',required=True);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
 src=json.loads(Path(a.source).read_text());rows=src['hypotheses'];assert len(rows)==len({x['hypothesis_id'] for x in rows})==50
 state,report=Path(a.state),Path(a.report);state.unlink(missing_ok=True);report.unlink(missing_ok=True);archive=Path(a.archive).resolve()
 cfg=LLMConfig(provider='openai',model='unused',api_key='local-no-auth',base_url='http://127.0.0.1:9/v1');cs=CoScientist(llm_config=cfg,storage_path=str(state),max_workers=1);cs.memory.metadata.update({'research_goal':src['question'],'research_plan_config':{'evaluation_criteria':CRITERIA},'campaign':{'name':src['campaign'],'protocol':'concise text-only complete bidirectional round robin'}})
 for row in rows:cs.memory.add_hypothesis(ResearchHypothesis(row['content'],row['title'],'concise-campaign',hypothesis_id=row['hypothesis_id'],metadata={'title':row['title']}))
 routers=checkpoint_router_specs(archive/'model',[{'name':'calibrated-1500','path':archive/'checkpoints/step-0001500'},{'name':'transfer-2200','path':archive/'checkpoints/step-0002200'},{'name':'balanced-2400','path':archive/'checkpoints/step-0002400'}],device='cuda',batch_size=a.batch_size);agent=cs.configure_laya_judges(routers);hs=cs.get_all_hypotheses();pairs=[];ids=[]
 for x,y in itertools.combinations(hs,2):ids.append((x.hypothesis_id,y.hypothesis_id));pairs.extend(((x,y),(y,x)))
 started=time.perf_counter();native=agent.judge_batch(pairs,batch_size=a.batch_size);seconds=time.perf_counter()-started;ratings={h.hypothesis_id:1200. for h in hs};fixtures=[]
 for i,(x,y) in enumerate(ids):
  ab,ba=native[2*i:2*i+2];wa,wb=semantic(ab),semantic(ba);winner=wa if wa and wa==wb else None;score=.5 if winner is None else 1. if winner==x else 0.;ea=expected(ratings[x],ratings[y]);delta=32*(score-ea);ratings[x]+=delta;ratings[y]-=delta;fixtures.append({'fixture':i+1,'hypothesis_ids':[x,y],'winner_ab':wa,'winner_ba':wb,'consensus_winner':winner,'score_first':score,'elo_delta_first':delta,'native_match_ids':[ab['match_id'],ba['match_id']]})
 by={h.hypothesis_id:h for h in hs};ranking=sorted([{'hypothesis_id':k,'title':by[k].summary,'elo_rating':v} for k,v in ratings.items()],key=lambda x:x['elo_rating'],reverse=True)
 for i,x in enumerate(ranking,1):x['rank']=i
 persisted=json.loads(state.read_text());matches=persisted['tournament_state']['matches'];checks={'hypotheses':len(persisted['hypotheses'])==50,'matches':len(matches)==2450,'fixtures':len(fixtures)==1225,'three_models':all(len(m.get('model_judgments',[]))==3 for m in matches),'concise_metadata':all(set(h.get('metadata',{}))=={'title'} for h in persisted['hypotheses']),'zero_sum':abs(sum(ratings.values())-60000)<1e-6};out={'schema_version':'jnana_membrane_text_only_v1','question':src['question'],'criteria':CRITERIA,'counts':{'hypotheses':50,'fixtures':1225,'oriented_matches':2450,'model_inferences':7350,'questions_per_inference':4},'timing':{'seconds':seconds,'oriented_matches_per_second':2450/seconds},'checks':checks,'ranking':ranking,'fixtures':fixtures};dump(report,out);print(json.dumps({'report':str(report),'timing':out['timing'],'checks':checks,'top5':ranking[:5]},indent=2));assert all(checks.values())
if __name__=='__main__':main()
