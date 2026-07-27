import json
import pytest
from src.agent import AgentToolCaller, RoutingPolicy, ToolArgumentsError, ToolDispatcher

class Client:
    def __init__(self,name,args): self.name,self.args=name,args
    def create_chat_completion(self,**kwargs):
        return {'choices':[{'message':{'tool_calls':[{'function':{'name':self.name,'arguments':json.dumps(self.args)}}]}}]}
class Service:
    def rank_airports(self,r): return r
    def compare_airports(self,r): return r
    def calculate_long_haul_share(self,r): return r
    def estimate_unmet_capacity(self,r): return r
    def get_airport_profile(self,r): return r

def route(name,args,question,policy=None):
    return AgentToolCaller(Client(name,args),dispatcher=ToolDispatcher(Service()),policy=policy).route(question)

def test_requested_metric_cannot_be_omitted():
    with pytest.raises(ToolArgumentsError,match='metric selector'):
        route('compare_airports',{'airport_codes':['LAX','SNA']},'Compare LAX and SNA congestion')

def test_requested_exclusion_cannot_be_omitted():
    with pytest.raises(ToolArgumentsError,match='exclusions'):
        route('rank_airports',{'region':'New England','limit':5},'Rank New England airports excluding Boston')

@pytest.mark.parametrize('excluded',[['PVD'],['BOS','PVD']])
def test_wrong_or_extra_exclusion_is_rejected(excluded):
    with pytest.raises(ToolArgumentsError,match='exclusions'):
        route('rank_airports',{'region':'New England','limit':5,'excluded_airports':excluded},'Rank New England airports excluding Boston')

def test_exact_exclusion_and_metric_are_preserved():
    ranked=route('rank_airports',{'region':'New England','limit':5,'excluded_airports':['BOS']},'Rank New England airports excluding Boston')
    compared=route('compare_airports',{'airport_codes':['LAX','SNA'],'metrics':['congestion_score']},'Compare LAX and SNA congestion')
    assert ranked.arguments['excluded_airports']==['BOS']
    assert compared.arguments['metrics']==['congestion_score']

def test_runtime_policy_supplies_omitted_numeric_defaults():
    policy=RoutingPolicy(ranking_limit=4,long_haul_threshold_miles=2800,target_load_factor=.86)
    result=route('calculate_long_haul_share',{'airport_code':'ANC'},'What share of ANC flights are long haul?',policy)
    assert result.arguments['threshold_miles']==2800
