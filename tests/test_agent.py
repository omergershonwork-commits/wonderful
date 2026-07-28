import json
import pytest
from src.agent import (
    AgentToolCaller, RoutingPolicy, ToolArgumentsError, ToolDispatcher,
    ToolSelectionError, UnknownToolError, approved_tool_schemas,
)

class FakeClient:
    def __init__(self, response): self.response=response
    def create_chat_completion(self, **kwargs): return self.response
class Service:
    def rank_airports(self,r): return {'tool':'rank_airports','args':r.model_dump(mode='json')}
    def compare_airports(self,r): return {'tool':'compare_airports','args':r.model_dump(mode='json')}
    def calculate_long_haul_share(self,r): return {'tool':'calculate_long_haul_share','args':r.model_dump(mode='json')}
    def estimate_unmet_capacity(self,r): return {'tool':'estimate_unmet_capacity','args':r.model_dump(mode='json')}
    def get_airport_profile(self,r): return {'tool':'get_airport_profile','args':r.model_dump(mode='json')}

def response(name,args):
    return {'choices':[{'message':{'tool_calls':[{'function':{'name':name,'arguments':json.dumps(args)}}]}}]}
def caller(name,args,policy=None):
    return AgentToolCaller(FakeClient(response(name,args)),dispatcher=ToolDispatcher(Service()),policy=policy)

def test_five_approved_schemas():
    assert [x['function']['name'] for x in approved_tool_schemas()] == [
        'rank_airports','compare_airports','calculate_long_haul_share','estimate_unmet_capacity','get_airport_profile'
    ]

@pytest.mark.parametrize(('question','name','args'),[
 ('Which airports in New England are strong candidates for terminal expansion?','rank_airports',{'region':'New England','limit':5}),
 ('Compare LA and Santa Ana airport congestion levels.','compare_airports',{'airport_codes':['LAX','SNA'],'metrics':['congestion_score']}),
 ('What is the percentage of long haul flights out of Anchorage airport?','calculate_long_haul_share',{'airport_code':'ANC'}),
 ('What is the unmet flight demand in SFO airport and why?','estimate_unmet_capacity',{'airport_code':'SFO'}),
])
def test_required_questions(question,name,args):
    result=caller(name,args).route(question)
    assert result.tool_name==name

def test_requested_metric_and_exclusion_cannot_be_omitted():
    with pytest.raises(ToolArgumentsError,match='metric selector'):
        caller('compare_airports',{'airport_codes':['LAX','SNA']}).route('Compare LAX and SNA congestion')
    with pytest.raises(ToolArgumentsError,match='exclusions'):
        caller('rank_airports',{'region':'New England','limit':5}).route('Rank New England airports excluding Boston')

@pytest.mark.parametrize('excluded',[['PVD'],['BOS','PVD']])
def test_wrong_or_extra_exclusion_rejected(excluded):
    with pytest.raises(ToolArgumentsError,match='exclusions'):
        caller('rank_airports',{'region':'New England','limit':5,'excluded_airports':excluded}).route('Rank New England airports excluding Boston')

def test_explicit_overrides_and_policy_defaults():
    assert caller('calculate_long_haul_share',{'airport_code':'ANC','threshold_miles':2500}).route('ANC long haul using 2,500 miles').arguments['threshold_miles']==2500
    assert caller('estimate_unmet_capacity',{'airport_code':'SFO','target_load_factor':.85}).route('SFO unmet capacity at target load factor 85%').arguments['target_load_factor']==.85
    policy=RoutingPolicy(ranking_limit=4,long_haul_threshold_miles=2800,target_load_factor=.86)
    assert caller('calculate_long_haul_share',{'airport_code':'ANC'},policy).route('What share of ANC flights are long haul?').arguments['threshold_miles']==2800

def test_airport_tool_and_structure_validation():
    with pytest.raises(ToolArgumentsError):
        caller('estimate_unmet_capacity',{'airport_code':'LAX'}).route('What is SFO unmet capacity?')
    with pytest.raises(ToolArgumentsError,match='contradicts'):
        caller('get_airport_profile',{'airport_code':'SFO'}).route('What is SFO unmet capacity?')
    with pytest.raises(UnknownToolError):
        caller('delete_database',{}).route('bad')
    with pytest.raises(ToolArgumentsError):
        caller('rank_airports',{'limit':11}).route('rank')

def test_prose_and_multiple_calls_rejected():
    with pytest.raises(ToolSelectionError):
        AgentToolCaller(FakeClient({'choices':[{'message':{'content':'prose'}}]}),dispatcher=ToolDispatcher(Service())).route('compare')
    payload={'choices':[{'message':{'tool_calls':[{'function':{'name':'get_airport_profile','arguments':'{}'}},{'function':{'name':'get_airport_profile','arguments':'{}'}}]}}]}
    with pytest.raises(ToolSelectionError):
        AgentToolCaller(FakeClient(payload),dispatcher=ToolDispatcher(Service())).route('profiles')
