from src.agent import RoutingPolicy, ToolDispatcher
from src.router import DeterministicFallbackRouter

class Service:
    def rank_airports(self,r): return r
    def compare_airports(self,r): return r
    def calculate_long_haul_share(self,r): return r
    def estimate_unmet_capacity(self,r): return r
    def get_airport_profile(self,r): return r

def fallback(policy=None): return DeterministicFallbackRouter(ToolDispatcher(Service()),policy=policy)

def test_fallback_preserves_explicit_overrides():
    router=fallback()
    assert router.route('Rank the top 3 New England airports').arguments['limit']==3
    assert router.route('What share of ANC flights are long haul using 2,500 miles?').arguments['threshold_miles']==2500
    assert router.route('Estimate SFO unmet capacity at a target load factor of 85%').arguments['target_load_factor']==.85

def test_fallback_uses_shared_runtime_policy_defaults():
    router=fallback(RoutingPolicy(ranking_limit=4,long_haul_threshold_miles=2700,target_load_factor=.87))
    assert router.route('Rank New England expansion candidates').arguments['limit']==4
    assert router.route('What share of ANC flights are long haul?').arguments['threshold_miles']==2700
    assert router.route('What is SFO unmet capacity?').arguments['target_load_factor']==.87
