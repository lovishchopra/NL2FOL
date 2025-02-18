(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun WillRiseMoreThan3Feet (BoundSet) Bool)
(declare-fun IsEconomicImpactSignificant (BoundSet) Bool)
(declare-fun IsEnvironmentalImpactSignificant (BoundSet) Bool)
(declare-fun IsSignificant (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (WillRiseMoreThan3Feet a)) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (WillRiseMoreThan3Feet f) (IsEconomicImpactSignificant g)))) (and (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (WillRiseMoreThan3Feet h) (IsEnvironmentalImpactSignificant i)))) (forall ((k BoundSet)) (forall ((j BoundSet)) (=> (WillRiseMoreThan3Feet j) (IsSignificant k))))))) (exists ((e BoundSet)) (exists ((c BoundSet)) (and (exists ((d BoundSet)) (( (and (IsEconomicImpactSignificant c) (IsEnvironmentalImpactSignificant d)))) (IsSignificant e)))))))
(check-sat)
(get-model)