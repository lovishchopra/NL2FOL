(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsUsedBy (BoundSet BoundSet) Bool)
(declare-fun IsContrary (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun HasDifficultyFinding (BoundSet BoundSet) Bool)
(declare-fun IsNotAThreat (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (exists ((b BoundSet)) (( (and (IsUsedBy b a) (IsContrary c)))) (HasDifficultyFinding c d))))) (exists ((e BoundSet)) (IsNotAThreat e)))))
(check-sat)
(get-model)