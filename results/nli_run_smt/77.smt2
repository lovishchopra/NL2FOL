(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInGreenNunUniform (BoundSet) Bool)
(declare-fun IsElderly (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (IsInGreenNunUniform b)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsElderly c) (IsInGreenNunUniform a)))))))
(check-sat)
(get-model)