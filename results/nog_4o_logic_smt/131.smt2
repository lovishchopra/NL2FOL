(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun TendingTo (BoundSet BoundSet) Bool)
(declare-fun Includes (BoundSet BoundSet) Bool)
(declare-fun TakingCare (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((d BoundSet)) (exists ((a BoundSet)) (exists ((c BoundSet)) (and (TendingTo a b) (and (Includes b c) (Includes b d))))))) (exists ((b BoundSet)) (exists ((a BoundSet)) (TakingCare a b))))))
(check-sat)
(get-model)