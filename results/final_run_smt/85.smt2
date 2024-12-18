(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsImprisoned (BoundSet) Bool)
(declare-fun EmbezzledMoney (BoundSet) Bool)
(declare-fun IsDishonest (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((c BoundSet)) (and (IsImprisoned b) (EmbezzledMoney c)))) (exists ((a BoundSet)) (IsDishonest a)))))
(check-sat)
(get-model)