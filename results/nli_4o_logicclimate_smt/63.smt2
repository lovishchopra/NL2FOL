(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun HasNotAccelerated (BoundSet) Bool)
(declare-fun IsNotSignificantProblem (BoundSet) Bool)
(declare-fun IsSignificantProblem (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (HasNotAccelerated a)) (and (forall ((c BoundSet)) (forall ((d BoundSet)) (=> (HasNotAccelerated c) (IsNotSignificantProblem d)))) (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (IsNotSignificantProblem e) (HasNotAccelerated f)))))) (exists ((b BoundSet)) (not (IsSignificantProblem b))))))
(check-sat)
(get-model)