(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsLeadsTo (BoundSet BoundSet) Bool)
(declare-fun IsImprovedSoilMoisture (BoundSet) Bool)
(declare-fun IsCreating (BoundSet) Bool)
(declare-fun IsSafer (BoundSet) Bool)
(declare-fun IsMorePeaceful (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsLeadsTo a b) (IsImprovedSoilMoisture c))))) (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (IsLeadsTo e f) (IsCreating e))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsCreating a) (and (IsSafer d) (IsMorePeaceful d))))))))
(check-sat)
(get-model)