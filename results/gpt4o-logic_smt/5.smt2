(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun CrossedRoad (BoundSet) Bool)
(declare-fun AttendingShoeSale (BoundSet) Bool)
(declare-fun IsBlonde (BoundSet) Bool)
(declare-fun IsAttractedByShoeSales (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (CrossedRoad a) (AttendingShoeSale a))) (exists ((d BoundSet)) (and (IsBlonde d) (IsAttractedByShoeSales d))))))
(check-sat)
(get-model)