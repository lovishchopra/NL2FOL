(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsEasyToKeep (BoundSet) Bool)
(declare-fun RequiresNoMaintenance (BoundSet) Bool)
(declare-fun IsInexpensive (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (IsEasyToKeep b)) (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsEasyToKeep f) (RequiresNoMaintenance g))))) (exists ((d BoundSet)) (exists ((e BoundSet)) (and (IsInexpensive d) (RequiresNoMaintenance e)))))))
(check-sat)
(get-model)