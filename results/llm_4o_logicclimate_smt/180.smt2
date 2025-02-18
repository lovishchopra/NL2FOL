(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInSuit (BoundSet) Bool)
(declare-fun IsCheering (BoundSet) Bool)
(declare-fun IsInLibrary (BoundSet) Bool)
(declare-fun IsInFrontOfChildren (BoundSet) Bool)
(declare-fun IsNearChildren (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (and (IsInSuit b) (and (IsCheering b) (and (IsInLibrary b) (IsInFrontOfChildren b))))) (and (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsCheering f) (IsNearChildren g)))) (and (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (IsInLibrary h) (IsCheering i)))) (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (IsInLibrary j) (IsNearChildren k))))))) (exists ((d BoundSet)) (and (IsCheering d) (IsNearChildren d))))))
(check-sat)
(get-model)