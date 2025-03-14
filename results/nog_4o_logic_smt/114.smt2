(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun boys () UnboundSet)
(declare-fun IsSmall (BoundSet) Bool)
(declare-fun IsInBlueSoccerUniforms (BoundSet) Bool)
(declare-fun HasHands (BoundSet) Bool)
(declare-fun IsInAdultSizedBathroom (BoundSet) Bool)
(declare-fun IsTwo (UnboundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsSmall b) (and (IsInBlueSoccerUniforms b) (and (HasHands b) (IsInAdultSizedBathroom a)))))) (exists ((e BoundSet)) (and (IsTwo boys) (HasHands e))))))
(check-sat)
(get-model)