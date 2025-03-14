(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun TalksWith (BoundSet BoundSet) Bool)
(declare-fun IsInFrontOfTeam (BoundSet) Bool)
(declare-fun IsInFrontOfCrowd (BoundSet) Bool)
(declare-fun IsInFrontOfEveryone (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (TalksWith a b) (and (IsInFrontOfTeam a) (IsInFrontOfCrowd a))))) (and (forall ((g BoundSet)) (=> (IsInFrontOfCrowd g) (IsInFrontOfEveryone g))) (forall ((h BoundSet)) (=> (IsInFrontOfEveryone h) (IsInFrontOfCrowd h))))) (exists ((a BoundSet)) (exists ((e BoundSet)) (and (TalksWith a e) (IsInFrontOfEveryone a)))))))
(check-sat)
(get-model)