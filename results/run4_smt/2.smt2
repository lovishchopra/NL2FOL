(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun a () UnboundSet)
(declare-fun b () UnboundSet)
(declare-fun c () UnboundSet)
(declare-fun IsKnitted (UnboundSet) Bool)
(declare-fun IsFor (UnboundSet) Bool)
(declare-fun IsWorn (UnboundSet) Bool)
(declare-fun MakesHappy (UnboundSet) Bool)
(assert (not (=> (and (IsKnitted a) (IsFor b)) (and (IsWorn c) (MakesHappy a)))))
(check-sat)
(get-model)