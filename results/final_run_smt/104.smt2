(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsRunTowards (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (IsRunTowards b a))) (exists ((c BoundSet)) (exists ((a BoundSet)) (IsRunTowards c a))))))
(check-sat)
(get-model)