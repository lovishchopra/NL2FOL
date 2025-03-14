(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsArrogant (BoundSet) Bool)
(declare-fun IsZealous (BoundSet) Bool)
(declare-fun SeeAs (BoundSet BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsArrogant a) (IsZealous a))) (forall ((d BoundSet)) (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsZealous d) (SeeAs d e f)))))) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (SeeAs a b c)))))))
(check-sat)
(get-model)