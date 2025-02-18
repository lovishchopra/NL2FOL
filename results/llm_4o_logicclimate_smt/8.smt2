(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsArrogant (BoundSet) Bool)
(declare-fun IsZealous (BoundSet) Bool)
(declare-fun IsSeenAs (BoundSet BoundSet) Bool)
(declare-fun SeenBy (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsArrogant a) (IsZealous a))) (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((c BoundSet)) (and (IsSeenAs b c) (SeenBy a b))))))))
(check-sat)
(get-model)