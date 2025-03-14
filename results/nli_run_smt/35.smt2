(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsOnRope (BoundSet) Bool)
(declare-fun IsOverlooking (BoundSet) Bool)
(declare-fun IsGated (BoundSet) Bool)
(declare-fun IsTall (BoundSet) Bool)
(declare-fun IsCooked (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((e BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsOnRope a) (and (IsOverlooking c) (and (IsGated d) (IsTall e)))))))) (exists ((f BoundSet)) (exists ((a BoundSet)) (IsCooked a f))))))
(check-sat)
(get-model)