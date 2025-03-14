(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsLaying (BoundSet BoundSet) Bool)
(declare-fun IsRelaxing (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (IsLaying a b))) (and (forall ((f BoundSet)) (forall ((e BoundSet)) (forall ((d BoundSet)) (=> (IsLaying d e) (IsRelaxing d f))))) (forall ((i BoundSet)) (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsRelaxing g h) (IsLaying g i))))))) (exists ((c BoundSet)) (exists ((a BoundSet)) (IsRelaxing a c))))))
(check-sat)
(get-model)