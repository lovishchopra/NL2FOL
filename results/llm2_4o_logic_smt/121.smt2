(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun Sings (BoundSet BoundSet) Bool)
(declare-fun From (BoundSet BoundSet) Bool)
(declare-fun IsFilledWith (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (Sings a b) (From b c))))) (and (forall ((g BoundSet)) (forall ((f BoundSet)) (forall ((i BoundSet)) (forall ((h BoundSet)) (=> (Sings f g) (IsFilledWith h i)))))) (and (forall ((l BoundSet)) (forall ((m BoundSet)) (forall ((k BoundSet)) (forall ((j BoundSet)) (=> (IsFilledWith j k) (Sings l m)))))) (forall ((q BoundSet)) (forall ((o BoundSet)) (forall ((n BoundSet)) (forall ((p BoundSet)) (=> (IsFilledWith n o) (From p q))))))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (IsFilledWith d e))))))
(check-sat)
(get-model)