(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun BreaksThrough (BoundSet BoundSet) Bool)
(declare-fun Rides (BoundSet BoundSet) Bool)
(declare-fun IsInDaytime (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (BreaksThrough c d) (Rides a b)))))) (and (forall ((i BoundSet)) (forall ((h BoundSet)) (forall ((g BoundSet)) (forall ((j BoundSet)) (=> (BreaksThrough g h) (Rides i j)))))) (and (forall ((m BoundSet)) (forall ((k BoundSet)) (forall ((l BoundSet)) (=> (BreaksThrough k l) (IsInDaytime m))))) (forall ((o BoundSet)) (forall ((n BoundSet)) (=> (Rides n o) (IsInDaytime n))))))) (exists ((b BoundSet)) (exists ((a BoundSet)) (and (Rides a b) (IsInDaytime a)))))))
(check-sat)
(get-model)