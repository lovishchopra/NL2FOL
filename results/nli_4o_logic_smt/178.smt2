(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWearingCap (BoundSet) Bool)
(declare-fun IsNotWearingShirt (BoundSet) Bool)
(declare-fun IsLayingOnBench (BoundSet) Bool)
(declare-fun IsRelaxing (BoundSet) Bool)
(declare-fun IsOnBench (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsWearingCap a) (and (IsNotWearingShirt a) (IsLayingOnBench a)))) (and (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (IsNotWearingShirt e) (IsRelaxing f)))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsLayingOnBench g) (IsRelaxing h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsRelaxing i) (IsLayingOnBench j)))) (and (forall ((k BoundSet)) (forall ((l BoundSet)) (=> (IsLayingOnBench k) (IsOnBench l)))) (forall ((n BoundSet)) (forall ((m BoundSet)) (=> (IsOnBench m) (IsLayingOnBench n))))))))) (exists ((c BoundSet)) (and (IsRelaxing c) (IsOnBench c))))))
(check-sat)
(get-model)