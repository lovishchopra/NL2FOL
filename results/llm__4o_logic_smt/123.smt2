(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAsian (BoundSet) Bool)
(declare-fun IsWearing (BoundSet BoundSet) Bool)
(declare-fun IsSittingOn (BoundSet BoundSet) Bool)
(declare-fun IsResting (BoundSet BoundSet) Bool)
(declare-fun IsSeated (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (exists ((e BoundSet)) (and (IsAsian a) (and (IsWearing a b) (and (IsSittingOn a c) (IsResting d e))))))))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsSeated g) (IsAsian h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (IsSeated i) (IsWearing j k))))) (and (forall ((n BoundSet)) (forall ((l BoundSet)) (forall ((m BoundSet)) (=> (IsSittingOn l m) (IsSeated n))))) (and (forall ((q BoundSet)) (forall ((o BoundSet)) (forall ((p BoundSet)) (=> (IsSeated o) (IsSittingOn p q))))) (forall ((s BoundSet)) (forall ((r BoundSet)) (forall ((t BoundSet)) (=> (IsSeated r) (IsResting s t)))))))))) (exists ((f BoundSet)) (IsSeated f)))))
(check-sat)
(get-model)