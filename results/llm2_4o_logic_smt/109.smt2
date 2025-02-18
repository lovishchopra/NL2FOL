(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInKilt (BoundSet) Bool)
(declare-fun HasThrownPole (BoundSet) Bool)
(declare-fun Watch (BoundSet BoundSet) Bool)
(declare-fun AreInKilts (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsInKilt a) (HasThrownPole a))) (and (forall ((g BoundSet)) (forall ((f BoundSet)) (forall ((h BoundSet)) (=> (IsInKilt f) (Watch g h))))) (and (forall ((i BoundSet)) (forall ((k BoundSet)) (forall ((j BoundSet)) (=> (Watch i j) (IsInKilt k))))) (and (forall ((l BoundSet)) (forall ((m BoundSet)) (=> (IsInKilt l) (AreInKilts m)))) (and (forall ((o BoundSet)) (forall ((n BoundSet)) (=> (AreInKilts n) (IsInKilt o)))) (and (forall ((q BoundSet)) (forall ((r BoundSet)) (forall ((p BoundSet)) (=> (HasThrownPole p) (Watch q r))))) (and (forall ((s BoundSet)) (forall ((u BoundSet)) (forall ((t BoundSet)) (=> (Watch s t) (HasThrownPole u))))) (forall ((v BoundSet)) (forall ((w BoundSet)) (=> (AreInKilts v) (HasThrownPole w))))))))))) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (Watch c d) (AreInKilts d)))))))
(check-sat)
(get-model)