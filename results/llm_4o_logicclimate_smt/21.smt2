(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun { () BoundSet)
(declare-fun | () BoundSet)
(declare-fun IsDrivenBy (BoundSet BoundSet) Bool)
(declare-fun IsMotivatedBy (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (IsDrivenBy a b) (and (IsDrivenBy a c) (IsDrivenBy a d))))))) (and (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsDrivenBy e f) (IsMotivatedBy e f)))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsMotivatedBy g h) (IsDrivenBy g h)))) (and (forall ((j BoundSet)) (forall ((i BoundSet)) (forall ((k BoundSet)) (=> (IsDrivenBy i j) (IsMotivatedBy i k))))) (and (forall ((m BoundSet)) (forall ((n BoundSet)) (forall ((l BoundSet)) (=> (IsDrivenBy l m) (IsMotivatedBy l n))))) (and (forall ((o BoundSet)) (forall ((p BoundSet)) (=> (IsDrivenBy o p) (IsMotivatedBy o p)))) (and (forall ((r BoundSet)) (forall ((q BoundSet)) (=> (IsMotivatedBy q r) (IsDrivenBy q r)))) (and (forall ((u BoundSet)) (forall ((t BoundSet)) (forall ((s BoundSet)) (=> (IsMotivatedBy s t) (IsDrivenBy s u))))) (and (forall ((v BoundSet)) (forall ((w BoundSet)) (forall ((x BoundSet)) (=> (IsMotivatedBy v w) (IsDrivenBy v x))))) (and (forall ((y BoundSet)) (forall ((z BoundSet)) (=> (IsDrivenBy y z) (IsMotivatedBy y z)))) (=> (IsMotivatedBy { |) (IsDrivenBy { |)))))))))))) (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((d BoundSet)) (exists ((c BoundSet)) (and (IsMotivatedBy a b) (and (IsMotivatedBy a c) (IsMotivatedBy a d))))))))))
(check-sat)
(get-model)