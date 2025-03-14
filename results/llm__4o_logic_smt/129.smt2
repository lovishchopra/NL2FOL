(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun { () BoundSet)
(declare-fun | () BoundSet)
(declare-fun IsDusty (BoundSet) Bool)
(declare-fun Runs (BoundSet) Bool)
(declare-fun IsDirtPath (BoundSet) Bool)
(declare-fun IsGrass (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(declare-fun IsRunning (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((c BoundSet)) (and (IsDusty a) (and (Runs a) (and (IsDirtPath b) (IsGrass c))))))) (and (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsDusty e) (IsOutside f)))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsOutside g) (IsDusty h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsDusty i) (IsRunning j)))) (and (forall ((l BoundSet)) (forall ((k BoundSet)) (=> (IsRunning k) (IsDusty l)))) (and (forall ((n BoundSet)) (forall ((m BoundSet)) (=> (Runs m) (IsOutside n)))) (and (forall ((o BoundSet)) (forall ((p BoundSet)) (=> (IsOutside o) (Runs p)))) (and (forall ((q BoundSet)) (forall ((r BoundSet)) (=> (Runs q) (IsRunning r)))) (and (forall ((s BoundSet)) (forall ((t BoundSet)) (=> (IsRunning s) (Runs t)))) (and (forall ((u BoundSet)) (forall ((v BoundSet)) (=> (IsOutside u) (IsDirtPath v)))) (and (forall ((x BoundSet)) (forall ((w BoundSet)) (=> (IsRunning w) (IsDirtPath x)))) (and (forall ((z BoundSet)) (forall ((y BoundSet)) (=> (IsOutside y) (IsGrass z)))) (=> (IsRunning {) (IsGrass |)))))))))))))) (exists ((d BoundSet)) (and (IsOutside d) (IsRunning d))))))
(check-sat)
(get-model)