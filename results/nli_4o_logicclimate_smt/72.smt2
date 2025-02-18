(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsSuspectedByScientists (BoundSet) Bool)
(declare-fun IsTrappedInSiberianIce (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun CouldCauseHumanSickness (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (exists ((b BoundSet)) (( (and (IsSuspectedByScientists a) (IsTrappedInSiberianIce a)))) (and (IsSuspectedByScientists b) (IsTrappedInSiberianIce b)))) (and (forall ((g BoundSet)) (=> (IsSuspectedByScientists g) (CouldCauseHumanSickness g))) (and (forall ((h BoundSet)) (=> (IsSuspectedByScientists h) (CouldCauseHumanSickness h))) (and (forall ((i BoundSet)) (=> (CouldCauseHumanSickness i) (IsSuspectedByScientists i))) (and (forall ((j BoundSet)) (=> (IsTrappedInSiberianIce j) (CouldCauseHumanSickness j))) (and (forall ((l BoundSet)) (forall ((k BoundSet)) (=> (IsTrappedInSiberianIce k) (CouldCauseHumanSickness l)))) (forall ((m BoundSet)) (=> (IsTrappedInSiberianIce m) (CouldCauseHumanSickness m))))))))) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (CouldCauseHumanSickness a) (CouldCauseHumanSickness b)))))))
(check-sat)
(get-model)