(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSmall (BoundSet) Bool)
(declare-fun IsInPinkDress (BoundSet) Bool)
(declare-fun IsPlayingDrumPads (BoundSet) Bool)
(declare-fun UsesSticks (BoundSet) Bool)
(declare-fun IsPlayingElectronicDrums (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsSmall a) (and (IsInPinkDress a) (and (IsPlayingDrumPads a) (UsesSticks a))))) (and (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsPlayingElectronicDrums g) (IsSmall h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsPlayingDrumPads i) (IsPlayingElectronicDrums j)))) (forall ((l BoundSet)) (forall ((k BoundSet)) (=> (IsPlayingElectronicDrums k) (IsPlayingDrumPads l))))))) (exists ((e BoundSet)) (IsPlayingElectronicDrums e)))))
(check-sat)
(get-model)