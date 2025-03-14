(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun BuysStocks (BoundSet) Bool)
(declare-fun RisksMoney (BoundSet BoundSet) Bool)
(declare-fun HasLittleChanceOfBigProfit (BoundSet) Bool)
(declare-fun BetsOnHorseRacing (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (exists ((c BoundSet)) (and (BuysStocks a) (and (RisksMoney a b) (HasLittleChanceOfBigProfit c)))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (BetsOnHorseRacing f) (BuysStocks g)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (forall ((h BoundSet)) (=> (RisksMoney h i) (BetsOnHorseRacing j))))) (and (forall ((l BoundSet)) (forall ((k BoundSet)) (forall ((m BoundSet)) (=> (BetsOnHorseRacing k) (RisksMoney l m))))) (forall ((n BoundSet)) (forall ((o BoundSet)) (forall ((p BoundSet)) (=> (RisksMoney n o) (HasLittleChanceOfBigProfit p))))))))) (exists ((d BoundSet)) (exists ((e BoundSet)) (and (BetsOnHorseRacing d) (HasLittleChanceOfBigProfit e)))))))
(check-sat)
(get-model)