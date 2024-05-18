import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
class Recommender:
    def train(self, prices, database) -> None:
        def eclat(P, minsup, prefix, F, num_transactions):
            for Xa, t_Xa in P.items():
                support_Xa = len(t_Xa)
                if support_Xa >= minsup:
                    rsup_Xa = support_Xa / num_transactions
                    F.append((prefix + [Xa], support_Xa, rsup_Xa))
                    Pa = {Xb: t_Xa & t_Xb for Xb, t_Xb in P.items() if Xb > Xa}
                    if Pa:
                        eclat(Pa, minsup, prefix + [Xa], F, num_transactions)
        def generate_association_rules(frequent_itemsets, min_confidence):
            rules = []
            for itemset, support, rsup in frequent_itemsets:
                if len(itemset) > 1:
                    for i in range(1, len(itemset)):
                        for antecedent in itertools.combinations(itemset, i):
                            consequent = tuple(sorted(set(itemset) - set(antecedent)))
                            if consequent:
                                antecedent_support = get_support(frequent_itemsets, antecedent)
                                confidence = support / antecedent_support
                                consequent_rsup = get_rsup(frequent_itemsets, consequent)
                                lift = confidence / consequent_rsup
                                leverage = rsup - (antecedent_support / len(database) * consequent_rsup)

                                profits = calculate_profits(consequent, prices)

                                if confidence >= min_confidence and leverage > 0 and lift > 1:
