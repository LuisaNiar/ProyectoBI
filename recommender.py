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
                                consequent_support = get_support(frequent_itemsets, consequent)
                                consequent_rsup = consequent_support / len(database)
                                lift = confidence / consequent_rsup
                                leverage = rsup - (antecedent_support / len(database) * consequent_rsup)

                                # Calculate profits
                                profits = sum(prices[item_id] for item_id in consequent)

                                # Calculate odds ratio
                                support_both = support / len(database)
                                odds_ratio = (support_both * (1 - rsup - consequent_rsup + support_both)) / ((rsup - support_both) * (consequent_rsup - support_both))

                                # Calculate Jaccard index
                                jaccard = support / (antecedent_support + consequent_support - support)

                                if confidence >= min_confidence and leverage > 0 and lift > 1:
                                    rules.append((antecedent, consequent, profits, confidence, lift, leverage, odds_ratio, jaccard))
            return rules

        def get_support(frequent_itemsets, itemset):
            itemset_set = set(itemset)
            for fi, support, _ in frequent_itemsets:
                if set(fi) == itemset_set:
                    return support
            return 0

        # Define minimum support and confidence
        minsup = max(1, int(0.2 * len(prices)))
        min_confidence = 0.1

        # Initialize P with unique items and their transactions
        P = defaultdict(set)
        for tid, transaction in enumerate(database):
            for item in transaction:
                P[item].add(tid)

        num_transactions = len(database)

        # Calculate frequent itemsets using the Eclat algorithm
        F = []
        eclat(P, minsup, [], F, num_transactions)

        # Generate association rules from the frequent itemsets
        rules = generate_association_rules(F, min_confidence)

        # Filter out rules with negative leverage
        rules = [rule for rule in rules if rule[5] >= 0]

        # Sort the rules first by profits, then by confidence + lift, then by support
        rules.sort(key=lambda x: (x[2], x[3] + x[4], x[1]), reverse=True)

        self.rules = rules
        self.prices = prices

        # Print frequent itemsets and rules (optional, can be commented out)
        df_frequent_itemsets = pd.DataFrame(F, columns=['Itemset', 'Support', 'Relative Support'])
        print("Frequent itemsets:")
        print(df_frequent_itemsets)

        df_rules = pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Profits', 'Confidence', 'Lift', 'Leverage', 'Odds Ratio', 'Jaccard'])
        print("Association rules:")
        print(df_rules)

        return self

    def get_recommendations(self, cart: list, max_recommendations: int = 3) -> list:
        recommendations = defaultdict(float)
        cart_set = set(cart)

        for rule in self.rules:
            antecedent, consequent, profits, _, _, _, _, _ = rule
            if set(antecedent).issubset(cart_set):
                for item in consequent:
                    if item not in cart_set:
                        recommendations[item] = profits

        # Filter recommendations by price and select the top items
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: self.prices[x[0]], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        print("Sorted Recommendations by Price:")
        print(sorted_recommendations)
        print("Recommended Items by Price:")
        print(recommended_items)

        return recommended_items
