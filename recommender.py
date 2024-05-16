import numpy as np
import itertools
from collections import defaultdict

class Recommender:
    def train(self, prices, database) -> None:
        def eclat(itemsets, minsup, prefix, frequent_itemsets, num_transactions):
            for item, transactions in itemsets.items():
                support_count = len(transactions)
                if support_count >= minsup:
                    relative_support = support_count / num_transactions
                    frequent_itemsets.append((prefix + [item], support_count, relative_support))
                    new_itemsets = {item_next: transactions & transactions_next for item_next, transactions_next in itemsets.items() if item_next > item}
                    if new_itemsets:
                        eclat(new_itemsets, minsup, prefix + [item], frequent_itemsets, num_transactions)

        def generate_association_rules(frequent_itemsets, min_confidence):
            rules = []
            for itemset, support, relative_support in frequent_itemsets:
                if len(itemset) > 1:
                    for i in range(1, len(itemset)):
                        for antecedent in itertools.combinations(itemset, i):
                            consequent = tuple(sorted(set(itemset) - set(antecedent)))
                            if consequent:
                                antecedent_support = get_support(frequent_itemsets, antecedent)
                                confidence = support / antecedent_support
                                consequent_rsup = get_rsup(frequent_itemsets, consequent)
                                lift = confidence / consequent_rsup
                                leverage = relative_support - (antecedent_support / len(database) * consequent_rsup)

                                profits = calculate_profits(consequent, prices)

                                if confidence >= min_confidence and leverage > 0 and lift > 1:
                                    rules.append((antecedent, consequent, profits, confidence, lift, leverage))
            return rules

        def calculate_profits(consequent, prices):
            return sum(prices[item_id] for item_id in consequent)

        def get_support(frequent_itemsets, itemset):
            itemset_set = set(itemset)
            for fi, support, _ in frequent_itemsets:
                if set(fi) == itemset_set:
                    return support
            return 0

        def get_rsup(frequent_itemsets, itemset):
            itemset_set = set(itemset)
            for fi, _, rsup in frequent_itemsets:
                if set(fi) == itemset_set:
                    return rsup
            return 0

        minsup = max(1, int(0.2 * len(prices)))
        min_confidence = 0.1

        item_transactions = defaultdict(set)
        for transaction_id, transaction in enumerate(database):
            for item in transaction:
                item_transactions[item].add(transaction_id)

        num_transactions = len(database)

        frequent_itemsets = []
        eclat(item_transactions, minsup, [], frequent_itemsets, num_transactions)

        rules = generate_association_rules(frequent_itemsets, min_confidence)

        self.rules = rules
        self.prices = prices
        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        recommendations = defaultdict(float)
        cart_set = set(cart)

        for rule in self.rules:
            antecedent, consequent, profits, confidence, lift, leverage = rule
            if set(antecedent).issubset(cart_set):
                for item in consequent:
                    if item not in cart_set:
                        recommendations[item] += lift

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        return recommended_items

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        return recommended_items
