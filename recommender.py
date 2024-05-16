import numpy as np
import itertools
from collections import defaultdict


class Recommender:
    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    def train(self, prices, database) -> None:
        def eclat(P, minsup, prefix, F):
            num_transactions = len(database)
            for Xa, t_Xa in P.items():
                support_Xa = len(t_Xa)  # Calcula el soporte directamente
                rsup_Xa = support_Xa / num_transactions  # Calcula el soporte relativo
                if support_Xa >= minsup:
                    F.append((prefix + [Xa], support_Xa, rsup_Xa))  # Almacena el itemset y su soporte
                    Pa = {}
                    for Xb, t_Xb in P.items():
                        if Xb > Xa:
                            t_Xab = t_Xa & t_Xb
                            support_Xab = len(t_Xab)  # Calcula el soporte directamente
                            rsup_Xab = support_Xab / num_transactions  # Calcula el soporte relativo
                            if support_Xab >= minsup:
                                Pa[Xb] = t_Xab
                    if Pa:
                        eclat(Pa, minsup, prefix + [Xa], F)

        def generate_association_rules(frequent_itemsets, min_confidence):
            rules = []
            for itemset, support, rsup in frequent_itemsets:
                if len(itemset) > 1:
                    for i in range(1, len(itemset)):
                        for antecedent in itertools.combinations(itemset, i):
                            consequent = tuple(sorted(set(itemset) - set(antecedent)))
                            if consequent:
                                antecedent_support = get_support(frequent_itemsets, antecedent)
                                rule_support = support
                                confidence = rule_support / antecedent_support
                                consequent_rsup = get_rsup(frequent_itemsets, consequent)
                                lift = confidence / consequent_rsup
                                leverage = rsup - (get_rsup(frequent_itemsets, antecedent) * consequent_rsup)

                                # Calculate profits using antecedent, consequent, and prices
                                profits = calculate_profits(consequent, prices)

                                if 1 > confidence >= min_confidence and 0.5 > leverage > 0 and lift > 1:
                                    rules.append((antecedent, consequent, profits, confidence, lift, leverage))
            return rules

        def calculate_profits(consequent, prices):
            consequent_prices = [prices[item_id] for item_id in consequent]
            return sum(consequent_prices)

        def get_support(frequent_itemsets, itemset):
            for fi, support, _ in frequent_itemsets:
                if set(fi) == set(itemset):
                    return support
            return 0

        def get_rsup(frequent_itemsets, itemset):
            for fi, _, rsup in frequent_itemsets:
                if set(fi) == set(itemset):
                    return rsup
            return 0

        # Definir el umbral mínimo de soporte
        minsup = 500
        min_confidence = 0.75

        # Inicializar P con los ítems únicos y sus transacciones
        P = defaultdict(set)
        for tid, transaction in enumerate(database):
            for item in transaction:
                P[item].add(tid)

        # Calcular los itemsets frecuentes utilizando el algoritmo Eclat
        F = []
        eclat(P, minsup, [], F)

        # Generar reglas de asociación a partir de los itemsets frecuentes
        rules = generate_association_rules(F, min_confidence)

        """
            allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
            :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
            :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
            :return: the object should return itself here (this is actually important!)
        """

        # do something
        self.rules = rules
        # return this object again
        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        """
            makes a recommendation to a specific user

            :param cart: a list with the items in the cart
            :param max_recommendations: maximum number of items that may be recommended
            :return: list of at most max_recommendations items to be recommended
        """
        recommendations = defaultdict(float)

        # Find rules where the cart items are in the antecedent
        for rule in self.rules:
            antecedent, consequent, profits, confidence, lift, leverage = rule
            if set(antecedent).issubset(set(cart)):
                for item in consequent:
                    if item not in cart:  # Avoid recommending items already in the cart
                        recommendations[item] += lift

        # Sort recommendations by lift (or any other measure) and return the top ones
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        # Calculate total profits of recommended items
        total_profits = sum([self.prices[item] for item in recommended_items])

        return recommended_items[42]  # always recommends the same item (requires that there are at least 43 items)


# Initialize the recommender
recommender = Recommender()

