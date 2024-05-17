import numpy as np
import itertools
from collections import defaultdict

class Recommender:
    def train(self, prices, database) -> None:
        def eclat(P, minsup, prefix, F, num_transactions):
            for Xa, t_Xa in P.items():
                support_Xa = len(t_Xa)
                rsup_Xa = support_Xa / num_transactions
                if rsup_Xa > 0:  # Filtrar elementos con soporte relativo 0
                    if support_Xa >= minsup:
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
                                profits = calculate_profits(consequent, prices)
                                lift = get_lift(frequent_itemsets, antecedent, consequent)
                                if lift > 1:
                                    combined_metric = profits + lift
                                    rules.append((antecedent, consequent, profits, lift, combined_metric, support))
            # Ordenar las reglas primero por ganancias y luego por lift de mayor a menor
            rules.sort(key=lambda x: (x[2], x[3]), reverse=True)
            return rules

        def calculate_profits(consequent, prices):
            return sum(prices[item_id] for item_id in consequent)

        def get_lift(frequent_itemsets, antecedent, consequent):
            antecedent_support = get_support(frequent_itemsets, antecedent)
            consequent_support = get_support(frequent_itemsets, consequent)
            joint_support = get_support(frequent_itemsets, antecedent + consequent)
            return joint_support / (antecedent_support * consequent_support)

        def get_support(frequent_itemsets, itemset):
            itemset_set = set(itemset)
            for fi, support, _ in frequent_itemsets:
                if set(fi) == itemset_set:
                    return support
            return 0

        # Definir el umbral mínimo de soporte como el 20% de la longitud de la lista de precios
        minsup = max(1, int(0.2 * len(prices)))

        # Calcular los itemsets frecuentes utilizando el algoritmo Eclat
        F = []
        num_transactions = len(database)
        P = defaultdict(set)
        for tid, transaction in enumerate(database):
            for item in transaction:
                P[item].add(tid)
        eclat(P, minsup, [], F, num_transactions)

        # Generar reglas de asociación a partir de los itemsets frecuentes
        rules = generate_association_rules(F, min_confidence=0.1)

        self.rules = rules
        self.prices = prices

        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        recommendations = defaultdict(float)
        cart_set = set(cart)

        for rule in self.rules:
            antecedent, consequent, profits, lift, combined_metric, support = rule
            if set(antecedent).issubset(cart_set):
                for item in consequent:
                    if item not in cart_set:
                        recommendations[item] += combined_metric

        # Ordenar recomendaciones por la métrica combinada
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        return recommended_items
