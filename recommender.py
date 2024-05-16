import numpy as np
import itertools
from collections import defaultdict
from multiprocessing import Pool, cpu_count

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
            def process_itemset(itemset_support_rsup):
                itemset, support, rsup = itemset_support_rsup
                rules = []
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
                                    rules.append((antecedent, consequent, profits, confidence, lift, leverage))
                return rules

            with Pool(cpu_count()) as pool:
                results = pool.map(process_itemset, frequent_itemsets)
                rules = [rule for sublist in results for rule in sublist]
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

        # Definir el umbral mínimo de soporte como el 20% de la longitud de la lista de precios
        minsup = max(1, int(0.2 * len(prices)))
        min_confidence = 0.1

        # Inicializar P con los ítems únicos y sus transacciones
        P = defaultdict(set)
        for tid, transaction in enumerate(database):
            for item in transaction:
                P[item].add(tid)

        num_transactions = len(database)

        # Calcular los itemsets frecuentes utilizando el algoritmo Eclat
        F = []
        eclat(P, minsup, [], F, num_transactions)

        # Generar reglas de asociación a partir de los itemsets frecuentes
        rules = generate_association_rules(F, min_confidence)

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
