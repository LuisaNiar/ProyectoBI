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
                                antecedent_support = get_support(frequent_itemsets, antecedent)
                                consequent_support = get_support(frequent_itemsets, consequent)
                                rule_support = support
                                confidence = rule_support / antecedent_support
                                lift = confidence / consequent_support

                                jaccard_numerator = rule_support
                                jaccard_denominator = antecedent_support + consequent_support - rule_support
                                jaccard_coefficient = jaccard_numerator / jaccard_denominator if jaccard_denominator != 0 else 0

                                profits = calculate_profits(consequent, prices)
                                conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf')

                                if confidence >= min_confidence and lift > 1:
                                    combined_metric = confidence + lift
                                    rules.append((antecedent, consequent, profits, confidence, lift, conviction, combined_metric, jaccard_coefficient, support))
            # Ordenar las reglas primero por ganancias y luego por confianza + lift y luego por soporte de mayor a menor
            rules.sort(key=lambda x: (x[2], x[6], x[7]), reverse=True)
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
        self.frequent_itemsets = F

        # Imprimir itemsets frecuentes
        print(f"{'Itemset':<20} {'Support':<10} {'Relative Support':<10}")
        for itemset, support, rsup in self.frequent_itemsets:
            print(f"{str(itemset):<20} {support:<10} {rsup:<10.2f}")

        # Imprimir reglas de asociación
        print(f"{'Antecedent':<20} {'Consequent':<20} {'Profit':<10} {'Confidence':<10} {'Lift':<10} {'Conviction':<10} {'Confidence + Lift':<10} {'Jaccard Coefficient':<10} {'Support':<10}")
        for rule in self.rules:
            antecedent, consequent, profits, confidence, lift, conviction, combined_metric, jaccard_coefficient, support = rule
            print(f"{str(antecedent):<20} {str(consequent):<20} {profits:<10.2f} {confidence:<10.2f} {lift:<10.2f} {conviction:<10.2f} {combined_metric:<10.2f} {jaccard_coefficient:<10.2f} {support:<10}")

        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        recommendations = defaultdict(float)
        cart_set = set(cart)

        strong_recommendations = []

        for rule in self.rules:
            antecedent, consequent, profits, confidence, lift, conviction, combined_metric, jaccard_coefficient, support = rule
            if set(antecedent).issubset(cart_set):
                strong_recommendations.append((rule, profits))

        # Ordenar las recomendaciones fuertes por confianza y lift
        strong_recommendations.sort(key=lambda x: (x[0][3], x[0][4]), reverse=True)

        for rule, profits in strong_recommendations:
            antecedent, consequent, _, _, _, _, _, _, _ = rule
            for item in consequent:
                if item not in cart_set:
                    recommendations[item] += profits

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        # Imprimir recomendaciones
        print("Recomendaciones:", recommended_items)

        return recommended_items
