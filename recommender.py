import numpy as np
import itertools
import pandas as pd
from collections import defaultdict


class Recommender:

    def train(self, prices, database) -> None:
        def eclat(P, minsup, prefix, F):
            num_transactions = len(database)
            for Xa, t_Xa in P.items():
                support_Xa = len(t_Xa)
                rsup_Xa = support_Xa / num_transactions
                if support_Xa >= minsup:
                    F.append((prefix + [Xa], support_Xa, rsup_Xa))
                    Pa = {}
                    for Xb, t_Xb in P.items():
                        if Xb > Xa:
                            t_Xab = t_Xa & t_Xb
                            support_Xab = len(t_Xab)
                            rsup_Xab = support_Xab / num_transactions
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

                                if 1 > confidence >= min_confidence and leverage > 0 and lift > 1:
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

        # Definir el umbral mínimo de soporte como el 20% de la longitud de la lista de precios
        minsup = max(1, int(0.2 * len(prices)))
        min_confidence = 0.1

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

        # Convertir el conjunto de itemsets frecuentes a un DataFrame
        df_frequent_itemsets = pd.DataFrame(F, columns=['Itemset', 'Support', 'Relative Support'])

        # Imprimir el DataFrame
        print("Conjunto de itemsets frecuentes:")
        print(df_frequent_itemsets)

        # Convertir el conjunto de reglas a un DataFrame
        df_rules = pd.DataFrame(rules,
                                columns=['Antecedent', 'Consequent', 'Profits', 'Confidence', 'Lift', 'Leverage'])

        # Filtrar reglas con leverage negativo
        df_rules = df_rules[df_rules['Leverage'] >= 0]

        # Imprimir el DataFrame de reglas filtrado
        print("Reglas de asociación:")
        print(df_rules)

        # Do something
        self.rules = rules
        self.prices = prices

        # Return this object again
        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:

        recommendations = defaultdict(float)

        # Find rules where the cart items are in the antecedent
        for rule in self.rules:
            antecedent, consequent, profits, confidence, lift, leverage = rule
            if set(antecedent).issubset(set(cart)):
                for item in consequent:
                    recommendations[item] += lift

        # Sort recommendations by lift (or any other measure) and return the top ones
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in sorted_recommendations[:max_recommendations]]

        print("Sorted Recommendations:")
        print(sorted_recommendations)
        print("Recommended Items:")
        print(recommended_items)

        return recommended_items
