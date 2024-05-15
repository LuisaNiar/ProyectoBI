from collections import defaultdict
from itertools import combinations

class Recommender:

    def __init__(self):
        self.compra = None
        self.base_datos = None
        self.soporte = None

    def train(self, prices, database, umbral_soporte=0.1) -> None:
        """
        Permite que el recomendador aprenda qué artículos existen, qué precios tienen y qué artículos se han comprado juntos en el pasado.
        :param prices: una lista de precios en USD para los artículos (los identificadores de los artículos van del 0 a la longitud de esta lista - 1)
        :param database: una lista de listas de identificadores de artículos que se han comprado juntos. Cada entrada corresponde a una transacción.
        :param umbral_soporte: umbral de soporte mínimo para considerar un conjunto de ítems.
        :return: el objeto debe devolverse a sí mismo aquí (¡esto es realmente importante!)
        """
        self.compra = prices
        self.base_datos = database
        transacciones = len(database)
        self.soporte = defaultdict(int)

        # Calcular el soporte para cada artículo individual
        for transaccion in database:
            for item in transaccion:
                self.soporte[frozenset([item])] += 1

        # Calcular el soporte para conjuntos de ítems de tamaño mayor a 1
        k = 2
        while True:
            nuevos_conjuntos = defaultdict(int)
            for transaccion in database:
                for conjunto in combinations(transaccion, k):
                    nuevos_conjuntos[frozenset(conjunto)] += 1

            nuevos_conjuntos = {conjunto: count / transacciones for conjunto, count in nuevos_conjuntos.items() if count / transacciones >= umbral_soporte}

            if nuevos_conjuntos:
                self.soporte.update(nuevos_conjuntos)
                k += 1
            else:
                break

        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
        Hace una recomendación a un usuario específico.

        :param cart: una lista con los artículos en el carrito.
        :param max_recommendations: número máximo de artículos que se pueden recomendar.
        :return: lista de hasta `max_recommendations` artículos que se recomendarán.
        """
        items_en_carrito = set(cart)
        recomendaciones = []

        for conjunto, soporte in sorted(self.soporte.items(), key=lambda x: x[1], reverse=True):
            if len(recomendaciones) >= max_recommendations:
                break

            if not conjunto.intersection(items_en_carrito):
                recomendaciones.extend(conjunto)

        return list(recomendaciones)[:max_recommendations]
