from collections import defaultdict
from itertools import combinations

class Recommender:

    def __init__(self):
        self.compra = None
        self.base_datos = None
        self.soporte = None

    def train(self, prices, database, umbral_soporte=0.1) -> None:
        """
        Permite que el recomendador aprenda qué artículos existen, qué precios tienen y qué artículos se han comprado juntos en el pasado
        :param prices: una lista de precios en USD para los artículos (los identificadores de los artículos van del 0 a la longitud de esta lista - 1)
        :param database: una lista de listas de identificadores de artículos que se han comprado juntos. Cada entrada corresponde a una transacción
        :param umbral_soporte: umbral de soporte mínimo para considerar un conjunto de ítems
        :return: el objeto debe devolverse a sí mismo aquí (¡esto es realmente importante!)
        """
        self.compra = prices
        self.base_datos = database
        self.soporte = self._calcular_soporte(database, umbral_soporte)
        return self

    def _calcular_soporte(self, base_datos, umbral_soporte):
        """
        Calcula el soporte de los conjuntos de ítems en la base de datos de transacciones.
        """
        transacciones = len(base_datos)
        soporte = defaultdict(int)

        # Calcular el soporte para cada artículo individual
        for transaccion in base_datos:
            for item in transaccion:
                soporte[frozenset([item])] += 1

        # Calcular el soporte para conjuntos de ítems de tamaño mayor a 1
        k = 2
        while True:
            nuevos_conjuntos = defaultdict(int)
            for transaccion in base_datos:
                for conjunto in combinations(transaccion, k):
                    nuevos_conjuntos[frozenset(conjunto)] += 1

            nuevos_conjuntos = {conjunto: count / transacciones for conjunto, count in nuevos_conjuntos.items() if count / transacciones >= umbral_soporte}

            if nuevos_conjuntos:
                soporte.update(nuevos_conjuntos)
                k += 1
            else:
                break

        return soporte

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
        Hace una recomendación a un usuario específico

        :param cart: una lista con los artículos en el carrito
        :param max_recommendations: número máximo de artículos que se pueden recomendar
        :return: lista de hasta `max_recommendations` artículos que se recomendarán
        """
        items_en_carrito = set(cart)
        recomendaciones = []

        for conjunto, soporte in sorted(self.soporte.items(), key=lambda x: x[1], reverse=True):
            if len(recomendaciones) >= max_recommendations:
                break

            if not conjunto.intersection(items_en_carrito):
                recomendaciones.extend(conjunto)

        return list(recomendaciones)[:max_recommendations]

# Ejemplo de uso
if __name__ == "__main__":
    precios = [10, 20, 15, 25, 30]
    base_datos = [[0, 1, 2], [0, 1, 3], [1, 3, 4], [2, 3, 4], [0, 2, 3]]
    
    recomendador = Recommender()
    recomendador.train(precios, base_datos, umbral_soporte=0.2)
    
    carrito = [0, 1]
    recomendaciones = recomendador.get_recommendations(carrito, 3)
    print("Recomendaciones:", recomendaciones)
