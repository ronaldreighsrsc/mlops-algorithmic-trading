import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

class HRPOptimizer:
    """
    Optimizador de Portafolio basado en Hierarchical Risk Parity (HRP)
    Arquitectura Matemática original de Marcos López de Prado.
    """
    def __init__(self):
        pass

    def get_inverse_variance(self, cov):
        """Calcula el portafolio de Varianza Inversa para un bloque de covarianza."""
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def get_cluster_var(self, cov, c_items):
        """Calcula la varianza total de un sub-cluster jerárquico."""
        cov_ = cov.loc[c_items, c_items] # Extraer submatriz
        w_ = self.get_inverse_variance(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    def get_quasi_diag(self, link):
        """Reordena la matriz original para que los elementos correlacionados queden contiguos."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3] # Número total de activos originales
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2) # Expandir para hacer espacio
            df0 = sort_ix[sort_ix >= num_items] # Encontrar los clusters virtuales
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0] # Reemplazar con el hijo 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]) # Reemplazar con el hijo 2
            sort_ix = sort_ix.sort_index() # Re-ordenar
            sort_ix.index = range(sort_ix.shape[0]) # Limpiar índices
            
        return sort_ix.tolist()

    def get_rec_bipart(self, cov, sort_ix):
        """Asignación de Pesos (Top-Down) mediante Bisección Recursiva de la varianza."""
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix] # Iniciar con todos los activos en 1 solo mega-cluster
        
        while len(c_items) > 0:
            # Dividir todos los clusters actuales a la mitad (Bisección)
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
            for i in range(0, len(c_items), 2): # Analizar de a pares (Cluster Izquierdo vs Derecho)
                c_items0 = c_items[i]   # Cluster 1
                c_items1 = c_items[i + 1] # Cluster 2
                
                # Calcular la varianza de cada rama
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                
                # Asignar peso proporcional inversamente a la varianza de la rama
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
                
        return w

    def correl_dist(self, corr):
        """
        Calcula una matriz de distancia válida matemáticamente (Métrica de Distancia).
        Donde correlación perfecta (1) = distancia 0.
        Donde correlación inversa (-1) = distancia 1.
        """
        dist = ((1 - corr) / 2.) ** 0.5 
        return dist

    def allocate(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Ejecuta el Algoritmo Completo de HRP.
        :param returns_df: DataFrame donde cada columna es el retorno histórico diario de una estrategia/activo.
        :return: Serie de pandas con los % de capital óptimo para cada activo.
        """
        # 1. Calcular Covarianzas y Correlaciones de las estrategias
        cov = returns_df.cov()
        corr = returns_df.corr()
        
        # 2. Transformar Correlación en Matriz de Distancia
        dist = self.correl_dist(corr)
        
        # 3. Clustering Jerárquico Computacional (Scipy)
        # Convertimos la matriz cuadrada a su versión condensada (solo el triángulo superior)
        dist_array = ssd.squareform(dist.values)
        link = sch.linkage(dist_array, method='single') # Generar el Dendrograma
        
        # 4. Quasi-Diagonalización de los clusters
        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist() # Mapear índices a nombres de las columnas reales
        
        # 5. Calcular los pesos óptimos mediante bisección
        df_cov = cov.loc[sort_ix, sort_ix]
        weights = self.get_rec_bipart(df_cov, sort_ix)
        
        # Retornar los pesos ordenados alfabéticamente
        return weights.sort_index()

# Bloque de testing aislado (Para validar que el algoritmo no colapse)
if __name__ == "__main__":
    print("Probando Motor Matemático HRP...")
    # Generamos datos aleatorios simulando el historial de retornos de 4 Estrategias Campeonas
    np.random.seed(42)
    datos_sinteticos = np.random.randn(1000, 4) 
    
    # Inyectar una alta correlación sintética entre el Activo A y el B para ver si el HRP funciona
    datos_sinteticos[:, 1] = datos_sinteticos[:, 0] + np.random.randn(1000) * 0.1 
    
    df_returns = pd.DataFrame(datos_sinteticos, columns=['Estrategia_SP500', 'Estrategia_ORO', 'Estrategia_EURUSD', 'Estrategia_ECH'])
    
    hrp = HRPOptimizer()
    weights = hrp.allocate(df_returns)
    
    print("\n✅ HRP Terminado. Pesos Óptimos Asignados:")
    for asset, weight in weights.items():
        print(f"  > {asset}: {weight*100:.2f}% del capital")
    
    print(f"\nSuma total del capital asignado: {weights.sum()*100:.2f}%")
