# primero se leen los 0s y 1s, se pasan a una lista y se convierten a 1s y -1s
def leer_patron(ruta_archivo):
    vector = []
    with open(ruta_archivo, 'r') as f:
        for linea in f:
            # separar por espacios y limpiar saltos
            elementos = linea.strip().split()
            for valor in elementos:
                if valor == '1':
                    vector.append(1)
                elif valor == '0':
                    vector.append(-1)
    return vector

# luego se crea una matriz donde # representa 1 y . representa -1
def imprimir_matriz(vector, filas=7, columnas=7):
    for i in range(filas):
        fila_str = ""
        for j in range(columnas):
            idx = i * columnas + j
            simbolo = "# " if vector[idx] == 1 else ". "
            fila_str += simbolo
        print(fila_str)
    print()

# ahora con la regla de Hebb se entrena la red creando la matriz de pesos W
def entrenar_hopfield(patrones_entrenamiento, n_neuronas=49):
    W = []
    for i in range(n_neuronas):
        fila = []
        for j in range(n_neuronas):
            fila.append(0)
        W.append(fila)
    
    # acumular el producto externo de cada patron de entrenamiento
    for patron in patrones_entrenamiento:
        for i in range(n_neuronas):
            for j in range(n_neuronas):
                if i != j:  # dejar la diagonal en cero
                    W[i][j] += patron[i] * patron[j]
    return W

# con la actualizacion asincrona se recupera el patron asociado a un patron inicial
def recuperar_patron(patron_inicial, W, max_iteraciones=100):
    n_neuronas = len(patron_inicial)
    
    # copia de la entrada
    estado = []
    for val in patron_inicial:
        estado.append(val)
        
    convergencia = False
    iteracion = 0
    
    # iterar hasta converger o alcanzar el limite
    while not convergencia and iteracion < max_iteraciones:
        cambio_en_iteracion = False
        
        # recorrer cada neurona
        for i in range(n_neuronas):
            suma_ponderada = 0
            
            # la entrada total a la neurona i
            for j in range(n_neuronas):
                suma_ponderada += W[i][j] * estado[j]
            
            # funcion de activacion
            if suma_ponderada >= 0:
                nuevo_estado = 1
            else:
                nuevo_estado = -1
            
            # ver si la neurona cambio de estado
            if nuevo_estado != estado[i]:
                estado[i] = nuevo_estado
                cambio_en_iteracion = True
                
        # si no hubo cambios ya hay convergencia
        if not cambio_en_iteracion:
            convergencia = True
            
        iteracion += 1

    print(f"-> Convergencia alcanzada en {iteracion} iteracion(es).")
    return estado

if __name__ == "__main__":
    archivos_base = ["corazon.txt", "diamante.txt", "pica.txt", "trebol.txt"]
    datos_entrenamiento = [leer_patron(f"dataset/{nom}") for nom in archivos_base]
    
    patron_prueba = leer_patron("dataset/x.txt")
    
    W = entrenar_hopfield(datos_entrenamiento, n_neuronas=49)
    
    print("--- PATRON DE ENTRADA CON RUIDO (x.txt) ---")
    imprimir_matriz(patron_prueba)
    
    # recuperar el patron
    patron_recuperado = recuperar_patron(patron_prueba, W)
    
    print("--- PATRON RECORDADO / RECUPERADO ---")
    imprimir_matriz(patron_recuperado)