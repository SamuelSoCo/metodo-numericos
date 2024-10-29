import numpy as np
import tkinter as tk
from tkinter import messagebox

def eliminacion_gaussiana(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            factorpivote = A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] -= factorpivote * A[i][k]
            b[j] -= factorpivote * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return x

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def Jacobi(A, b, tolerancia=1e-6, max_iteraciones=40):
    x = np.zeros_like(b, dtype=np.double)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for a in range(max_iteraciones):
        x_nuevo = (b - np.dot(R, x)) / D
        if euclidean_distance(x, x_nuevo) < tolerancia:
            return x_nuevo, a
        x = x_nuevo
    return x_nuevo, max_iteraciones    

def Gauss_Seidel(A, b, tolerancia=1e-10, max_iteraciones=40):
    x = np.zeros_like(b, dtype=np.double)
    for a in range(max_iteraciones):
        x_nuevo = np.copy(x)
        n = A.shape[0]
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma += A[i][j] * x_nuevo[j]
            x_nuevo[i] = (b[i] - suma) / A[i][i]  
        if euclidean_distance(x, x_nuevo) < tolerancia:
            return x_nuevo, a
        x = x_nuevo
    return x_nuevo, max_iteraciones

# Pruebas de datos
A = np.array([[10, 1, 2], [4, 6, -1], [-2, 3, 8]]).astype(float)
b = np.array([3, 9, 51])

# Interfaz gráfica usando tkinter
def mostrar_resultado(metodo):
    if metodo == "Eliminación Gaussiana":
        x = eliminacion_gaussiana(A.copy(), b.copy())
        resultado = f"Raíces por Eliminación Gaussiana: {x}"
    elif metodo == "Jacobi":
        x, iteraciones = Jacobi(A, b)
        resultado = f"Raíces por Jacobi: {x} (Iteraciones: {iteraciones})"
    elif metodo == "Gauss-Seidel":
        x, iteraciones = Gauss_Seidel(A, b)
        resultado = f"Raíces por Gauss-Seidel: {x} (Iteraciones: {iteraciones})"
    else:
        resultado = "Método no reconocido"
    messagebox.showinfo("Resultado", resultado)

def main():
    root = tk.Tk()
    root.title("Selecciona un método de solución")
    root.geometry("300x200")

    label = tk.Label(root, text="Seleccione un método:")
    label.pack(pady=10)

    boton_gaussiana = tk.Button(root, text="Eliminación Gaussiana", command=lambda: mostrar_resultado("Eliminación Gaussiana"))
    boton_gaussiana.pack(pady=5)

    boton_jacobi = tk.Button(root, text="Jacobi", command=lambda: mostrar_resultado("Jacobi"))
    boton_jacobi.pack(pady=5)

    boton_gauss_seidel = tk.Button(root, text="Gauss-Seidel", command=lambda: mostrar_resultado("Gauss-Seidel"))
    boton_gauss_seidel.pack(pady=5)

    root.mainloop()

# Ejecutar la interfaz
if __name__ == "__main__":
    main()
