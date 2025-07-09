# 🧠⚗️ La Neurona Química

**Demostración de Neuronas Artificiales Aplicadas a Química Física**

Este proyecto implementa neuronas artificiales desde cero para modelar fenómenos químicos, comparando implementaciones manuales con bibliotecas profesionales como TensorFlow.

---

## 🚀 Características

- **Neurona Única**: Modelado de gases ideales (Ley de Charles/Gay-Lussac)
- **Red Neuronal**: Solubilidad del CO₂ en cerveza (relación no lineal)
- **TensorFlow**: Implementación profesional y comparativa
- **Visualización 3D**: Gráficos interactivos con Plotly
- **Educativo**: Código comentado y explicaciones detalladas

---

## 📋 Requisitos

- **Python**: 3.12.10 o superior
- **Sistema**: Windows (PowerShell)
- **Herramientas**: VS Code, Jupyter Notebook
- **Librerías**: Ver `requirements.txt`

---

## 🛠️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/neurona-quimica.git
cd neurona-quimica
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Jupyter

```bash
python -m ipykernel install --user --name=.venv
```

---

## 🧪 Contenido del Notebook

### **Parte 1: Neurona Única (Lineal)**

```python
# Modelado de gases ideales
P = k * T  # Presión = constante × Temperatura
```

- Implementación de descenso de gradiente
- Optimización de parámetros (w, b)
- Comparación con valores teóricos

### **Parte 2: Red Neuronal (No Lineal)**

```python
# Solubilidad del CO₂ en cerveza
S = k * P * exp(-E_a/(R*T))
```

- MLP con capas ocultas
- Función de activación ReLU
- Visualización 3D interactiva

### **Parte 3: TensorFlow**

- Implementación profesional
- Comparación de rendimiento
- Superficies 3D avanzadas

---

## 🎯 Casos de Uso

### **Educativo**

- Entender el funcionamiento interno de neuronas
- Aprender descenso de gradiente desde cero
- Visualizar conceptos de ML

### **Científico**

- Modelar fenómenos químicos
- Predicción de propiedades físicas
- Optimización de procesos

### **Práctico**

- Predicción de solubilidad en bebidas
- Control de calidad en industria cervecera
- Simulación de condiciones de almacenamiento

---

## 📊 Resultados Esperados

### **Neurona Única**

- MSE final: ~0.0001
- Recuperación exacta de constantes físicas
- Convergencia en ~100 épocas

### **Red Neuronal**

- Modelado preciso de no-linealidades
- Predicciones físicamente consistentes
- Visualización clara de relaciones P-T-S

---

## 🔧 Uso

### **Ejecutar el Notebook**

1. Abrir VS Code en el directorio del proyecto
2. Seleccionar kernel `.venv (Python 3.12.10)`
3. Ejecutar `La Neurona Química.ipynb` celda por celda

### **Verificar Instalación**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("✅ Entorno configurado correctamente")
```

---

## 📁 Estructura del Proyecto

```
DL Deep Learning/
├── .venv/                          # Entorno virtual (ignorado)
├── .gitignore                      # Configuración Git
├── requirements.txt                # Dependencias
├── CONFIGURACION_ENTORNO.md       # Documentación técnica
├── README.md                      # Este archivo
└── La Neurona Química.ipynb       # Notebook principal
```

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'feat: agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

---

## 📚 Recursos Adicionales

### **Conceptos Químicos**

- [Ley de Charles](https://en.wikipedia.org/wiki/Charles%27s_law)
- [Ley de Henry](https://en.wikipedia.org/wiki/Henry%27s_law)
- [Solubilidad de gases](https://en.wikipedia.org/wiki/Gas_solubility)

### **Machine Learning**

- [Descenso de gradiente](https://en.wikipedia.org/wiki/Gradient_descent)
- [Perceptrón multicapa](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- [Función ReLU](<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>)

### **Herramientas**

- [TensorFlow](https://www.tensorflow.org/)
- [Plotly](https://plotly.com/python/)
- [Jupyter](https://jupyter.org/)

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Desarrollado con ❤️ para la educación en Machine Learning aplicado a Química**

- 📧 Email: tu.email@ejemplo.com
- 🐱 GitHub: [@tu-usuario](https://github.com/tu-usuario)
- 💼 LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

---

## 🙏 Agradecimientos

- Comunidad de Python científico
- Documentación de TensorFlow
- Ejemplos de Plotly para visualización científica

---

**¡Explora las neuronas químicas! 🧠⚗️**
