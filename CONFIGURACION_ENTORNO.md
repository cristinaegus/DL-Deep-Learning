# Configuración del Entorno - La Neurona Química

## 📋 Resumen de Configuración

Este documento describe la configuración completa del entorno virtual para el notebook **"La Neurona Química.ipynb"** que demuestra neuronas artificiales aplicadas a química física.

---

## 🛠️ Configuración Realizada

### 1. **Entorno Virtual**

- **Tipo**: Virtual Environment (.venv)
- **Ubicación**: `c:\Users\Dell\PyhtonIA\DL Deep Learning\.venv\`
- **Python**: Versión 3.12.10
- **Comando de ejecución**: `"C:/Users/Dell/PyhtonIA/DL Deep Learning/.venv/Scripts/python.exe"`

### 2. **Kernel de Jupyter**

- **Kernel configurado**: `.venv (Python 3.12.10)`
- **Estado**: Activo y funcional
- **Soporte**: Jupyter Notebook completo

---

## 📦 Librerías Instaladas

### **Librerías Core**

- **numpy**: Operaciones matemáticas y arrays multidimensionales
- **pandas**: Manipulación y análisis de datos (DataFrames)
- **scikit-learn**: Machine learning (train_test_split, StandardScaler)

### **Visualización**

- **matplotlib**: Gráficos 2D y 3D estáticos
- **plotly**: Visualizaciones interactivas 3D

### **Deep Learning**

- **tensorflow**: Redes neuronales y deep learning

### **Dependencias del Sistema**

- **ipykernel**: Kernel de Jupyter para Python
- **jupyter_client**: Cliente de Jupyter
- **jupyter_core**: Funcionalidades core de Jupyter

---

## 🧪 Aplicaciones del Notebook

### **Parte 1: Neurona Única (Lineal)**

- **Modelo**: Ley de Charles/Gay-Lussac para gases ideales
- **Ecuación**: `P = k * T`
- **Implementación**: Neurona simple con descenso de gradiente

### **Parte 2: Red Neuronal (No Lineal)**

- **Modelo**: Solubilidad del CO₂ en cerveza
- **Ecuación**: `S = k * P * exp(-E_a/(R*T))`
- **Implementación**: MLP con capas ocultas y activación ReLU

### **Parte 3: TensorFlow**

- **Implementación profesional** de ambos modelos
- **Comparación** entre implementación manual y TensorFlow
- **Visualizaciones avanzadas** con superficies 3D

---

## 🚀 Instrucciones de Uso

### **Activación del Entorno**

```bash
# En PowerShell/CMD
cd "c:\Users\Dell\PyhtonIA\DL Deep Learning"
.\.venv\Scripts\activate
```

### **Ejecución del Notebook**

1. Abrir VS Code en el directorio del proyecto
2. Seleccionar el kernel `.venv (Python 3.12.10)`
3. Ejecutar las celdas secuencialmente

### **Verificación de Instalación**

```python
# Ejecutar en una celda para verificar las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("✅ Todas las librerías instaladas correctamente")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
```

---

## 🔧 Resolución de Problemas

### **Si el kernel no aparece:**

1. Reinstalar ipykernel: `pip install ipykernel`
2. Registrar el kernel: `python -m ipykernel install --user --name=.venv`

### **Si faltan librerías:**

```bash
# Activar entorno y reinstalar
.\.venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib plotly tensorflow
```

### **Si hay errores de importación:**

- Verificar que el kernel correcto esté seleccionado
- Reiniciar el kernel desde VS Code
- Ejecutar las celdas de importación nuevamente

---

## 📊 Estructura del Proyecto

```
DL Deep Learning/
├── .venv/                          # Entorno virtual
│   ├── Scripts/
│   │   └── python.exe             # Ejecutable de Python
│   └── Lib/                       # Librerías instaladas
├── La Neurona Química.ipynb       # Notebook principal
├── CONFIGURACION_ENTORNO.md       # Este archivo
└── solubilidad_co2_cerveza_3d.jpg # Gráfico generado
```

---

## 🎯 Objetivos del Proyecto

### **Educativos**

- Entender el funcionamiento de neuronas artificiales
- Aplicar ML a problemas de química física
- Comparar implementaciones manuales vs bibliotecas

### **Técnicos**

- Descenso de gradiente desde cero
- Redes neuronales multicapa
- Visualización de datos científicos

### **Aplicaciones Prácticas**

- Modelado de gases ideales
- Predicción de solubilidad en bebidas
- Optimización de procesos químicos

---

## 📅 Fecha de Configuración

**Configurado el**: 9 de julio de 2025

## 👨‍💻 Herramientas Utilizadas

- **VS Code**: Editor principal
- **Python 3.12.10**: Lenguaje base
- **Jupyter Notebook**: Entorno interactivo
- **Git**: Control de versiones

---

## 📝 Notas Adicionales

1. **Rendimiento**: El entorno está optimizado para cálculos científicos
2. **Compatibilidad**: Funciona en Windows con PowerShell
3. **Extensibilidad**: Fácil agregar nuevas librerías con `pip install`
4. **Portabilidad**: El entorno es autocontenido y transferible

---

## 🔄 Control de Versiones con Git

### **Archivos del Repositorio**

```
DL Deep Learning/
├── .venv/                          # IGNORADO - Entorno virtual
├── .gitignore                      # Configuración de Git
├── requirements.txt                # Dependencias del proyecto
├── CONFIGURACION_ENTORNO.md       # Este archivo
├── La Neurona Química.ipynb       # Notebook principal
└── README.md                      # Documentación (opcional)
```

### **Configuración Inicial de Git**

```bash
# Inicializar repositorio
git init

# Configurar usuario (si es primera vez)
git config --global user.name "Tu Nombre"
git config --global user.email "tu.email@ejemplo.com"

# Agregar archivos
git add .
git commit -m "Initial commit: Configuración completa del entorno neuronal"
```

### **Comandos Útiles**

```bash
# Ver estado del repositorio
git status

# Agregar cambios específicos
git add "La Neurona Química.ipynb"
git add CONFIGURACION_ENTORNO.md

# Commit con mensaje descriptivo
git commit -m "feat: Implementación de red neuronal para solubilidad CO2"

# Ver historial
git log --oneline

# Crear rama para experimentos
git checkout -b experimentos-neurona
```

### **Subir a GitHub**

```bash
# Conectar con repositorio remoto
git remote add origin https://github.com/tu-usuario/neurona-quimica.git

# Subir código
git push -u origin main

# Para subidas futuras
git push
```

### **Archivos Ignorados**

- **Entorno virtual** (.venv/): Muy grande, se recrea con requirements.txt
- **Outputs generados** (_.jpg, _.png): Se regeneran al ejecutar el notebook
- **Archivos temporales** (**pycache**/, .ipynb_checkpoints/)
- **Logs y cache** (\*.log, .cache/)
- **Configuraciones locales** (config.ini, .env)

### **Reproducir el Entorno**

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/neurona-quimica.git
cd neurona-quimica

# Crear entorno virtual
python -m venv .venv

# Activar entorno
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar kernel de Jupyter
python -m ipykernel install --user --name=.venv
```

---

## 🆘 Soporte

Si encuentras problemas:

1. Verifica que el kernel correcto esté seleccionado
2. Reinicia VS Code y el kernel
3. Revisa que todas las librerías estén instaladas
4. Consulta este documento para comandos específicos

**¡Tu entorno está listo para explorar las neuronas químicas!** 🧪⚗️🤖
