<h1 align="center">Práctica 3</h1>

<h2 align="center">Asignatura: Visión por Computador</h2>

Universidad de Las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Curso 2025/2026 

<h2 align="center">Autores</h2>

- Asmae Ez Zaim Driouch
- Javier Castilla Moreno

<h2 align="center">Bibliotecas utilizadas</h2>

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%43FF6400?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000?style=for-the-badge&logo=pillow)](https://pypi.org/project/pillow/)
[![Tkinter](https://img.shields.io/badge/Tkinter-%2334A853?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/tkinter.html)


## Cómo usar
### Primer paso: clonar este repositorio
```bash
git clone "https://github.com/Javier-Castilla/VC-P3"
```
### Segundo paso: Activar tu envinroment e instalar dependencias
> [!NOTE]
> Todas las dependencias pueden verse en [este archivo](envinronment.yml). Si se desea, puede crearse un entorno de Conda con dicho archivo.

Si se opta por crear un nuevo `Conda envinronment` a partir del archivo expuesto, es necesario abrir el `Anaconda Prompt` y ejecutar lo siguiente:

```bash
conda env create -f environment.yml
```

Posteriormente, se activa el entorno:

```bash
conda activate VC_P3
```

### Tercer paso: ejecutar el cuaderno
Finalmente, abriendo nuestro IDE favorito y teniendo instalado todo lo necesario para poder ejecutar notebooks, se puede ejecutar el cuaderno de la práctica [Practica3.ipynb](Practica3.ipynb) seleccionando el envinronment anteriormente creado.

> [!IMPORTANT]
> Todos los bloques de código deben ejecutarse en órden, de lo contrario, podría ocasionar problemas durante la ejecución del cuaderno.

<h1 align="center">Tareas</h1>

<h2 align="center">Tarea 1</h2>

<h2 align="center">Tarea 2</h2>

<h2 align="center">Tarea 2</h2>

Para la realización de esta tarea se extraerán características geométricas y visuales de los diferentes tipos de microplásticos para posteriormente. Posteriormetne, se usarán esas características extraídas tratando de clasificar correctamente las 3 clases de microplásticos diferentes sobre la imagen [MPs_test.png](imgs/MPs_test.png).

El sistema es capaz de identificar y clasificar tres tipos diferentes de microplásticos:

- **Pellets (PEL)**: Partículas esféricas o redondeadas
- **Fragmentos (FRA)**: Piezas irregulares y angulosas
- **Alquitrán (TAR)**: Partículas oscuras con forma irregular

El clasificador alcanza un **accuracy del 85.71%** en el conjunto de test por lo que podemos afirmar que se ha conseguido segmentar muy bien las partículas de las 3 imágenes usadas para extraer características. 
Ahora, era necesario filtrar algunos contornos que podrían ser pequeñas manchas en la imagen. Para lograr esto, se usó un método estadístico estudiado en cursos anteriores, la `eliminación de outliers`. Además, se propuso un área mínima de contorno, descartando todos aquellos que no la superasen.

Para lograr este objetivo, se han realizado los siguientes procedimientos:
1. Uso de un conjunto de 3 imágenes, una para cada clase.
2. Segmentación de cada imagen para la extracción de contornos.
3. Tratamiento de los contornos extraídos, obteniendo características de cada uno de ellos.
4. Estandarizado de los valores de las características.
5. Introducción de características en el clasificador RandomForest.
6. Repetición de los puntos del 2 al 4 para la imagen de test.
7. Clasficación de los contornos detectados en la imagen de test.
8. Evaluación de resultados.

<h3 align="center">Segmentación y extracción de contornos</h3>

El proceso de segmentación se ha ido modificando a lo largo de la realización de esta tarea. Esto ha sido impulsado por un descontento inicial con los resultados obtenidos las primeras veces, pues notamos que realmente se debía a una segmentación algo pobre de las imágenes iniciales sobre las que se extraerían las características.

En los primeros pasos, se usaba una segmentación simple mediante un umbralizado recurriendo a la función `cv2.threshold` con OTSU. En la mayoría de contornos funcionaba bien, pero cuando aparecían microplásticos con un color muy parecido al fondo, esta técnica de segmentación fallaba.

<img src="">

Posteriormente, decidimos usar el `umbralizado adaptativo Gaussiano`. Parecía dar mejores resultados, pero el desenfoque en las imágenes iniciales provocaba la presencia de demasiado ruido en la detección de contornos, por lo que decidimos aplicar la función `cv2.medianBlur` con buenos resultados.

<img src="">

En este punto los resultados de la clasificación mejoraron bastante. Se incrementó la precisión `de un 52% a un 67%`, pero creímos que no era suficiente. Por ello, decidimos hacer una combinación de las dos técnicas de segmentación que habíamos planteado junto con una `dilatación de bordes`. Este enlace permitía rellenar en la umbralización Gaussiana aquellos bordes que sí pudieron ser detectados con el umbralzado, es decir, ambos umbralizados se complementaban, y es ahí donde el filtrado de mediana nos sirvió de gran ayuda, pues el umbralizado adaptativo como bien se explicó anteriormente producía mucho ruido, pero el filtro de mediana consiguió eliminar prácticamente la totalidad de este.

<img src="">

En el caso de la imagen de test tiene ajustes específicos distintos a los usados para las imágenes de entrenamiento debido a las sombras presentes en la misma.

No se usa Otsu por las características de iluminación. 

Se emplean características morfológicas para dilatar:

```python
kernel = np.ones((3, 3), np.uint8)
adap_th = cv2.dilate(adap_th, kernel, iterations=1)
```
Expande los píxeles blancos para cerrar pequeños huecos dentro de partículas y conectar regiones fragmentadas de una misma partícula tras el el umbralizado adaptativo.

```python
lower_bound = max(0, np.percentile(areas, 75))
cv2.contourArea(x) > 90  # vs 325 en training
```

  - Test es más **inclusivo** (detecta partículas más pequeñas)
  - Training fue más **restrictivo** pues solo se deseaba muestras de alta calidad.

<img src="">


<h3 align="center">Filtrado de contornos</h3>

Se elimina ruido de baja frecuencia y pequeños artefactos filtrando las áreas mínimas:

```python
cv2.contourArea(x) > 325  # Para entrenamiento
cv2.contourArea(x) > 90   # Para test (más permisivo)
```

La diferencia entre entrenamiento (325) y test (90) se debe a que:
- En entrenamiento queremos **muestras de alta calidad** sin ambigüedades
- En test queremos ser **más inclusivos** para no perder detecciones válidas

> [!NOTE]
> La eliminación de outliers mediante percentiles consiste en analizar la distribución de áreas de todos los contornos detectados y descartar aquellos que estén fuera de un rango en este caso (75 - 100).

**Implementación en el código**:
```python
areas = np.array([cv2.contourArea(contour) for contour in current_contours])
lower_bound = max(0, np.percentile(areas, 75))
upper_bound = np.percentile(areas, 100)
current_contours = [x for x in current_contours 
                    if lower_bound <= cv2.contourArea(x) <= upper_bound]
```

Esto se realiza tras identificar que sin la eliminación de los outliers se obtenían falsos positivos pues el modelo estaba entranando con contornos muy pequeños que correspondían al ruido o partículas inrrelevantes.



<h3 align="center">extracción de características
</h3>

El sistema extrae **14 características** por cada contorno detectado para capturar propiedades discriminativas entre los tres tipos de microplásticos.

#### 1. **Área (Area)**
```python
area = cv2.contourArea(contour)
```
Mide el número de píxeles dentro del contorno.
Se emplea debido a que los pellets tienden a tener áreas más uniformes y regulares, mientras que los fragmentos varían más.

#### 2. **Perímetro (Perimeter)**
```python
perimeter = cv2.arcLength(contour, True)
```
Mide la longitud del borde del contorno.
Relevante pues los fragmentos con bordes irregulares tienen perímetros mayores relativos a su área.

#### 3. **Compacidad (Compacity)**
```python
compacity = (perimeter**2) / area
```
Relación entre perímetro al cuadrado y área

#### 4. **Circularidad (Circularity)**
```python
circularity = (4*np.pi*area) / (perimeter**2)
```
Inverso de la compacidad, normalizado (0-1)

Ambos, tanto la compacidad como la circularidad, son claves para separar pellets de fragmento y alquitrán pues los pellets son muy circulares por lo que se diferenciarían al tener una compacidad baja y un valor cercano al valor uno en circularidad. Fragmentos y alquitrán son muy irregulares por lo que obtendrían una alta capacidad.

#### 6. **Aspect Ratio**
```python
aspect_ratio = w / h
```
Mide relación ancho/alto del bounding box. Ayuda a detectar elongación

#### 7. **E_ratio (Ellipse Ratio)**
```python
if len(contour) >= 5:
    (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
    e_ratio = major_axis / minor_axis
```
Mide la relación entre eje mayor y menor de la elipse ajustada

#### 8. **D_ratio (Distance Ratio)**
```python
M = cv2.moments(contour)
xc = M["m10"]/M["m00"]
yc = M["m01"]/M["m00"]
dist = np.sqrt((contour[:,0,0]-xc)**2 + (contour[:,0,1]-yc)**2)
d_ratio = dist.min() / dist.max()
```
Mide la relación entre distancia mínima y máxima desde el centroide. 
Excelente discriminador entre  entre pellets regulares y fragmentos o alquitrán con protuberancias

#### 9. **Intensidad Media (Intensity)**
```python
gray_pixels = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)[mask == 255]
mean_intensity = np.mean(gray_pixels)
```
Mide el brillo promedio dentro del contorno.

Este descriminador es necesario pues se observa que el alquitrán y los fragmentos geométricamente son similares pues ambos son irregulares con la única clara diferencia siendo el color. Las partículas de alquitrán son siempre de color negro.


#### 10. **Desviación Estándar de Intensidad (Std)**
```python
std_intensity = np.std(gray_pixels)
```
Mide la variabilidad de brillo dentro del contorno.

Pese a diferenciar el alqutrán por la intensidad ya que se identifica que son de colores negros y esa característica es clave para su identificación, se observa que los fragmentos presentan diversos colores pudiendo obtener altas intensidad si son de collor azul oscuro por ejemplo y ser confundidos por tanto con el alquitrán. 

La solución encontrada es que los fragmentos pese a poderse encontrar en colores oscuros no presentan un color uniforme debido a su textura y reflejos. 

Por ello la variabilidad ayuda a detectar **heterogeneidad** del material

#### 11. **Solidez (Solidity)**
```python
hull = cv2.convexHull(contour)
hull_area = cv2.contourArea(hull)
solidity = area / hull_area
```
Mide proporción del área del contorno respecto a su envolvente convexa
  - **Valor ~1.0** → forma convexa (pellets)
  - **Valor <0.8** → forma cóncava con hendiduras (fragmentos irregulares)

Para detectar irregularidades.

#### 12-14. **Features Experimentales (test, test2, test3)**
```python
"test": mean_intensity * circularity * std_intensity,
"test2": solidity * circularity,
"test3": solidity * circularity * mean_intensity
```
Mide combinaciones no lineales de features existentes pues hemos llegado a la conclusión que ciertas características tienen más peso que otras.
    - **test** descrimina el alquitrán pues no son circulares y presentan bajas intensidades por su color negro.
    - **test2** combina forma (circularity, solidity) → discrimina pellets
     - **test3** discrimina alquitrán oscuro de los fragmentos oscuros pues estos últimos presentan no uniformidad por su textura y reflejos mientras uwe los alquitranes si son uniformes

El código incluye **manejo exhaustivo de casos edge**:

```python
if len(contour) < 3:
    return {k: 0 for k in [...]}  # Contornos degenerados

if perimeter == 0 or area == 0:
    return {k: 0 for k in [...]}  # Evita divisiones por cero

if M["m00"] != 0:  # Verifica momentos válidos
```
Evitando así crashes.


<h3 align="center">Estandarizado de valores</h3>

Las 14 características extraídas tienen **escalas muy diferentes**:
- **Área**: Puede ser 500-8000 píxeles²
- **Circularity**: Está en rango [0, 1]
- **Intensity**: Rango [0, 255]

Puesto que Random Forest basado en distancias daría **más peso** a features con valores grandes (Área, Perimeter) features con valores pequeños (Circularity) serían **ignoradas**, aunque sean discriminativas.

### Implementación: StandardScaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)  # Entrena en training
X_test_scaled = scaler.transform(X)       # Aplica en test
```

Para cada característica:
```
z = (x - μ) / σ
```
Donde:
- `μ` = media del feature en training
- `σ` = desviación estándar del feature en training
- `x` = valor original
- `z` = valor estandarizado

Resultando en que todas las features tienen:
- **Media = 0**
- **Desviación estándar = 1**

Así todas las features contribuyen por igual. 

**Uso de parámetros de training en test**:
```python
scaler.fit_transform(X_train)  # Calcula μ y σ del training
scaler.transform(X_test)        # Usa los mismos μ y σ en test
```
Con esto se evita filtración de información de test a training.

StandardScaler usa media/desv.est, sensibles a outliers. Por eso el **filtrado previo de outliers** (percentil 75-100) es una alternativa manual al RobustScaler, MinMaxScaler


<h3 align="center">Introducción de características en el clasificador RandomForest</h3>

```python
clf = RandomForestClassifier(
    n_estimators=100,      # Número de árboles
    random_state=42,       # Reproducibilidad
    max_depth=10,          # Profundidad máxima de árboles
    min_samples_split=5,   # Mínimo de muestras para dividir
    min_samples_leaf=2,    # Mínimo de muestras en hoja
    n_jobs=1               # Procesamiento en serie
)
```

**n_estimators=100**: Entrena 100 árboles de decisión independientes. Se elige 100 para una mejor generalización y menos varianza sin excesivo tiempo de entrenamiento.

**max_depth=10**: Limita la profundidad de cada árbol a 10 niveles y así los árboles no capturan patrones (underfitting). Dado las 14 features y 28 muestras con 10 niveles, cada árbol puede crear hasta 2^10 = 1024 hojas (en teoría)

**min_samples_split=5**: Un nodo solo se divide si tiene ≥5 muestras dado el dataset pequeño (28 muestras / 3 clases ≈ 9 por clase)

**min_samples_leaf=2**:: Cada hoja debe tener ≥2 muestras

**random_state=42**: Fija la semilla aleatoria.Resultados reproducibles entre ejecuciones


Se elige Random Forest para este problema debido a lo siguiente:
   - 28 muestras es muy poco para deep learning
   - Random Forest funciona bien con pocos datos
   - Permite saber qué features son más discriminativas
   - No requiere que los datos sean linealmente separables
   - Resistente a overfitting: El promedio de 100 árboles reduce varianza
   - Regularización mediante max_depth, min_samples_split


<h3 align="center">Evaluación de resultados</h3>

Se emplea Índice Espacial R-Tree.Se utiliza esta estructura de datos para validación más rápida y eficiente.

```python
from rtree import index

idx = index.Index()
for i, ann in enumerate(annotations):
    x_min, y_min, x_max, y_max = ann['bbox']
    idx.insert(i, (x_min, y_min, x_max, y_max))
```

Organiza bounding boxes jerárquicamente

```python
def buscar_anotacion(cx, cy):
    posibles = list(idx.intersection((cx, cy, cx, cy)))  # O(log M)
    for i in posibles:
        # Solo revisa candidatos espacialmente cercanos
```
- **Complejidad**: O(N × log M)
- Para 98 predicciones → **~650 comparaciones** (15× más rápido)

la función de matching empleada es:
```python
def buscar_anotacion(cx, cy):
    posibles = list(idx.intersection((cx, cy, cx, cy)))
    
    for i in posibles:
        x_min, y_min, x_max, y_max = annotations[i]['bbox']
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            return annotations[i]
    
    return None
```

```python
if real_label is None: continue
```

**Si no hay match**:
- El contorno se **descarta** de la evaluación
- **No afecta al modelo**, solo a las métricas
- **Implicación**: Las métricas reportadas son **optimistas**
  - Solo evalúan predicciones que pudieron ser validadas
  - Ignoramos contornos detectados pero no anotados

<img src="">

#### 1. **Accuracy: 85.71%**
```python
accuracy = accuracy_score(y_true, y_pred)
```

#### 2. **Precision: 85.82%**
```python
precision = precision_score(y_true, y_pred, average="weighted")
```
De todas las veces que el modelo dice "es clase X", **85.82% tiene razón**

#### 3. **Recall: 85.71%**
```python
recall = recall_score(y_true, y_pred, average='weighted')
```
De todos los microplásticos reales de clase X, **85.71% son detectados correctamente**

#### 4. **F1-Score: 85.14%**
```python
f1 = f1_score(y_true, y_pred, average='weighted')
```
**Media armónica** de precision y recall
**85.14%** indica buen balance entre ambas

