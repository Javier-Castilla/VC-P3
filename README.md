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

En esta tarea se realizará un sistema que sea capaz de contar en dinero que hay en una imagen únicamente detectando las monedas que aparecen en ella para posteriormente, clasificar las detecciones asignándoles su valor monetario.

Para ello, nuestra técnica ha sido umbralizar la imagen para intentar separar las monedas del fondo para luego encontrar contornos exteriores y encerrarlos en una elipse. El motivo por el que hemos decidio usar una elipse es porque esta nos permite hacer un conteo correcto incluso cuando hay sombras presentes en la imagen.

Con la imagen inicial propuesta, el resultado de la umbralización es el siguiente:

```python
image = cv2.imread("imgs/coins_v2.jpg", cv2.IMREAD_COLOR_RGB)

image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 2)

canny = cv2.Canny(image_gray, 50, 125)

_, threshold_image = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

<img src='imgs/preproceso_monedas.jpg'>

Como se puede observar en el resultado de aplicar Canny, los bordes de las monedas no se han cerrado correctamente, razón de más por la que hemos seguido la técnica de identificar contornos circulares.

Una vez se ha procesado la imagen para segmentar las monedas, se buscan sus contornos, filtrando aquellos que no cumplan un área mínima y que no son suficientemente circulares.

> [NOTE!]
> Para medir la circularidad, se ha usado la siguiente fórmula: 4 * np.pi * (area / (perimeter * perimeter)). En cuanto más se acerque a 1 el resultado, más circular será el contorno.

El método desarrollado para esta fin es el siguiente:

```python
def __find_circular_contours(self, image):
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        [c for c in contours if self.__calculate_circularity(c)],
        key=lambda c: cv2.contourArea(c), reverse=True
    )
    contours_ellipses = {}

    for i, c in enumerate(contours):
        ellipse = cv2.fitEllipse(c)
        contours_ellipses[i] = {
            'ellipse': ellipse,
            'center': ellipse[0],
            'width': min(ellipse[1]),
            'angle': ellipse[2]
        }
        
    return contours_ellipses
```

Y el método correspondiente al cálculo de la circularidad y área mínima es el siguiente:

```python
def __calculate_circularity(self, contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity >= 0.6 and area > 50
```

> [NOTE!]
> El motivo de elegir una elipse y no un círculo para encerrar el contorno, es porque el eje menor de la elipse coincidirá con el diámetro de la moneda sin tener en cuenta la sombra. De este modo, se evita detectar formas más grandes de las reales.

En este punto ya se ha procesado la imagen para segmentarla y se han encontrado contornos circulares que podrían asociarse a una moneda. Ahora solo falta poner una moneda de referencia para obtener el ratio de píxel / milímetro con el fin de clasificar correctamente los contornos.

El modo de proceder ha sido sencillo, simplemente mostramos la imagen con los contornos dibujados encima y un índice. Posteriormente, se pide un `input` del índice de la moneda de 1€. Con esto, ya se puede calcular perfectamente el ratio.

```python
def show_detection_and_pick_coin(self, coin):
    self.selected_coin = coin
    drawed = self.__draw_found__ellipses_information()
    plt.imshow(drawed, cmap='gray')
    plt.axis('off')
    plt.show()
    self.selected_index = input(f"Selecciona el índice de la moneda con valor {coin}")
    return drawed, self.selected_index
```

Tras todo esto, se ha segmentado la imagen, se han detectado contornos circulares, se ha calculado el ratio a partir de la selección del usuario como moneda de 1€. Ahora... `let's do the math!` ( hacer el conteo ). Para ello, se ha declarado en un diccionario los diámetros de cada moneda con su respectivo valor monetario en céntimos para evitar errores por `float`. Seguidamente, para cada contorno, se coge el diámetro del diccionario nombrado que tenga menor error con el eje menor de dicho contorno y se añade su valor al contador.

```python
def do_the_math(self):
    if len(self.ellipses) == 0:
        print('No se han encontrado figuras suficientes para hacer las mates.')
        return

    self.PIXEL_MM_RATIO = self.ellipses[int(self.selected_index)]['width'] / MoneyCounter.coins_d['100']

    current = []

    for c in self.ellipses.values():
        r = c['width']
        d = r / self.PIXEL_MM_RATIO
        k = min(MoneyCounter.coins_invert, key=lambda k: abs(k - d))
        current.append(Coin(MoneyCounter.coins_invert[k], k))

    return sum(current), current
```

> [!NOTE]
> El método que realiza el conteo devuelve la suma total del dinero y las monedas seleccionadas para cada contorno detectado.

Cabe destacar que se ha realizado el siguiente modelo para una representación más clara de los resultados:

```python
from dataclasses import dataclass

@dataclass
class Coin:
    value: int
    diameter: float
    
    def __add__(self, other):
        if isinstance(other, Coin):
            return Money(self.value + other.value)
        elif isinstance(other, Money):
            return Money(self.value + other.value)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return Money(self.value)
        elif isinstance(other, Money):
            return Money(self.value + other.value)
        return NotImplemented
            

@dataclass
class Money:
    value: int

    def amount(self):
        return self.value / 100
```

Todo lo anteriormente nombrado han sido métodos desarrollados dentro de una clase cuyo motivo de existencia es la realización de pruebas mediante el mismo método de conteo en diferentes imágenes. A continuación, se muestran los resultados obtenidos para cada una de las imágenes seleccionadas, además de la propuesta inicialmente.

> [!NOTE]
> La clase desarrollada puede verse en el cuaderno de Python [Practica3.ipynb](Practica3.ipynb#clase-desarrollada-para-el-conteo).

<table align="center">
    <tr align="center">
        <td>
            <h3 align="center">Monedas.jpg</h3>
            <img src='imgs/conteo_0.jpg'>
            <div>
                <h4>Resultado obtenido: 3.88€</h4>
            </div>
        </td>
        <td>
            <h3 align="center">coins_with_shadow.jpg</h3>
            <img src='imgs/conteo_1.jpg'>
            <div>
                <h4>Resultado obtenido: 3.88€</h4>
            </div>
        </td>
    </tr>
    <tr align="center">
        <td>
            <h3 align="center">coins_v2.jpg</h3>
            <img src='imgs/conteo_2.jpg'>
            <div>
                <h4>Resultado obtenido: 8€</h4>
            </div>
        </td>
        <td>
            <h3 align="center">coins_v3.jpg</h3>
            <img src='imgs/conteo_3.jpg'>
            <div>
                <h4>Resultado obtenido: 3.88€</h4>
            </div>
        </td>
    </tr>
</table>

> [!IMPORTANT]
> Hay que tener el cuenta que las imágenes usadas han tenido bastante buena calidad. Este sistema empieza a fallar cuando el umbralizado no es suficiente para separar bien las monedas del fondo por ejemplo, cuando el color del fondo es muy parecido al de las monedas o estas tienen brillos irregulares o incluso deformaciones por la lente de la cámara.

<h2 align="center">Tarea 2</h2>