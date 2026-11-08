# ElectronicaLab

Aplicación local para Windows: biblioteca musical, reproducción, inspección de audio y clasificación personal de géneros. Esta entrega contiene código fuente, sin ejecutables compilados. `INICIAR.bat` abre la app con doble clic una vez preparado el entorno.

## Ejecutar

Si ya tienes el entorno preparado, abre una consola en la carpeta del proyecto:

```powershell
.venv\Scripts\python.exe app.py
```

Para preparar el proyecto por primera vez, instala Python 3.11 de 64 bits y ejecuta:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

No necesitas activar el entorno virtual. FFmpeg se obtiene mediante la dependencia imageio-ffmpeg. Las dependencias necesitan internet durante la instalación; el modo Ligero funciona después localmente.

## Uso y cambios

Las herramientas ocasionales están en la barra de menús superior:

- **Biblioteca**: importar carpeta, exportar CSV, copia de seguridad y abrir datos.
- **Análisis**: analizar/continuar, pausar tarea y seleccionar el motor Ligero o CLAP.
- **Clasificador**: entrenar con mis etiquetas, actualizar sugerencias y ver el informe.
- **Ver**: mostrar u ocultar el registro de actividad y los detalles/estadísticas de la canción. Ambos paneles están ocultos por defecto; se recuerda tu elección. La lista de géneros ocupa el espacio disponible.

Los menús superiores y sus opciones resaltan al pasar el ratón. Encima de todos los géneros aparece **Sugerencias · Top 3**, con las tres puntuaciones más altas guardadas para la canción seleccionada. Un clic asigna el género inmediatamente, conserva los otros y marca la canción como revisada. Se muestran para una sola canción seleccionada; si no tiene predicciones, se indica. Al asignar se limpian sus sugerencias anteriores, igual que al editar sus etiquetas.

Las acciones incompatibles se desactivan mientras hay una tarea en curso. «Pausar tarea» está disponible durante importación o análisis.

Junto al reproductor aparece **Ahora sonando: artista - título**. El texto indica también pausa o detención y corresponde a la pista cargada, independientemente de la selección de la tabla.

Selecciona una sola canción y pulsa **Buscar género en Google** para abrir en tu navegador predeterminado una consulta como `de qué género es la canción "Zonderling - Kronkel"`. Usa los metadatos de artista y título; si no hay título, usa el nombre del archivo. El botón se desactiva sin selección o con varias canciones. La búsqueda no cambia las etiquetas automáticamente.

- Importa una carpeta y pulsa **Analizar / continuar**. Pausar termina primero el archivo actual; puedes continuar después.
- Selecciona una canción y pulsa reproducir, o haz doble clic. La forma de onda de **toda la canción** sustituye a la barra del track; el espectrograma completo aparece debajo. Se cargan en segundo plano leyendo todo el audio y reduciéndolo a una vista de resolución limitada, sin guardar la canción decodificada completa en memoria. No requieren analizar ni entrenar previamente. Al cambiar de canción se cancela la vista anterior y se descartan sus resultados. La primera carga puede tardar, especialmente en pistas largas.
- Haz clic o arrastra sobre la forma de onda para cambiar de posición. Un cursor vertical muestra la reproducción sobre la onda y el espectrograma; al moverse solo se repinta el cursor, sin recalcular el audio. Las flechas izquierda/derecha avanzan o retroceden cinco segundos con la vista enfocada. El volumen conserva una barra con clic directo y arrastre. Ya no hay vistas de 12 segundos.
- Todas las columnas, incluida **Título / artista**, se pueden redimensionar arrastrando el borde de su cabecera. Haz clic en cualquier cabecera de la tabla para ordenar; otro clic invierte el orden. BPM, bitrate, frecuencia y corte se ordenan como números; sugerencias, por su puntuación máxima. Los valores ausentes aparecen primero en orden ascendente.
- La última columna muestra la **ruta del archivo**. El botón **Columnas…** permite mostrar u ocultar columnas mediante casillas; la elección se conserva al cerrar la app. Ocultar una columna no elimina datos.
- **Un clic en un botón de género lo añade y guarda inmediatamente** en las canciones seleccionadas, conservando sus demás géneros. Si todas ya lo tienen, el clic lo quita. Los botones resaltados indican géneros comunes a toda la selección; en una selección mixta el clic añade el género a las que les falta. No hay casillas ni botón Guardar. Quitar el último género devuelve la canción a «Sin revisar». El desplegable inferior solo elige el género que deseas renombrar o eliminar.
- **Eliminar género** elimina el género seleccionado de la lista y de las canciones que lo tenían. Esas canciones quedan sin revisar y conservan otros géneros. Si no les queda ninguno, quedan por identificar. Se invalidan las sugerencias: vuelve a entrenar. Los géneros eliminados no reaparecen al reiniciar.
- **Eliminar archivo del dispositivo** borra definitivamente los archivos de las filas seleccionadas y sus registros de la biblioteca. Solicita confirmación, con las rutas en «Mostrar detalles», y no usa la papelera. Detiene y libera la pista abierta. Si el sistema impide borrar un archivo, conserva su registro y muestra el error. Una ruta que ya no existe se elimina de la biblioteca. Durante análisis/importación/entrenamiento el botón está desactivado; si se está decodificando el archivo que vas a borrar, espera a que termine. El borrado quita solo las filas afectadas y actualiza las sugerencias invalidadas; no reconstruye la tabla ni la interfaz. Mantiene el ancho de columnas y la reproducción de otra pista.

## Ligero y CLAP: diferencias en esta aplicación

| | Ligero | CLAP |
|---|---|---|
| Representación | 60 características acústicas: MFCC, contraste espectral, croma y BPM | Huella de audio de un modelo preentrenado, congelado |
| Coste | Menor consumo; sin descargar una red neuronal | Más memoria, espacio y tiempo de análisis; dependencias y descarga adicionales |
| Entrenamiento personal | Regresión logística por género con tus etiquetas revisadas | El mismo tipo de clasificador, usando las huellas de CLAP |
| Precisión | Debe medirse con tus canciones | Puede aportar una representación más rica; no garantiza distinguir mejor todos tus subgéneros |

Ninguno aprende continuamente al pulsar una etiqueta: guarda las correcciones y pulsa **Entrenar con mis etiquetas**. No se entrena CLAP desde cero. Ambos necesitan ejemplos revisados; esta app no usa descripciones de texto para clasificar sin ejemplos. El bitrate y el corte espectral no se usan como características de género.

En tu equipo puedes empezar con Ligero. CLAP está implementado en CPU; la GTX 1060 no es necesaria. Cambiar de motor y analizar reemplaza las características guardadas de cada canción; después debes reentrenar con ese motor. Las etiquetas se conservan.

### Preparar CLAP opcional

```powershell
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-clap.txt
.venv\Scripts\python.exe download_model.py
```

La descarga explícita obtiene `laion/clap-htsat-unfused` y lo guarda en la carpeta de datos. Después selecciona CLAP, analiza y entrena. No se sube música. La descarga y ejecución real de CLAP no se han validado en este entorno.

## Entrenamiento y límites

Se necesitan al menos diez canciones analizadas y revisadas; cada género entrenable requiere al menos tres positivos y tres negativos. Son mínimos de funcionamiento, no una garantía de precisión. Admite varias etiquetas y estilos nuevos. Se guardan coeficientes numéricos en JSON, sin cargar pickle.

La evaluación se realiza cuando hay suficientes ejemplos y grupos, separando por artista o por archivo si no hay artista. Evita duplicados idénticos, pero no agrupa automáticamente todos los remixes. Consulta **Ver informe del modelo**. Las puntuaciones no son probabilidades calibradas.

El análisis usa hasta tres fragmentos por canción. El BPM puede ser mitad o doble del real. El corte a 16/18/20 kHz es un indicio espectral, no prueba de recompresión ni una forma segura de conocer el bitrate original. RMS no equivale a LUFS. No se infiere contenido por encima de la frecuencia de Nyquist del archivo.

## Datos y actualización

En Windows los datos están en `%LOCALAPPDATA%\ElectronicaLab`: `library.sqlite3`, `personal-model.json` y, si lo descargas, `clap/`. Actualizar el código no borra esos datos. La app no escribe etiquetas dentro de los archivos musicales. El borrado definitivo sí elimina el archivo original seleccionado.

**Copia de seguridad** exporta SQLite (etiquetas y análisis); el modelo personal se puede regenerar. **Exportar CSV** incluye rutas, metadatos y resultados. Una copia de SQLite no contiene los archivos de música.

## Desarrollo y validación

```powershell
.venv\Scripts\python.exe -m pip install ruff
.venv\Scripts\python.exe -m ruff format .
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas gráficas se ejecutan en un proceso separado para aislar Qt del análisis numérico. Cubren importación, análisis, entrenamiento, borrado y sus errores, persistencia de géneros y columnas, ordenación/selección, volumen, clic sobre la onda, reutilización del gráfico al mover el cursor, cobertura del final de la canción, etiquetado de un clic y borrado sin reconstruir filas supervivientes. Quedan pendientes la reproducción audible y el comportamiento específico del backend multimedia y bloqueo de archivos en Windows.
