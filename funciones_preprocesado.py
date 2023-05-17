import os
from os.path                    import isfile, join


def ficheros(ruta):

    contenido = os.listdir(ruta)
    archivos = [nombre for nombre in contenido if isfile(join(ruta, nombre))]

    return archivos

