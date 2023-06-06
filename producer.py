import csv
import os
from kafka import KafkaProducer
import time
import sys
import pandas as pd

bootstrap_servers = ['localhost:9092']
topicName = 'attacks'

original_file = "./data/x_test.csv"
temp_file = "./data/x_test_id.csv"

# Creamos un archivo temporal para tener el x_test con IDs que se borrará al apagar el producer
with open(original_file, "r") as csv_in, open(temp_file, "w", newline="") as csv_out:
    reader = csv.reader(csv_in)
    writer = csv.writer(csv_out)

    header = next(reader)
    writer.writerow(["ID"] + header)

    for i, row in enumerate(reader):
        writer.writerow([i] + row)

csv_in.close()
csv_out.close()

data = open("./data/x_test_id.csv","r")

producer = KafkaProducer(bootstrap_servers = bootstrap_servers)

try:
    skip_first_row = True
    n_muestras = 0

    for line in data:
        if skip_first_row:
            skip_first_row = False
            continue
        
        print('Se va a enviar el siguiente ataque:')
        print(line)
        producer.send(topicName, line.encode('utf-8'))
        n_muestras = n_muestras +1
        time.sleep(1)

except KeyboardInterrupt:
    print('Se ha cortado el envío de muestras')
    print('Se han enviado {} muestras'.format(n_muestras))
    data.close()
    os.remove(temp_file)
    sys.exit()

print('hola')
producer.flush()
producer.close()

