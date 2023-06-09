import os
import time
import sys
from kafka                          import KafkaProducer


bootstrap_servers = ['localhost:9092']
topicName = 'Attacks'

data = open("./data/x_test.csv","r")

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
    producer.close()

    sys.exit()


