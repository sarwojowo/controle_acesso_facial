import cv2
import statistics
from pyfirmata import Arduino,SERVO
from time import sleep

port = 'COM3'
pin = 10
board = Arduino(port)

board.digital[pin].mode = SERVO

def rotateServo(pin,angle):
    board.digital[pin].write(angle)
    sleep(0.015)

camera = cv2.VideoCapture(1)
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")

def capturar():
    print('Reconhecendo Rosto')
    idSeq= []
    i = 0
    #definir tempo em que o loop estará rodando (millesegundos)
    while i <150:
        success, img = camera.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classificador.detectMultiScale(imgGray, scaleFactor=1.5, minSize=(50, 50))

        for (x, y, l, a) in faces:
            imgFace = cv2.resize(imgGray[y:y+a,x:x+l],(220,220))
            cv2.rectangle(img, (x, y), (x + l, y + a), (1, 237, 0), 2)
            id, confianca = reconhecedor.predict(imgFace)
            idSeq.append(id)

        cv2.imshow('Cam',img)
        cv2.waitKey(1)
        i =i+1
    cv2.destroyAllWindows()

    if idSeq:
        return statistics.mode(idSeq)
    else:
        #caso o rosto não seja reconhecido nenhuma vez
        return -1


def monitor():

    while True:
        success, img = camera.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classificador.detectMultiScale(imgGray, scaleFactor=1.5, minSize=(100, 100))

        if len(faces) >0:
            print('Face encontrada!')
            return True
            break
        else:
            return False


def main():
    while True:
        # num0 = monitor aberto, num -1 = pessoa desconhecida, num 1 acima= pessoa conhecida
        print('Monitorando ...')
        num = 0
        comando = monitor()
        if comando:
            num = capturar()
            if num ==0:
                pass
            elif num==-1:
                print('pessoa desconhecida!')
            elif num >=1:
                rotateServo(pin, 100)
                sleep(5)
                rotateServo(pin, 0)



main()