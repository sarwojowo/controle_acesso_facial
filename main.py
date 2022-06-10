import cv2
import statistics

camera = cv2.VideoCapture(1)
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")

def capturar():
    idSeq= []
    i = 0
    #definir tempo em que o loop estará rodando
    while i <101:
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

    return statistics.mode(idSeq)

num = capturar()
print(str(num))