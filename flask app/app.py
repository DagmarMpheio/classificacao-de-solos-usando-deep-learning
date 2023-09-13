import os
import uuid
import flask
import urllib
from PIL import Image
from keras.models import load_model
from flask import Flask, render_template, request, send_file
from keras.utils import load_img, img_to_array

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#carregar o modelo treinado
model = load_model(os.path.join(BASE_DIR, 'soil-detention-75-epochs.h5'))

""" Extensões permitidas """
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['Solo Aluvial', 'Solo Negro', 'Solo Argiloso', 'Solo Vermelho']

# metodo de classificacao do modelo
def predict(filename, model):
    # carregar a imagem com as devidas dimensoes
    img = load_img(filename, target_size=(100, 100))
    # transformar a imagem em array
    img = img_to_array(img)
    # redimencionar a imagem
    img = img.reshape(1, 100, 100, 3)

    # transformar o array sob forma de imagens em float
    img = img.astype('float32')
    img = img/255.0
    # classificar a imagem
    result = model.predict(img)

    # lidar com as classes existentes e armazenar no dict_result
    dict_result = {}
    for i in range(4):
        dict_result[result[0][i]] = classes[i]

    # primeiro resultado obtido na classificacao feita pelo modelo
    res = result[0]
    # ordernar os resultados por ordem crescente
    res.sort()
    # ordernar os resultados por ordem decrescente
    res = res[::-1]
    # obter as 4 primeiras probabilidades
    prob = res[:4]

    # associar as 4 classes com cada probabilidade
    prob_result = []
    class_result = []
    for i in range(4):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result, prob_result


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    # directiorio 'static/images' para enviar a imagens
    target_img = os.path.join(os.getcwd(), 'static/images')
    # se o usuario preencher o formulario, faça
    if request.method == 'POST':
        # se a imagem vir atraves da URL, faça
        if(request.form):
            link = request.form.get('link')
            try:
                # abrir o link
                resource = urllib.request.urlopen(link)
                # nome do ficheiro sera com strings aleatorias e unicos
                unique_filename = str(uuid.uuid4())
                # o formato sera jpg
                filename = unique_filename+".jpg"
                # definir o directorio onde a imagem obtida pelo link sera enviada
                img_path = os.path.join(target_img, filename)
                # abrir a imagem obtida pelo link
                output = open(img_path, "wb")
                # guardar a imagem 'static/images'
                output.write(resource.read())
                # fechar o buffer de arquivos
                output.close()
                # nome da imagem
                img = filename

                # armazenar os resultados calculados pelo modelo
                class_result, prob_result = predict(img_path, model)

                # armazenar as 2 classes e 2 probabilidades calculadas pelo modelo na varivel predictions
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                }

            except Exception as e:
                print(str(e))
                error = 'Esta imagem deste site não é acessível ou uma entrada inapropriada'

            # se nao houver erros, abrir a pagina de sucesso
            if(len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            # senao mostrar erro
            else:
                return render_template('index.html', error=error)

        # se a imagem for escolhida pelo usuario localmente
        elif (request.files):
            # obter o ficheiro atraves da requisicao
            file = request.files['file']
            # se na requisicao tiver um ficheiro e está de acordo com as extensões permitidas, faça
            if file and allowed_file(file.filename):
                # enviar todas imagens no directiorio 'static/images'
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                # nome da imagem
                img = file.filename

                # armazenar os resultados calculados pelo modelo
                class_result, prob_result = predict(img_path, model)

                # armazenar as 4classes e 4probabilidades calculadas pelo modelo na varivel predictions
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                }

            # motrar erro se o ficheiro nao é uma imagem
            else:
                error = "Por favor, faca upload de imagens apenas nas extensões jpg , jpeg e png"

            # se nao houver erros, abrir a pagina de sucesso
            if(len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)

            # senao mostrar erro
            else:
                return render_template('index.html', error=error)

    # se o usuario nao preencher o formulario, mantem na mesma pagina
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run()
