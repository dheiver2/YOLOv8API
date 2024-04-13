```markdown
<div align="center">
  <img width="450" src="assets/Flask_logo.svg">
</div>

# Yolov8 Flask API para detecção e segmentação

<div align="center">
  <a href="https://www.buymeacoffee.com/hdnh2006" target="_blank">
    <img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Me pague um café">
  </a>
</div>

![GIF da tela](assets/screen.gif)

Este código é baseado no código YOLOv8 da Ultralytics e tem todas as funcionalidades que o código original tem:
- Diferentes fontes: imagens, vídeos, webcam, câmeras RTSP.
- Todos os pesos são suportados: TensorRT, Onnx, DNN, openvino.

A API pode ser chamada de forma interativa e também como uma única chamada de API a partir do terminal e suporta todas as tarefas fornecidas pelo YOLOv8 (deteção, segmentação, classificação e estimativa de pose) na mesma API!!!!

Todos os [modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) são baixados automaticamente a partir do último [release](https://github.com/ultralytics/assets/releases) da Ultralytics na primeira utilização.

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png">

## Requisitos

Python 3.8 ou posterior com todas as dependências do [requirements.txt](requirements.txt) instaladas, incluindo `torch>=1.7`. Para instalar, execute:

```bash
$ pip3 install -r requirements.txt
```

## Modelos de [Deteção](https://docs.ultralytics.com/tasks/detect), [Segmentação](https://docs.ultralytics.com/tasks/segment), [Classificação](https://docs.ultralytics.com/tasks/classify) e [Estimativa de Pose](https://docs.ultralytics.com/tasks/pose) pré-treinados no [COCO](https://docs.ultralytics.com/datasets/detect/coco) na mesma API

`predict_api.py` pode lidar com várias fontes e pode ser executado na CPU, mas é altamente recomendável rodá-lo na GPU.

```bash
Uso - formatos:
    $ python predict_api.py --weights yolov8s.pt                # PyTorch
                                     yolov8s.torchscript        # TorchScript
                                     yolov8s.onnx               # ONNX Runtime ou OpenCV DNN com --dnn
                                     yolov8s_openvino_model     # OpenVINO
                                     yolov8s.engine             # TensorRT
                                     yolov8s.mlmodel            # CoreML (apenas macOS)
                                     yolov8s_saved_model        # TensorFlow SavedModel
                                     yolov8s.pb                 # TensorFlow GraphDef
                                     yolov8s.tflite             # TensorFlow Lite
                                     yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                                     yolov8s_paddle_model       # PaddlePaddle

Uso - tarefas:
    $ python predict_api.py --weights yolov8s.pt                # Detecção
                                     yolov8s-seg.pt             # Segmentação
                                     yolov8s-cls.pt             # Classificação
                                     yolov8s-pose.pt            # Estimativa de Pose
```

## Implementação interativa

Você pode implantar a API capaz de rotular de forma interativa.

Execute:

```bash
$ python predict_api.py --device cpu # para rodar na CPU (por padrão é GPU)
```
Abra o aplicativo em qualquer navegador 0.0.0.0:5000 e faça upload da sua imagem ou vídeo conforme mostrado no vídeo acima.


## Como usar a API

### Forma interativa
Basta abrir seu navegador favorito e ir para 0.0.0.0:5000 e intuitivamente carregar a imagem que deseja rotular e pressionar o botão "Enviar imagem".

A API retornará a imagem ou vídeo rotulado.

![Alt text](assets/zidane_bbox.png)

Todas as tarefas são suportadas, para estimativa de pose, os resultados devem ser como segue:
![Pose do Zidane](assets/zidane_pose.png)


### Chamada a partir do terminal ou script Python
O código `client.py` fornece vários exemplos sobre como a API pode ser chamada. Uma maneira muito comum de fazer isso é chamar uma imagem pública a partir de uma URL e obter as coordenadas das caixas delimitadoras:

```python
import requests

resp = requests.get("http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg&save_txt=T",
                    verify=False)
print(resp.content)

```
E você obterá um json com os seguintes dados:

```
b'{"results": [{"name": "person", "class": 0, "confidence": 0.8892598748207092, "box": {"x1": 747.315673828125, "y1": 41.47210693359375, "x2": 1140.3927001953125, "y2": 712.91650390625}}, {"name": "person", "class": 0, "confidence": 0.8844665288925171, "box": {"x1": 144.88815307617188, "y1": 200.0352783203125, "x2": 1107.232177734375, "y2": 712.7000732421875}}, {"name": "tie", "class": 27, "confidence": 0.7176544070243835, "box": {"x1": 437.38336181640625, "y1": 434.477294921875, "x2": 529.9751586914062, "y2": 717.05126953125}}]}'
```

No caso da estimativa de pose, os resultados são assim:
```
b'{"results": [{"name": "person", "class": 0, "confidence": 0.9490957260131836, "box": {"x1": 239.0, "y1": 15.0, "x2": 1018.0, "y2": 1053.0}, "keypoints": {"x": [604.9951782226562, 653.2091064453125, 552.5707397460938, 697.6889038085938, 457.49749755859375, 786.6876831054688, 358.194

091796875, 954.072998046875, 488.3907775878906, 684.831298828125, 802.8469848632812, 687.2332153320312, 412.4287414550781, 924.52685546875, 632.3346557617188, 811.2559814453125, 768.5433349609375], "y": [316.5501403808594, 260.7156066894531, 257.27691650390625, 291.1667175292969, 285.6615905761719, 566.11962890625, 596.4549560546875, 909.6119384765625, 965.7925415039062, 997.584716796875, 841.6057739257812, 1066.0, 1066.0, 850.1934204101562, 812.7511596679688, 954.5965576171875, 951.3284912109375], "visible": [0.9959749579429626, 0.9608340859413147, 0.9934138655662537, 0.4281827211380005, 0.9349473118782043, 0.9848191738128662, 0.9723504185676575, 0.8565006852149963, 0.8561225533485413, 0.9004713296890259, 0.9377612471580505, 0.10934382677078247, 0.08168646693229675, 0.008380762301385403, 0.008864155039191246, 0.0017155600944533944, 0.001865472993813455]}}]}'
```

## A FAZER
- [ ] Retornar valores de txt para vídeos
- [ ] salvar pasta de acordo com a tarefa: detectar, posar, segmentar, ...
- [ ] Suportar qualquer outro modelo: SAM, RTDETR, NAS.
- [ ] Arquivos Docker
- [ ] Melhorar o modelo de índice


## Sobre mim e contato

Este código é baseado no código YOLOv8 da Ultralytics e foi modificado por Henry Navarro
 
Se você quiser saber mais sobre mim, visite meu blog: [henrynavarro.org](https://henrynavarro.org).
