'''
TCC - IHM de olhos por meio de Visão Computacional
Código criado por Igor Ciampi em 09/09/2023
Última vez editado em 20/10/2023
'''

''' Bibliotecas '''

import cv2 # OpenCV 2 - ferramentas de visão computacional e processamento digital de imagem
import numpy as np # Numpy - ferramentas matemáticas
import dlib # DLib - conjunto de ferramentas para visão computacional e machine learning
import ui
from math import hypot
import time
import datetime

''' Funções '''

def midpoint(p1, p2):
    ''' Função que encontra o ponto médio entre dois pontos '''
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
        
        ''' 
        Função que tem como input uma lista com os pontos oculares (já definidos no modelo), as landmarks faciais e output como
        a razão entre a medida ocular vertical e a medida ocular horizontal. Essa função serve para mais de um olho, sendo assim uma
        forma de reduzir o código.
        
        '''

        # Crio um ponto no lado esquerdo, no lado direito, no topo e abaixo de um olho
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))


        # Crio segmentos de reta horizontal e vertical, ligando os pontos criados
        #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        # Mensuro os tamanhos das linhas - quando se pisca, o tamanho da linha vertical diminui
        hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
        # Observação: para tunar esse valor, basta usar o print da linha vertical e ver o quanto é para aquela pessoa o valor do limiar de blink

        # Razão entre linha horizontal e linha vertical
        ratio = hor_line_len / ver_line_len
        return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
     # Detecção do olhar

        # Região do olho - array com os pontos de landmarks
        eye_region = np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        
        
        # Deve-se criar uma máscara que cubra toda a imagem, exceto o olho
        # Uma vez que a máscara cobrir toda a imagem exceto o olho, pode-se pegar o olho
        height, width, _ = frame.shape
        mask = np.zeros((height,width), np.uint8)
        
        # Mostra a região do olho
        #cv2.polylines(frame, [eye_region], True, (0, 255, 0), 2)

        # Faz o preenchimento do olho com cor branca na máscara criada
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
    
        # Pego os pontos extremos do olho
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        max_y = np.max(eye_region[:, 1])
        min_y = np.min(eye_region[:, 1])

        # Pego os frames que correspondem ao olho
        gray_eye = eye[min_y:max_y, min_x:max_x]

        #gray_eye = cv2.equalizeHist(gray_eye)

        # Escala de preto e branco que separa a posição da íris e da pupila do olho - importante para definir a direção que estou olhando
        # Uma parte fica toda branca, enquanto outra está preta
        #_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY) # 70 e 255  definem um limite de binarização entre preto e branco
        #_, threshold_eye = cv2.threshold(gray_eye, 80, 255, cv2.THRESH_BINARY)
        _, threshold_eye = cv2.threshold(gray_eye, 110, 255, cv2.THRESH_BINARY)

        #colocar para + de 70 (abaixo = ruido) 

        # Pego altura e largura do olho
        height, width = threshold_eye.shape
        
        # Pego os valores numéricos do threshold
        left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
        
        # Pego os valores da parte branca do olho - não zeros são partes brancas/em escala, enquanto zeros [0,0,0] são partes pretas do olho
        left_side_white = cv2.countNonZero(left_side_threshold)

        # Repito para o lado direito
        right_side_threshold = threshold_eye[0: height, int(width/2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        
        # Para cima
        up_side_threshold = threshold_eye[0: int(height/2), 0: width]
        up_side_white = cv2.countNonZero(up_side_threshold)
        
        # Para baixo
        bottom_side_threshold = threshold_eye[int(height/2): height, 0: width]
        bottom_side_white = cv2.countNonZero(bottom_side_threshold)
        
        # Problema com o vertical: olhar para baixo não gera região branca!

        # Razão do olhar - indica para qual lado na HORIZONTAL estou olhando
        # Condicionais de segurança para evitar erro de divisão por zero - coloco limites de valores para esquerda e direita
        
        if left_side_white == 0:
            hor_gaze_ratio = 0.1
        elif right_side_white == 0:
             hor_gaze_ratio = 0.9
        else: 
            hor_gaze_ratio = left_side_white/right_side_white

        
        # Repete-se o mesmo para a vertical
        if up_side_white == 0:
            ver_gaze_ratio = 0.1
        elif bottom_side_white == 0:
             ver_gaze_ratio = 0.9
        else: 
            ver_gaze_ratio = up_side_white/bottom_side_white
        

        #gaze_ratio = (hor_gaze_ratio + ver_gaze_ratio)/2

        # Testes na horizontal
        #hor_gaze_ratio = left_side_white/right_side_white
        #cv2.putText(frame, str(left_side_white), (50,100), font, 2, (0,0,255), 3)
        #cv2.putText(frame, str(right_side_white), (50,150), font, 2, (0,0,255), 3)
        #cv2.putText(frame, str(hor_gaze_ratio), (50,200), font, 2, (255,0,0), 3)
        #print(f"Horizontal: {hor_gaze_ratio}")
        
        # Testes na vertical
        #ver_gaze_ratio = up_side_white/bottom_side_white
        #cv2.putText(frame, str(up_side_white), (100,100), font, 2, (0,255,0), 3)
        #cv2.putText(frame, str(bottom_side_white), (100,150), font, 2, (0,255,0), 3)
        #cv2.putText(frame, str(ver_gaze_ratio), (100,200), font, 2, (255,0,0), 3)
        #print(f"Vertical:{ver_gaze_ratio}")

        # Teste total
        #cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0,0,255), 3)
        #print(gaze_ratio)

        # Aumento do frame do olho   
        eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        threshold_eye = cv2.resize(threshold_eye, None, fx=3, fy=3)

        # "Prints" do OpenCV
        cv2.imshow("Eye", eye)
        cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Right Eye", right_eye)
        

        return hor_gaze_ratio, ver_gaze_ratio

def select_option(option, blink_verify):
    
    if option == 0 and blink_verify == True:
          print("Opção esquerda superior selecionada!")
          blink_verify = False

    elif option == 1 and blink_verify == True:
          print("Opção superior selecionada!")
          blink_verify = False

    elif option == 2 and blink_verify == True:
          print("Opção direita superior selecionada!")
          blink_verify = False
    
    elif option == 3 and blink_verify == True:
          print("Opção esquerda selecionada!")
          blink_verify = False
    
    elif option == 4 and blink_verify == True:
          print("Opção central selecionada!")
          blink_verify = False
    
    elif option == 5 and blink_verify == True:
          print("Opção direita selecionada!")
          blink_verify = False
    
    elif option == 6 and blink_verify == True:
          print("Opção inferior esquerda selecionada!")
          blink_verify = False

    elif option == 7 and blink_verify == True:
          print("Opção inferior selecionada! selecionada!")
          blink_verify = False

    elif option == 8 and blink_verify == True:
          print("Opção inferior direita selecionada!")
          blink_verify = False
          
    return option  


''' Variáveis '''

# Contadores
frames = 0
letter_index = 0
blinking_frames = 0
contador_est = 0
frames_transition = 0


# Fonte que aparecerá na tela do OpenCV
font = cv2.FONT_HERSHEY_PLAIN

# Threshold ou valor limiar da razão para determinar se uma pessoa está piscando
blink_lim = 5.7 # IDEIA: treinar uma RNA que detecte automaticamente esse valor baseando-se na pessoa, distância, etc

# Verificação de piscar

blink_verify = False

''' Setup do OpenCV, do detector e do preditor '''

# Captura o video da webcam/display padrão
cap = cv2.VideoCapture(0)

# Detector do rosto
detector = dlib.get_frontal_face_detector()

# Preditor de landmarks do rosto
predictor = dlib.shape_predictor("TCC\shape_predictor_68_face_landmarks.dat")

''' Aplicação '''

while True:
    
    # Pega o timestamp do inicio

    current_time = datetime.datetime.now()
    t_init = current_time.timestamp()

    # Lê cada frame do vídeo
    _, frame = cap.read()
    
    # Reseta a luz da tecla
    ui.keyboard[:] = (0,0,0)

    # Acrescenta +1 no contador de frames
    
    frames += 1
    print(f"Frames = {frames}")

    # Contador estatístico

    if frames >= 100:
         media = (contador_est/frames) * 100
         print(f"Media geral = {media}%")
         media_frames_detectados = (contador_est/(frames-frames_transition)) * 100
         print(f"Media por frames detectados = {media_frames_detectados}%")
         media_frames_nao_detectados = (frames_transition / frames) * 100
         print(f"Media por frames nao detectados = {media_frames_nao_detectados}%")
         
         

    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Array com rostos em escala de cinza
    faces = detector(gray)
    
    # Letra ativa
    active_letter = ui.keys_set_1[letter_index]

    # Percorre os rostos no array de rostos detectados pelo detector
    for face in faces:
        
        # Para plotar um retangulo na face basta descomentar essas linhas abaixo
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)

        # Cria as landmarks para o rosto em grayscale - posições exatas de todos os pontos do rosto
        landmarks = predictor(gray, face)

        # Razão de piscar para cada olho
        right_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        left_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        
        # Caso essa razão seja maior que o padrão para o piscar de olhos da pessoa, printa BLINKING na tela
        # Caso a pessoa pisque por 5 frames, considera-se uma seleção
        if left_eye_ratio > blink_lim:
            
            cv2.putText(frame, "PISCANDO", (50, 150), font, 3, (255, 0, 0))

            #Contadores de frames com detecção de piscar de olhos
            blinking_frames += 1
            
            #frames -= 1
            
            if blinking_frames == 5:
                
                
                blink_verify = True

                cv2.putText(frame, "Detectei seleção", (50, 200), font, 3, (255, 0, 0))
                
                opt = select_option(letter_index, blink_verify)

                time.sleep(1)
        
        else:
            blinking_frames = 0
            blink_verify = False
            

        # Detecção do olhar
        hor_gaze_ratio_left_eye, ver_gaze_ratio_left_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        hor_gaze_ratio_right_eye, ver_gaze_ratio_right_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        
        # Realizo uma média entre os "gazes" e obtenho uma medida mais exata
        hor_gaze_ratio = (hor_gaze_ratio_left_eye + hor_gaze_ratio_right_eye) / 2
        ver_gaze_ratio = (ver_gaze_ratio_left_eye + ver_gaze_ratio_right_eye) / 2

        #gaze_ratio = (hor_gaze_ratio + ver_gaze_ratio) / 2
        #print(f"Final: {gaze_ratio}")
        
        # Empiricamente nota-se faixas de valores que classificam regiões em que se está olhando
        # Necessita de fine tuning!
        #cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0,0,255), 3)
        
        '''
        # Código antigo - teclado com 5 opções
        if hor_gaze_ratio < 0.8 and 0.5 < ver_gaze_ratio < 0.8 :
             cv2.putText(frame, "DIREITA", (50,100), font, 2, (0,0,255), 3)
             letter_index = 5
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, DIREITA")
        
        elif 0.9 < hor_gaze_ratio < 1.0 and 0.5 < ver_gaze_ratio < 0.8 :
             cv2.putText(frame, "MEIO", (50,100), font, 2, (0,0,255), 3)
             letter_index = 4
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, MEIO")
        
        elif hor_gaze_ratio > 1.1 and 0.5 < ver_gaze_ratio < 0.8 :
             cv2.putText(frame, "ESQUERDA", (50,100), font, 2, (0,0,255), 3)
             letter_index = 3
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, ESQUERDA")
        
        elif 0.8 < hor_gaze_ratio < 1.1 and ver_gaze_ratio < 0.4 :
             
             cv2.putText(frame, "CIMA", (50,100), font, 2, (0,0,255), 3)
             letter_index = 1
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CIMA")
        
        elif 0.8 < hor_gaze_ratio < 1.1 and ver_gaze_ratio > 0.9 :
             cv2.putText(frame, "BAIXO", (50,100), font, 2, (0,0,255), 3)
             letter_index = 7
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, BAIXO")
          '''
        

        if hor_gaze_ratio < 0.8 and 0.6 < ver_gaze_ratio < 0.9 :
             cv2.putText(frame, "DIREITA", (50,100), font, 2, (0,0,255), 3)
             
             letter_index = 5
             
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, DIREITA")
             contador_est += 1
             
        
        elif 0.9 < hor_gaze_ratio < 1.1 and 0.6 < ver_gaze_ratio < 0.9 :
             cv2.putText(frame, "MEIO", (50,100), font, 2, (0,0,255), 3)
             
             letter_index = 4
             
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, MEIO")
             
        
        elif hor_gaze_ratio > 1.2 and 0.6 < ver_gaze_ratio < 0.9 :
             cv2.putText(frame, "ESQUERDA", (50,100), font, 2, (0,0,255), 3)
             
             letter_index = 3

             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, ESQUERDA")
             
        
        elif 0.9 < hor_gaze_ratio < 1.1 and ver_gaze_ratio < 0.5 :
             
             cv2.putText(frame, "CIMA", (50,100), font, 2, (0,0,255), 3)

             letter_index = 1
             
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CIMA")
        
        elif 0.9 < hor_gaze_ratio < 1.1 and ver_gaze_ratio > 1.0 :
             cv2.putText(frame, "BAIXO", (50,100), font, 2, (0,0,255), 3)

             letter_index = 7
             
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, BAIXO")
             

        elif hor_gaze_ratio > 1.2 and ver_gaze_ratio < 0.5 :
             cv2.putText(frame, "CANTO SUP. ESQ.", (50,100), font, 2, (0,0,255), 3)
             letter_index = 0
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CANTO SUP. ESQ.")
             

        elif hor_gaze_ratio < 0.8 and ver_gaze_ratio < 0.5 :
             cv2.putText(frame, "CANTO SUP. DIR.", (50,100), font, 2, (0,0,255), 3)
             letter_index = 2
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CANTO SUP. DIR.")
             

        elif hor_gaze_ratio > 1.2 and ver_gaze_ratio > 1.0 :
             cv2.putText(frame, "CANTO INF. ESQ.", (50,100), font, 2, (0,0,255), 3)
             letter_index = 6
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CANTO INF. ESQ.")
             

        elif hor_gaze_ratio < 0.8 and ver_gaze_ratio > 1.0 :
             cv2.putText(frame, "CANTO INF. DIR.", (50,100), font, 2, (0,0,255), 3)
             letter_index = 8
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, CANTO INF. DIR.")

        else:
             print(f"Hor. Gaze = {hor_gaze_ratio}, Ver. Gaze = {ver_gaze_ratio}, Zona de Transição")
             frames_transition += 1


        

        # Aumento do frame do olho   
        #eye = cv2.resize(gray_eye, None, fx=5, fy=5)

        # "Prints" do OpenCV
        #cv2.imshow("Eye", eye)
        #cv2.imshow("Threshold", threshold_eye)
        #cv2.imshow("Right Eye", right_eye)

    # Atualiza contadores
    if letter_index == 10:
         letter_index += 1
         frames = 0

    # Monta o teclado
    for i in range(9):
        if i == letter_index:
            light = True
        else:
            light = False 
        ui.letter(i, ui.keys_set_1[i], light)
     
    # Tempo final de execução
    current_time = datetime.datetime.now()
    t_final = current_time.timestamp()

    # Diferença entre tempo inicial e final
    delta_time = t_final - t_init
    print(f"Delta time = {delta_time}")

    # Framerate
    fps = 1/delta_time
    print(f"Framerate = {fps} FPS")

    # Mostra o frame
    cv2.imshow("Menu", ui.keyboard)
    cv2.imshow("Frame", frame)

    # Caso eu aperte "ESC" no teclado, saio do loop e finalizo o código
    key = cv2.waitKey(1)
    if key == 27:
        break

# Sai da captura de vídeo e fecha todas as janelas abertas pelo OpenCV
cap.release()
cv2.destroyAllWindows()