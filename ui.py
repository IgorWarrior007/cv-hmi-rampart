'''
Interface gráfica - IHM

Código realizado em 21/09/2023 por Igor Ciampi
Última vez editado em 21/09/2023

'''

# Bibliotecas
import cv2
import numpy as np

# Teclado
keyboard = np.zeros((800, 600, 3), np.uint8)

keys_set_1 = {0: 'Q', 1: 'W', 2: 'E', 
              3: 'A', 4: 'S', 5: 'D', 
              6: 'Z', 7: 'X', 8: 'C'
              }

def letter(letter_index, text, letter_light):
    
    # Opções
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 0
        y = 200
    elif letter_index == 4:
        x = 200
        y = 200
    elif letter_index == 5:
        x = 400
        y = 200
    elif letter_index == 6:
        x = 0
        y = 400
    elif letter_index == 7:
        x = 200
        y = 400
    elif letter_index == 8:
        x = 400
        y = 400
    
    width = 200
    height = 200
    thicc = 3

    if letter_light is True:
        cv2.rectangle(keyboard, (x + thicc, y + thicc), (x + width - thicc, y + height - thicc), (255, 255, 255), -1)
    else: 
        cv2.rectangle(keyboard, (x + thicc, y + thicc), (x + width - thicc, y + height - thicc), (255, 0, 0), thicc)

    # Configurações do texto
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 8
    font_thicc = 4
    font_color = (255,0,0)

    # Texto
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_thicc)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text)/2) + x
    text_y = int((height + height_text)/2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, font_color, font_thicc)

'''
# Monta o teclado
for i in range(9):
    if i == 5:
        light = True
    else:
        light = False 
    letter(i, keys_set_1[i], light)
'''

'''
# Mostra na tela
cv2.imshow("Menu", keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''