#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Pedro Paulo Moreno Camargo
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from numpy.linalg import inv
import cv2
class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    view_matrix = np.identity(4)
    P = np.identity(4)
    forward_direction = np.array([0,0,1])
    transform = [np.identity(4)]
    mipmaps = []
    directional_light = None
    view_direction = None
    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        # Pega a cor emissiva do dicionario
        emissiveColor = colors.get('emissiveColor')
        #Associa a cor emissiva para valores RGB
        r = int(emissiveColor[0] * 255)
        g = int(emissiveColor[1] * 255)
        b = int(emissiveColor[2] * 255)
        
        # Loopa pela lista de pontos e desenha eles
        for i in range(0, len(point),2):
            x = int(point[i])*2
            y = int(point[i + 1])*2
            # Desenha o ponto
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        # Pega a cor emissiva do dicionario
        emissiveColor = colors.get('emissiveColor')
        #Associa a cor emissiva para valores RGB
        rgb = [int(emissiveColor[0] * 255), int(emissiveColor[1] * 255),int(emissiveColor[2] * 255)]
        # Screen bounds (replace these with the actual screen dimensions)
        xmax, ymax = GL.width, GL.height
        #Referencia: https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm
        def desenha_linha(p0, p1, rgb):
            x0, y0 = p0
            x1, y1 = p1

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            
            if dx > dy:
                err = dx / 2.0
                while x0 != x1:
                    if 0 <= x0 < xmax and 0 <= y0 < ymax:
                        gpu.GPU.draw_pixel([int(x0), int(y0)], gpu.GPU.RGB8, rgb)
                    err -= dy
                    if err < 0:
                        y0 += sy
                        err += dx
                    x0 += sx
                if 0 <= x0 < xmax and 0 <= y0 < ymax:
                    gpu.GPU.draw_pixel([int(x0), int(y0)], gpu.GPU.RGB8, rgb)
            else:
                err = dy / 2.0
                while y0 != y1:
                    if 0 <= x0 < xmax and 0 <= y0 < ymax:
                        gpu.GPU.draw_pixel([int(x0), int(y0)], gpu.GPU.RGB8, rgb)
                    err -= dx
                    if err < 0:
                        x0 += sx
                        err += dy
                    y0 += sy
                if 0 <= x0 < xmax and 0 <= y0 < ymax:
                    gpu.GPU.draw_pixel([int(x0), int(y0)], gpu.GPU.RGB8, rgb)

                
        # Loopa pela lista de pontos e desenha eles
        for i in range(0, len(lineSegments)-2,2):
            p0 = (int(lineSegments[i])*2,int(lineSegments[i + 1])*2)
            p1 = (int(lineSegments[i + 2])*2,int(lineSegments[i + 3])*2)
            desenha_linha(p0,p1,rgb)

    @staticmethod
    def circle2D(radius, colors):
        emissiveColor = colors.get('emissiveColor')
        rgb = [int(emissiveColor[0] * 255), int(emissiveColor[1] * 255),int(emissiveColor[2] * 255)]

        #Referencia: https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
        x = 0
        y = radius*2
        d = 3 - 4 * radius  # Initial decision parameter

        # Screen bounds (replace these with the actual screen dimensions)
        xmax, ymax = GL.width, GL.height
        def draw_circle_points(x0, y0, x, y, rgb):
            # Check bounds and draw the points using symmetry
            points = [
                (x0 + x, y0 + y),
                (x0 - x, y0 + y),
                (x0 + x, y0 - y),
                (x0 - x, y0 - y),
                (x0 + y, y0 + x),
                (x0 - y, y0 + x),
                (x0 + y, y0 - x),
                (x0 - y, y0 - x),
            ]
            for px, py in points:
                if 0 <= px < xmax and 0 <= py < ymax:
                    gpu.GPU.draw_pixel([int(px), int(py)], gpu.GPU.RGB8, rgb)

        # Draw the initial points
        draw_circle_points(0, 0, x, y, rgb)

        while y >= x:
            x += 1
            if d > 0:
                y -= 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
            draw_circle_points(0, 0, x, y, rgb)

    @staticmethod
    def triangleSet2D(vertices, colors, color_vertices=[1,1,1], color_per_vertex=False, Z=None,tex_coords = None,texture = None,normal = None):
        # Cor emissiva ou difusa (default)
        emissiveColor = np.array(colors.get('emissiveColor', [0.0, 0.0, 0.0])) 
        diffuseColor = np.array(colors.get('diffuseColor', [0.0, 0.0, 0.0]))  
        specularColor = np.array(colors.get('specularColor', [0.0, 0.0, 0.0]))  
        shininess = colors.get('shininess', 0) 
        material_ambientIntensity = colors.get('ambientIntensity', 0.0)
        transparency = colors.get('transparency', 0.0)
        if normal is not None and GL.directional_light is not None:  
            view_direction = np.array(GL.view_direction)  
            if len(GL.directional_light) == 0:
                light_direction = view_direction
                light_intensity = 1.0  
                light_color = np.array([1.0, 1.0, 1,0])  
                light_ambient_intensity = 1.0
            else:
                if len(GL.directional_light) == 3:
                    light_direction = [0,0,1]  
                else:
                    light_direction = -np.array(GL.directional_light[3])  
                light_intensity = GL.directional_light[2] 
                light_color = np.array(GL.directional_light[1]) 
                light_ambient_intensity = GL.directional_light[0]
 
            normal = normal
            # Ambient lighting contribution
            ambientI = light_ambient_intensity * diffuseColor * 1

            diffuse_factor = max(np.dot(normal, light_direction), 0.0)
            # Diffuse lighting contribution (Lambertian reflection)
            diffuseI = light_intensity * diffuseColor * diffuse_factor


            halfway = light_direction + view_direction
            halfway_norm = np.linalg.norm(halfway)
            if halfway_norm != 0:
                halfway = halfway / halfway_norm
            NdotH = max(np.dot(normal, halfway), 0.0)
            specular_factor = (NdotH ** (shininess * 128)) 

            # Final specular intensity
            specularI = specularColor * light_intensity * specular_factor
            final_color = (ambientI + diffuseI + specularI) * light_color + emissiveColor
            final_color = np.abs(final_color * 255)  
            default_rgb = np.clip(final_color, 0, 255)

    
        else:
            if emissiveColor[0] == 0 and emissiveColor[1] == 0 and emissiveColor[2] == 0:
                default_rgb = [int(diffuseColor[0] * 255), int(diffuseColor[1] * 255), int(diffuseColor[2] * 255)]
            else:
                default_rgb = [int(emissiveColor[0] * 255), int(emissiveColor[1] * 255), int(emissiveColor[2] * 255)]
        
        def sign(x, y, p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            return ((x - x0) * (y1 - y0)) - ((y - y0) * (x1 - x0))
        
        def inside_triangle(x, y, p0, p1, p2):
            L0 = sign(x, y, p0, p1)
            L1 = sign(x, y, p1, p2)
            L2 = sign(x, y, p2, p0)
            threshold = -1e-6
            return (L0 >= threshold and L1 >= threshold and L2 >= threshold)
        
        def barycentric_coords(p, p0, p1, p2):
            #Parametros para calculo da interpolação baricentrica
            x, y = p
            xA, yA = p0
            xB, yB = p1
            xC, yC = p2
            
            # Calcular alpha
            numer_alpha = -(x - xB) * (yC - yB) + (y - yB) * (xC - xB)
            denom_alpha = -(xA - xB) * (yC - yB) + (yA - yB) * (xC - xB)
            alpha = numer_alpha / denom_alpha
            
            # Calcular beta
            numer_beta = -(x - xC) * (yA - yC) + (y - yC) * (xA - xC)
            denom_beta = -(xB - xC) * (yA - yC) + (yB - yC) * (xA - xC)
            beta = numer_beta / denom_beta
            
            # Calcular gamma
            gamma = 1.0 - alpha - beta

            return alpha, beta, gamma

        
        def interpolated_color(alpha_parameter,beta_parameter,gamma_parameter, interpolated_Z,c0, c1, c2):
            

            # Interpolar as cores ajustadas pela profundidade Z
            r = (alpha_parameter * c0[0] + beta_parameter * c1[0] + gamma_parameter * c2[0]) * interpolated_Z
            g = (alpha_parameter * c0[1] + beta_parameter * c1[1] + gamma_parameter * c2[1]) * interpolated_Z
            b = (alpha_parameter * c0[2] + beta_parameter * c1[2] + gamma_parameter * c2[2]) * interpolated_Z

            return [int(r * 255), int(g * 255), int(b * 255)]
        
        def draw_pixel_with_depth(x, y, Z, rgb):
            # Lê o valor de profundidade atual do Z-buffer para o pixel (x, y)
            current_depth = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)
            
            # Lê a cor atual do pixel (x, y)
            current_color = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
            if Z < current_depth[0]:
                # Atualiza o Z-buffer com o novo valor de profundidade
                gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [Z])

                #Aplica a transparencia
                previous_color = current_color*transparency
                new_color = np.array(rgb)*(1-transparency)

                #Atualiza a cor do pixel com transparencia
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8,(previous_color + new_color).astype(int))

        def z_buffer(z, near, far):
            #Constantes para a fórmula de Z-buffer
            A = -(far + near) / (far - near)
            B = -2 * far * near / (far - near)
            #Converte Z para NDC
            z_ndc = (A * z + B) / (-z)
            #Normaliza Z para o intervalo [0, 1]
            z_normalized = (z_ndc + 1) / 2
            #Converte Z para 32 bits
            z_buffer_32b = round(z_normalized * (2**32 - 1))
            return z_buffer_32b

        def bilinear_interpolation(uv, texture):
            # Extrai as coordenadas u e v
            u, v = uv
            # Extrai as dimensões da textura
            height, width = texture.shape[0], texture.shape[1]
            
            u = 1 - u
            # Normaliza as coordenadas UV para o intervalo [0, 1]
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)

            # Converte as coordenadas UV para coordenadas de textura        
            u = u * (width - 1)
            #Inverte a coordenada v para que a origem seja no canto superior esquerdo e va para baixo
            v = v * (height - 1) 

            # Calcula os índices dos quatro texels vizinhos
            x0, y0 = int(np.floor(u)), int(np.floor(v))
            x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

            # Calcula a razão Horizontal e Vertical para interpolação bilinear
            s_ratio = u - x0
            t_ratio = v - y0

            # Pegar os texels vizinhos
            texel00 = texture[y0, x0]
            texel01 = texture[y1, x0]
            texel10 = texture[y0, x1]
            texel11 = texture[y1, x1]

            # Interpola Horizontalmente
            color_top = texel00 * (1 - s_ratio) + texel10 * s_ratio
            color_bottom = texel01 * (1 - s_ratio) + texel11 * s_ratio
            #Interpola Verticalmente
            final_color = color_top * (1 - t_ratio) + color_bottom * t_ratio

            return final_color.astype(int)
        
        def apply_texture(p, p0, p1,p2,tex_coords, texture,Z):

            #Multiplicador para calculo MipMap  
            multiplier = texture.shape[0]
            
            alpha, beta, gamma = barycentric_coords(p, p0, p1, p2)
            
            # Calcula cordenadas baricentricas corrigidas pela perspectiva
            w0 = alpha / Z[0]
            w1 = beta / Z[1]
            w2 = gamma / Z[2]
            w_sum = w0 + w1 + w2

            alpha_p = w0 / w_sum
            beta_p = w1 / w_sum
            gamma_p = w2 / w_sum

            #Cordenadas UV
            v = alpha_p * tex_coords[0][0] + beta_p * tex_coords[1][0] + gamma_p * tex_coords[2][0]
            u = alpha_p * tex_coords[0][1] + beta_p * tex_coords[1][1] + gamma_p * tex_coords[2][1]

            uv = (u, v)
            
            #Delta para calcular derivadas
            delta = 1

            # Pixels vizinhos
            p_dx = (p[0] + delta, p[1])
            p_dy = (p[0], p[1] + delta)

            # Calcula os parâmetros baricêntricos para os pixels vizinhos
            alpha_dx, beta_dx, gamma_dx = barycentric_coords(p_dx, p0, p1, p2)
            alpha_dy, beta_dy, gamma_dy = barycentric_coords(p_dy, p0, p1, p2)

            # Corrige os parâmetros baricêntricos para a perspectiva
            w0_dx = alpha_dx / Z[0]
            w1_dx = beta_dx / Z[1]
            w2_dx = gamma_dx / Z[2]
            w_sum_dx = w0_dx + w1_dx + w2_dx

            alpha_p_dx = w0_dx / w_sum_dx
            beta_p_dx = w1_dx / w_sum_dx
            gamma_p_dx = w2_dx / w_sum_dx

            w0_dy = alpha_dy / Z[0]
            w1_dy = beta_dy / Z[1]
            w2_dy = gamma_dy / Z[2]
            w_sum_dy = w0_dy + w1_dy + w2_dy

            alpha_p_dy = w0_dy / w_sum_dy
            beta_p_dy = w1_dy / w_sum_dy
            gamma_p_dy = w2_dy / w_sum_dy

            # Calcula as coordenadas UV para os pixels vizinhos
            v_dx = alpha_p_dx * tex_coords[0][0] + beta_p_dx * tex_coords[1][0] + gamma_p_dx * tex_coords[2][0]
            u_dx = alpha_p_dx * tex_coords[0][1] + beta_p_dx * tex_coords[1][1] + gamma_p_dx * tex_coords[2][1]

            v_dy = alpha_p_dy * tex_coords[0][0] + beta_p_dy * tex_coords[1][0] + gamma_p_dy * tex_coords[2][0]
            u_dy = alpha_p_dy * tex_coords[0][1] + beta_p_dy * tex_coords[1][1] + gamma_p_dy * tex_coords[2][1]

            # Calcula as derivadas parciais
            dudx = multiplier*(u_dx - u)/delta
            dvdx = multiplier*(v_dx - v)/delta
            dudy = multiplier*(u_dy - u)/delta
            dvdy = multiplier*(v_dy - v)/delta

            # Calcula o comprimento do gradiente
            L = max(np.sqrt(dudx ** 2 + dvdx ** 2),np.sqrt(dudy ** 2 + dvdy ** 2))

            # Calcula o nível do MipMap
            D = max(0,np.log2(L))
            
            #Seleciona a textura
            mip_level = math.floor(np.clip(D, 0, len(GL.mipmaps) - 1))

            selected_texture = GL.mipmaps[mip_level]
            texture_color = bilinear_interpolation(uv, selected_texture)

            return [texture_color[0], texture_color[1], texture_color[2]]

    
        def draw_triangle(p0, p1, p2, c0, c1, c2):
            # Definir bounding box para o triângulo
            xmin = int(np.floor(min(p0[0], min(p1[0], p2[0]))))
            xmax = int(np.ceil(max(p0[0], max(p1[0], p2[0]))))
            ymin = int(np.floor(min(p0[1], min(p1[1], p2[1]))))
            ymax = int(np.ceil(max(p0[1], max(p1[1], p2[1]))))


            # Limites da tela
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, GL.width-1)
            ymax = min(ymax, GL.height-1)

            # Percorrer todos os pixels no bounding box
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    if inside_triangle(x+0.5, y+0.5, p0, p1, p2):
                        if Z is None:
                            draw_pixel_with_depth(x, y, 0, default_rgb)
                        else:
                            #Pega o ponto atual
                            ponto_atual = (x + 0.5, y + 0.5)
                            #Calcula os parametros da interpolação baricentrica
                            alpha, beta, gamma = barycentric_coords(ponto_atual, p0, p1, p2)
                            #Calcula a profundidade interpolada
                            Z_interpolated = 1 / (alpha / Z[0] + beta / Z[1] + gamma / Z[2])
                            interpolated_rgb = default_rgb
                            if color_per_vertex:
                                # Interpolar a cor do ponto atual usando as cores dos vértices e a profundidade interpolada
                                interpolated_rgb = interpolated_color(alpha/Z[0],beta/Z[1],gamma/Z[2],Z_interpolated ,c0, c1, c2)
                            if texture is not None:
                                # Aplica a textura no pixel atual
                                interpolated_rgb = apply_texture(ponto_atual, p0, p1,p2, tex_coords, texture,Z)
                            #Calcula o valor de profundidade da tela em 32 bits
                            z_32b = z_buffer(Z_interpolated, GL.near, GL.far)
                            draw_pixel_with_depth(x, y, z_32b, interpolated_rgb)

        # Loopa pela lista de vértices e desenha os triângulos
        for i in range(0, len(vertices), 6):
            #Triangulos 3D
            if Z is not None:
                p0 = (vertices[i], vertices[i + 1])
                p1 = (vertices[i + 2], vertices[i + 3])
                p2 = (vertices[i + 4], vertices[i + 5])
            #Triangulos 2D
            else:
                p0 = (vertices[i]*2, vertices[i + 1]*2)
                p1 = (vertices[i + 2]*2, vertices[i + 3]*2)
                p2 = (vertices[i + 4]*2, vertices[i + 5]*2)
            # Cores dos vértices
            if color_per_vertex:
                c0 = color_vertices[0]
                c1 = color_vertices[1]
                c2 = color_vertices[2]
            else:
                c0 = c1 = c2 = default_rgb
            draw_triangle(p0, p1, p2, c0, c1, c2)





    @staticmethod
    def triangleSet(point, colors,color_vertices= None,color_per_vertex = False,text_coords = None,texture = None):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        for i in range(0, len(point),9):
            #Realiza a Matriz dos vértices
            vertices = np.array([[point[i],point[i+3],point[i+6],],
                        [point[i+1],point[i+4],point[i+7],],
                        [point[i+2],point[i+5],point[i+8],],
                        [1,1,1]])

            #Aplica as transformações da pilha
            transformadas = GL.transform

            for i in range(len(GL.transform)-1,-1,-1):
                vertices = np.matmul(transformadas[i], vertices)
                
            # Aplica a matriz de visão nos vértices
            vertices_look_at = np.matmul(GL.view_matrix, vertices)

            p0_look_at = vertices_look_at[:3, 0]  # Extract the first three coordinates (x, y, z)
            p1_look_at = vertices_look_at[:3, 1]
            p2_look_at = vertices_look_at[:3, 2]
            #Calcula a normal dos vertices do triangulo
            v1 = p1_look_at - p0_look_at
            v2 = p2_look_at - p0_look_at
            normal = np.cross(v1, v2)
            norm_value = np.linalg.norm(normal)
            if norm_value != 0:
                normal = normal / norm_value
            else:
                # Handle the case where the normal is a zero vector
                normal = np.zeros_like(normal) 
            camera_forward = np.array([0, 0, 1])

            GL.view_direction = camera_forward
            Z = [vertices_look_at[2][0],vertices_look_at[2][1],vertices_look_at[2][2]]
            # Aplica a matriz de projeção nos vértices
            vertices_NDC = np.matmul(GL.P, vertices_look_at)

            #Normaliza os vertices
            p0 = np.array([vertices_NDC[0][0]/vertices_NDC[3][0],vertices_NDC[1][0]/vertices_NDC[3][0], vertices_NDC[2][0]/vertices_NDC[3][0]])
            p1 = np.array([vertices_NDC[0][1]/vertices_NDC[3][1],vertices_NDC[1][1]/vertices_NDC[3][1], vertices_NDC[2][1]/vertices_NDC[3][1]])
            p2 = np.array([vertices_NDC[0][2]/vertices_NDC[3][2],vertices_NDC[1][2]/vertices_NDC[3][2], vertices_NDC[2][2]/vertices_NDC[3][2]])

            #Aplica a matriz da tela nos vertices
            w = GL.width
            h = GL.height
            tela = np.array([[w/2,0,0,w/2],
                            [0,-h/2,0,h/2],
                            [0,0,1,0],
                            [0,0,0,1]])

            vertices_tela = np.array([[p0[0],p1[0],p2[0]],
                                    [p0[1],p1[1],p2[1]],
                                    [p0[2],p1[2],p2[2]],
                                    [1,1,1]])


            vertices_finais = np.matmul(tela, vertices_tela)

            #Desenha o triangulo
            pontos = np.array([vertices_finais[0][0], vertices_finais[1][0],
                        vertices_finais[0][1], vertices_finais[1][1],
                        vertices_finais[0][2], vertices_finais[1][2]])
            GL.triangleSet2D(pontos, colors, color_vertices, color_per_vertex,Z,text_coords,texture,normal)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        #Extrai o valor da posição
        pos_x, pos_y, pos_z = position

        #Cria a matriz de translação
        translation_matrix = np.array([
            [1, 0, 0, pos_x],
            [0, 1, 0, pos_y],
            [0, 0, 1, pos_z],
            [0, 0, 0, 1]
        ])

        #Extrai o valor da orientação
        axis_x, axis_y, axis_z, angle = orientation

        #Normaliza o vetor de orientação
        norm = np.linalg.norm([axis_x, axis_y, axis_z])
        axis_x /= norm
        axis_y /= norm
        axis_z /= norm

        #Cria a matriz de rotação
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        sin_half_angle = np.sin(half_angle)
        axis_x *= sin_half_angle
        axis_y *= sin_half_angle
        axis_z *= sin_half_angle

        rotation_matrix = np.array([
            [1 - 2*axis_y*axis_y - 2*axis_z*axis_z, 2*axis_x*axis_y - 2*w*axis_z, 2*axis_x*axis_z + 2*w*axis_y, 0],
            [2*axis_x*axis_y + 2*w*axis_z, 1 - 2*axis_x*axis_x - 2*axis_z*axis_z, 2*axis_y*axis_z - 2*w*axis_x, 0],
            [2*axis_x*axis_z - 2*w*axis_y, 2*axis_y*axis_z + 2*w*axis_x, 1 - 2*axis_x*axis_x - 2*axis_y*axis_y, 0],
            [0, 0, 0, 1]
        ])

        #Calcula as inversas
        inv_translation_matrix = inv(translation_matrix)
        inv_rotation_matrix = inv(rotation_matrix)

        #Cria a matriz de visão
        GL.view_matrix =  np.matmul(inv_rotation_matrix, inv_translation_matrix)



        #Calcula a matriz de projeção
        w = GL.width
        h = GL.height
        fovy = 2 * np.arctan(np.tan(fieldOfView / 2) * (h/((w**2 + h**2)**0.5)))

        aspect = w/h
        near = GL.near
        far = GL.far
        top = near * np.tan(fovy)
        right = top * aspect

        GL.P = np.array([  [near/right, 0, 0, 0],
                        [0, near/top, 0, 0],
                        [0, 0, -((far + near)/(far - near)), (-2*far*near)/(far - near)],
                        [0, 0, -1, 0]
                    ])
        

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        if translation:
            #Cria matriz de translacao
            translacao = np.array([[1, 0, 0, translation[0]],
                                    [0, 1, 0, translation[1]],
                                    [0, 0, 1, translation[2]],
                                    [0, 0, 0, 1]])
        if scale:
            #Cria matriz de escala
            escala = np.array([[scale[0], 0, 0, 0],
                               [0, scale[1], 0, 0],
                               [0, 0, scale[2], 0],
                               [0, 0, 0, 1]])
            
        if rotation:
            #Cria matriz de rotação
            x, y, z, theta = rotation
            half_theta = theta / 2.0
            w = np.cos(half_theta)
            sin_half_theta = np.sin(half_theta)
            x *= sin_half_theta
            y *= sin_half_theta
            z *= sin_half_theta
            
            rotacao = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
                [0, 0, 0, 1]
            ])

        # Combinação das transformações: escala, rotação e translação
        transformacao_total = np.matmul(translacao, np.matmul(rotacao, escala))
        # Adiciona a matriz de transformação na pilha
        GL.transform.append(transformacao_total)

        

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
    
        # Recupera a matriz de transformação da pilha
        GL.transform.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors, color_vertices = None, color_per_vertex = False, tex_coords = None, texture = None):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados.
        # O parâmetro 'point' contém uma lista de coordenadas x, y e z dos vértices.
        # 'stripCount' especifica quantos vértices formam cada tira de triângulos.
        
        index = 0  # Índice que rastreia a posição atual na lista de pontos

        for count in stripCount:
            # Para cada faixa de triângulos, percorremos os pontos
            for x in range(count - 2): 
                # Pegando vertices do triangulo
                p0 = (point[index*3], point[index*3 + 1], point[index*3 + 2])
                p1 = (point[(index + 1)*3], point[(index + 1)*3 + 1], point[(index + 1)*3 + 2])
                p2 = (point[(index + 2)*3], point[(index + 2)*3 + 1], point[(index + 2)*3 + 2])

                if x % 2 == 0:
                    # Se a orientação estiver correta, desenha o triângulo
                    GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors, color_vertices, color_per_vertex, tex_coords, texture)
                else:
                    # Se a orientação estiver invertida, troca a ordem dos vértices
                    GL.triangleSet([p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]], colors, color_vertices, color_per_vertex, tex_coords, texture)

                # Avança para o próximo ponto
                index += 1
            
            # Após processar uma faixa de triângulos, avança o índice para o próximo conjunto de vértices
            index += 2  

    def orientation(p0,p1,p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p0)
        
        normal = np.cross(v1, v2)
        norm_value = np.linalg.norm(normal)
    
        if norm_value != 0 and not np.isnan(norm_value):
            normal = normal / norm_value
        else:
            normal = np.array([0, 0, 0])

        d = np.dot(normal, GL.forward_direction)
        return d
            
                
    @staticmethod
    def indexedTriangleStripSet(point, index, colors, color_vertices = None, color_per_vertex = False, tex_coords = None, texture = None):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        current_strip = []  # Lista temporária para armazenar a sequência atual de vértices

        for i in index:
            if i == -1:
                # Quando encontramos -1, processamos a tira atual se ela tiver pelo menos 3 vértices
                if len(current_strip)//3 >= 3:
                    GL.triangleStripSet(current_strip, [len(current_strip)//3], colors,color_vertices, color_per_vertex, tex_coords, texture)

                # Limpar a lista atual para a próxima tira
                current_strip = []
            else:
                # Adiciona o vértice correspondente ao índice atual na tira atual
                current_strip.append(point[i*3])
                current_strip.append(point[i*3 + 1])
                current_strip.append(point[i*3 + 2])


    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.
    
        #Pega os pontos
        coordinate_x = size[0]/2
        coordinate_y = size[1]/2
        coordinate_z = size[2]/2
        # Cria os pontos da caixa
        points = [
            #P0
            -coordinate_x, coordinate_y, -coordinate_z,
            #P1
            -coordinate_x, coordinate_y, coordinate_z,
            #P2
            coordinate_x, coordinate_y, coordinate_z,
            #P3
            coordinate_x, coordinate_y, -coordinate_z,
            #P4
            -coordinate_x, -coordinate_y, -coordinate_z,
            #P5
            -coordinate_x, -coordinate_y, coordinate_z,
            #P6
            coordinate_x, -coordinate_y, coordinate_z,
            #P7
            coordinate_x, -coordinate_y, -coordinate_z

        ]
        #Tessela a caixa
        coordIndex = [
            #Face1
            0,1,2,3,-1,
            #Face2
            0,4,5,1,-1,
            #Face3
            1,5,6,2,-1,
            #Face4
            2,6,7,3,-1,
            #Face5
            3,7,4,0,-1,
            #Face6
            4,7,6,5,-1
        ]
        GL.indexedFaceSet(points, coordIndex, False, None, None, None, None, colors, None)

    def generate_mipmaps(texture):
        mipmaps = []
        mipmaps.append(texture)
        
        # Gera os níveis de mipmaps até que a textura tenha 1x1 pixels
        while texture.shape[1] > 1 and texture.shape[0] > 1:
            # Reduce the size of the texture by half
            texture = cv2.resize(texture, (max(1, texture.shape[1] // 2), max(1, texture.shape[0] // 2)), interpolation=cv2.INTER_LINEAR)
            mipmaps.append(texture)
        
        return mipmaps
    
    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                    texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet com triangulação."""
        image = None

        if not colorPerVertex:
            color = [1, 1, 1]

        if colorPerVertex:
            if not color or not colorIndex:
                colorPerVertex = False
                color = [1, 1, 1]

        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            GL.mipmaps = GL.generate_mipmaps(image)
            if not texCoordIndex:
                texCoordIndex = coordIndex

        current_face_indices = []
        faces = []

        # Coleta as faces a partir de coordIndex
        for index in coordIndex:
            if index == -1:
                if len(current_face_indices) >= 3:
                    # Armazena a face atual para processamento posterior
                    faces.append(current_face_indices.copy())
                current_face_indices.clear()
            else:
                current_face_indices.append(index)

        # Processa quaisquer índices restantes após o loop
        if len(current_face_indices) >= 3:
            faces.append(current_face_indices.copy())

        # Para cada face, triangula e renderiza
        for face_indices in faces:
            # Triangula a face usando um triângulo fan
            for i in range(1, len(face_indices) - 1):
                idx0 = face_indices[0]
                idx1 = face_indices[i]
                idx2 = face_indices[i + 1]

                # Coordenadas dos pontos
                p0 = coord[idx0 * 3: idx0 * 3 + 3]
                p1 = coord[idx1 * 3: idx1 * 3 + 3]
                p2 = coord[idx2 * 3: idx2 * 3 + 3]
                triangle_points = p0 + p1 + p2

                # Cores por vértice
                triangle_colors = None
                if colorPerVertex and color and colorIndex:
                    c0_idx = colorIndex[idx0] * 3
                    c1_idx = colorIndex[idx1] * 3
                    c2_idx = colorIndex[idx2] * 3
                    c0 = [color[c0_idx], color[c0_idx + 1], color[c0_idx + 2]]
                    c1 = [color[c1_idx], color[c1_idx + 1], color[c1_idx + 2]]
                    c2 = [color[c2_idx], color[c2_idx + 1], color[c2_idx + 2]]
                    triangle_colors = [c0, c1, c2]

                # Coordenadas de textura
                triangle_tex_coords = None
                if texCoord and texCoordIndex:
                    t0_idx = texCoordIndex[idx0] * 2
                    t1_idx = texCoordIndex[idx1] * 2
                    t2_idx = texCoordIndex[idx2] * 2
                    t0 = (texCoord[t0_idx], texCoord[t0_idx + 1])
                    t1 = (texCoord[t1_idx], texCoord[t1_idx + 1])
                    t2 = (texCoord[t2_idx], texCoord[t2_idx + 1])
                    triangle_tex_coords = [t0, t1, t2]

                # Chama triangleSet para renderizar o triângulo atual
                GL.triangleSet(triangle_points, colors, triangle_colors, colorPerVertex,
                            triangle_tex_coords, image)



    
    @staticmethod
    def sphere(radius, colors, subdivisions=5):
        # Generate initial icosahedron
        t = (1.0 + np.sqrt(5.0)) / 2.0

        # Initial vertices of an icosahedron
        initial_vertices = [
            [-1,  t,  0],
            [ 1,  t,  0],
            [-1, -t,  0],
            [ 1, -t,  0],
            [ 0, -1,  t],
            [ 0,  1,  t],
            [ 0, -1, -t],
            [ 0,  1, -t],
            [ t,  0, -1],
            [ t,  0,  1],
            [-t,  0, -1],
            [-t,  0,  1],
        ]

        # Normalize initial vertices to lie on the sphere
        initial_vertices = [np.array(v) / np.linalg.norm(v) * radius for v in initial_vertices]

        # Define the 20 faces of the icosahedron with consistent winding order (counter-clockwise)
        initial_faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
            [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
            [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
            [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
        ]

        # Create a dictionary to store middle point indices to avoid duplicates
        middle_point_cache = {}

        # Function to get a middle point between two vertices and normalize it to lie on the sphere
        def get_middle_point(v1_idx, v2_idx):
            # Order the indices to check for existing middle point
            smaller_idx = min(v1_idx, v2_idx)
            greater_idx = max(v1_idx, v2_idx)
            key = (smaller_idx, greater_idx)

            if key in middle_point_cache:
                return middle_point_cache[key]

            # Compute the middle point and normalize it
            point1 = vertices[smaller_idx]
            point2 = vertices[greater_idx]
            middle = (point1 + point2) / 2.0
            middle = middle / np.linalg.norm(middle) * radius  # Normalize to lie on the sphere

            vertices.append(middle)
            index = len(vertices) - 1
            middle_point_cache[key] = index

            return index

        # Start with initial vertices and faces
        vertices = initial_vertices.copy()
        faces = initial_faces.copy()

        # Subdivide faces
        for _ in range(subdivisions):
            faces_subdivided = []
            for tri in faces:
                v1_idx, v2_idx, v3_idx = tri

                # Get middle points for each edge of the triangle
                a = get_middle_point(v1_idx, v2_idx)
                b = get_middle_point(v2_idx, v3_idx)
                c = get_middle_point(v3_idx, v1_idx)

                # Create four new triangles with consistent winding order
                faces_subdivided.append([v1_idx, a, c])
                faces_subdivided.append([a, v2_idx, b])
                faces_subdivided.append([b, v3_idx, c])
                faces_subdivided.append([a, b, c])

            faces = faces_subdivided

        # Prepare the Points list
        Points = []
        for v in vertices:
            Points.extend(v.tolist())

        # Prepare the coordIndex list
        coordIndex = []
        for face in faces:
            coordIndex.extend(face + [-1])  # Add -1 to indicate the end of face

        # Call the indexedFaceSet function to render the icosphere
        GL.indexedFaceSet(
            Points, coordIndex, False, None, None,
            None, None, colors, None
        )




    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        
        # Apex of the cone
        apex = [0, height/2, 0]
        #Pontos
        Points = []
        Points.extend(apex)
        #Pontos da base
        N = 100
        #Lista thetas
        base = np.linspace(0, 2 * np.pi, N,endpoint=False)
        #Cria os pontos da base
        for theta in base:
            x = bottomRadius * np.cos(theta)
            z = bottomRadius * np.sin(theta)
            Points.extend([x, -height/2, z])
        #Centro da base
        Points.extend([0, -height/2, 0])
        #Lista de Indices
        coordIndex = []
        #Tessela o Apex com a base
        for i in range(N):
            next_idx = (i + 1) % N
            coordIndex.extend([0, i + 1, next_idx + 1, -1])

        # Cria a face da base
        base_face_indices = [i + 1 for i in range(N - 1, -1, -1)]
        coordIndex.extend(base_face_indices + [-1])
        #Desenha o cone
        GL.indexedFaceSet(Points, coordIndex, False, None, None, None, None, colors, None)

    @staticmethod
    def cylinder(radius, height, colors):
        """Function to render cylinders."""
        # Cilindro centrado em (0, 0, 0) no sistema de coordenadas local com eixo vertical Y.
        #Divide o cilindro em N segmentos ao redor do raio e M segmentos ao longo da altura.
        N = 100
        M = 100
        
        #Lista Iterações	
        theta_list = np.linspace(0, 2 * np.pi, N, endpoint=False)
        y_list = np.linspace(-height / 2, height / 2, M)

        # Lista Pontos
        Points = []

        # Cria os pontos do cilindro
        for y in y_list:
            for theta in theta_list:
                x = radius * np.cos(theta)
                z = radius * np.sin(theta)
                Points.extend([x, y, z])  

        #Lista de Indices
        coordIndex = []

        # Tessela o lado do cilindro
        for i in range(M - 1):
            for j in range(N):
                #Pontos da Face
                next_j = (j + 1) % N
                current_idx = i * N + j
                next_idx = i * N + next_j
                upper_current_idx = (i + 1) * N + j
                upper_next_idx = (i + 1) * N + next_j
                #Organiza Face 
                coordIndex.extend([upper_current_idx, current_idx, next_idx, upper_next_idx , -1])

        # Tessela a base inferior (em Y = -height/2)
        base_face_indices = [i for i in range(N - 1, -1, -1)]
        coordIndex.extend(base_face_indices + [-1])

        # Tessela a base superior (em Y = -height/2)
        comeco_base_superior = (M - 1) * N
        fim_base_superior = M*N
        base_face_indices = [i for i in range(comeco_base_superior, fim_base_superior)]
        coordIndex.extend(base_face_indices + [-1])

        GL.indexedFaceSet(Points, coordIndex, False, None, None, None, None, colors, None)


    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        if headlight:
            GL.directional_light = [0.0, [1, 1, 1], 1]

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.
        GL.directional_light = [ambientIntensity, color, intensity, direction]

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed
        
        
        gpu.GPU.clear_buffer()
        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed


    hermite_matrix = np.array([
        [2, -2, 1, 1],
        [-3, 3, -2, -1],
        [0, 0, 1, 0],
        [1, 0, 0, 0]
        ])
    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.
        # Find the segment in which the interpolation occurs
        #Encontra o segmento em que a interpolação ocorre
        #Caso a fração seja menor que o primeiro valor
        if set_fraction <= key[0]:
            return keyValue[:3]
        #Caso a fração seja maior que o último valor
        elif set_fraction >= key[-1]:
            return keyValue[-3:]
        #Encontra o segmento em que a interpolação ocorre
        index = np.searchsorted(key, set_fraction) - 1
        #Calcula a fração de interpolação local ou seja porcentagem do segmento a ser interpolada
        t = (set_fraction - key[index]) / (key[index + 1] - key[index])
        
        p1 = np.array(keyValue[3 * index : 3 * index + 3])
        p2 = np.array(keyValue[3 * (index + 1) : 3 * (index + 1) + 3])

        #Calculo das tangentes
        #Caso seja o primeiro ponto
        if index == 0:
            #Se for fechado p0 é o último ponto
            if closed:
                p0 = np.array(keyValue[-3:])
            else:
                #Se não for fechado p0 é o mesmo que p1
                p0 = p1
        #Caso não seja o primeiro ponto
        else:
            p0 = np.array(keyValue[3 * (index - 1) : 3 * (index - 1) + 3])
        #Caso seja o último ponto
        if index + 2 == len(key):
            #Se for fechado p3 é o primeiro ponto
            if closed:
                p3 = np.array(keyValue[:3])
            else:
                #Se não for fechado p3 é o mesmo que p2
                p3 = p2
        #Caso não seja o último ponto
        else:
            p3 = np.array(keyValue[3 * (index + 2) : 3 * (index +2) + 3])
        
        #Calcula a interpolação cúbica de Hermite
        h0 = p1
        h1 = p2
        h2 = (p2-p0)*0.5
        h3 = (p3-p1)*0.5

        T = np.array([t**3, t**2, t, 1])
        H = np.array([h0, h1, h2, h3])
        #Calcula o valor interpolado
        value_changed = np.matmul(T, np.matmul(GL.hermite_matrix, H))
        return value_changed.tolist()


    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Optimized interpolation of rotation values using slerp."""
        
        # Cria uma lista de rotações a partir dos valores de keyValue
        rotations = np.reshape(keyValue, (-1, 4))
        
        # Checa condições de borda
        if set_fraction <= key[0]:
            return rotations[0].tolist()
        elif set_fraction >= key[-1]:
            return rotations[-1].tolist()
        
        # Acha o segmento em que a interpolação ocorre
        i = np.searchsorted(key, set_fraction) - 1
        
        # Calcula a fração de interpolação local
        t = (set_fraction - key[i]) / (key[i+1] - key[i])
        
        # Pega os valores de rotação (eixo e ângulo) para q0 e q1
        axis0, angle0 = rotations[i, :3], rotations[i, 3]
        axis1, angle1 = rotations[i+1, :3], rotations[i+1, 3]
        
        # Normaliza os eixos de q0 e q1
        axis0_normalized = axis0 / np.linalg.norm(axis0) if np.linalg.norm(axis0) != 0 else axis0
        axis1_normalized = axis1 / np.linalg.norm(axis1) if np.linalg.norm(axis1) != 0 else axis1
        
        # Converte os ângulos para meio-ângulo
        half_angle0 = angle0 / 2
        half_angle1 = angle1 / 2
        
        # Constrói q0 e q1 a partir dos valores normalizados e ângulos
        q0 = np.hstack((np.cos(half_angle0), axis0_normalized * np.sin(half_angle0)))
        q1 = np.hstack((np.cos(half_angle1), axis1_normalized * np.sin(half_angle1)))
        
        # Calcula o produto escalar entre q0 e q1
        dot = np.dot(q0, q1)
        
        # Se o produto escalar for negativo, inverte q1 para pegar o caminho mais curto
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        
        # Usa clamp para garantir estabilidade
        dot = np.clip(dot, -1.0, 1.0)
        
        # Calcula o ângulo entre q0 e q1
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        
        # Aplica SLERP
        if sin_theta_0 < 1e-6:
            q_interp = (1.0 - t) * q0 + t * q1
        else:
            sin_theta = np.sin((1.0 - t) * theta_0)
            sin_theta_t = np.sin(t * theta_0)
            q_interp = (sin_theta / sin_theta_0) * q0 + (sin_theta_t / sin_theta_0) * q1
        
        # Normaliza o quaternion interpolado
        q_interp /= np.linalg.norm(q_interp)
        
        # Converte o quaternion interpolado de volta para eixo e ângulo
        w, x, y, z = q_interp
        angle = 2.0 * np.arccos(w)
        sin_half_angle = np.sqrt(1.0 - w * w)
        
        if sin_half_angle < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])  
        else:
            axis = np.array([x, y, z]) / sin_half_angle
        
        value_changed = [axis[0], axis[1], axis[2], angle]
        return value_changed




    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
