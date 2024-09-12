#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from numpy.linalg import inv
class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    view_matrix = np.identity(4)
    P = np.identity(4)
    forward_direction = np.array([0,0,1])
    transform = [np.identity(4)]
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
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
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
            x = int(point[i])
            y = int(point[i + 1])
            # Desenha o ponto
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
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
            p0 = (int(lineSegments[i]),int(lineSegments[i + 1]))
            p1 = (int(lineSegments[i + 2]),int(lineSegments[i + 3]))
            desenha_linha(p0,p1,rgb)

    @staticmethod
    def circle2D(radius, colors):
        emissiveColor = colors.get('emissiveColor')
        rgb = [int(emissiveColor[0] * 255), int(emissiveColor[1] * 255),int(emissiveColor[2] * 255)]

        #Referencia: https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
        x = 0
        y = radius
        d = 3 - 2 * radius  # Initial decision parameter

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
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        # Pega a cor emissiva do dicionario
        emissiveColor = colors.get('emissiveColor')
        diffuseColor = colors.get('diffuseColor')
        if emissiveColor == [0,0,0]:
            r = int(diffuseColor[0] * 255)
            g = int(diffuseColor[1] * 255)
            b = int(diffuseColor[2] * 255)
        else:
            r = int(emissiveColor[0] * 255)
            g = int(emissiveColor[1] * 255)
            b = int(emissiveColor[2] * 255)
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        #Associa a cor emissiva para valores RGB
        rgb = [r,g,b]

        def sign(x, y, p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            return ((x - x0) * (y1 - y0)) - ((y - y0) * (x1 - x0))
        
        def inside_triangle(x, y, p0, p1, p2):
            L0 = sign(x, y, p0, p1)
            L1 = sign(x, y, p1, p2)
            L2 = sign(x, y, p2, p0)
            return (L0 >= 0 and  L1>=0 and L2 >=0)

    
        def desenha_triangulo(p0, p1, p2, rgb):
            #Create bounding box for the triangle points and screen bounds
            xmin = int(np.floor(min(p0[0], p1[0], p2[0])))
            xmax = int(np.ceil(max(p0[0], p1[0], p2[0])))
            ymin = int(np.floor(min(p0[1], p1[1], p2[1])))
            ymax = int(np.ceil(max(p0[1], p1[1], p2[1])))
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > GL.width:
                xmax = GL.width
            if ymax > GL.height:
                ymax = GL.height
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    if inside_triangle(x+0.5, y+0.5, p0, p1, p2):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)
        # Loopa pela lista de pontos e desenha eles
        for i in range(0, len(vertices),6):
            p0 = (vertices[i],vertices[i + 1])
            p1 = (vertices[i+2],vertices[i + 3])
            p2 = (vertices[i+4],vertices[i + 5])
            desenha_triangulo(p0,p1,p2,rgb)



    @staticmethod
    def triangleSet(point, colors, colorPerVertex=False, color=None):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
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

            # Aplica a matriz de projeção nos vértices
            vertices_NDC = np.matmul(GL.P, vertices_look_at)

            #Normaliza os vertices
            p0 = np.array([vertices_NDC[0][0]/vertices_NDC[3][0],vertices_NDC[1][0]/vertices_NDC[3][0], vertices_NDC[2][0]/vertices_NDC[3][0]])
            p1 = np.array([vertices_NDC[0][1]/vertices_NDC[3][1],vertices_NDC[1][1]/vertices_NDC[3][1], vertices_NDC[2][1]/vertices_NDC[3][1]])
            p2 = np.array([vertices_NDC[0][2]/vertices_NDC[3][2],vertices_NDC[1][2]/vertices_NDC[3][2], vertices_NDC[2][2]/vertices_NDC[3][2]])

            #Aplica a matriz da tela nos vertices
            w = 300
            h = 200
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
            GL.triangleSet2D(pontos, colors)

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
        w = 300
        h = 200
        fovy = 2 * np.arctan(np.tan(fieldOfView / 2) * (h/((w**2 + h**2)**0.5)))

        aspect = w/h
        near = 0.01
        far = 1000
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
        
        print(transformacao_total)
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
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados.
        # O parâmetro 'point' contém uma lista de coordenadas x, y e z dos vértices.
        # 'stripCount' especifica quantos vértices formam cada tira de triângulos.
        
        index = 0  # Índice que rastreia a posição atual na lista de pontos

        for count in stripCount:
            # Para cada faixa de triângulos, percorremos os pontos
            for x in range(count - 2):  # Garantimos que existem pelo menos 3 vértices por faixa
                # Pegando três vértices consecutivos
                p0 = (point[index*3], point[index*3 + 1], point[index*3 + 2])
                p1 = (point[(index + 1)*3], point[(index + 1)*3 + 1], point[(index + 1)*3 + 2])
                p2 = (point[(index + 2)*3], point[(index + 2)*3 + 1], point[(index + 2)*3 + 2])

                orientation = GL.orientation(p0,p1,p2)

                if orientation > 0:
                    # Se a orientação estiver correta, desenha o triângulo
                    GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors)
                else:
                    # Se a orientação estiver invertida, troca a ordem dos vértices
                    GL.triangleSet([p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]], colors)

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
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
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
                if len(current_strip) >= 3:
                    for x in range(len(current_strip) - 2):
                        p0 = current_strip[x]
                        p1 = current_strip[x + 1]
                        p2 = current_strip[x + 2]

                        # Verificar a orientação do triângulo (sentido horário/anti-horário)
                        orientation = GL.orientation(p0, p1, p2)
                        if orientation > 0:
                            GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors)
                        else:
                            GL.triangleSet([p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]], colors)

                # Limpar a lista atual para a próxima tira
                current_strip = []
            else:
                # Adiciona o vértice correspondente ao índice atual na tira corrente
                current_strip.append((point[i*3], point[i*3 + 1], point[i*3 + 2]))

        # Caso o último grupo de vértices não seja seguido por um -1
        if len(current_strip) >= 3:
            for x in range(len(current_strip) - 2):
                p0 = current_strip[x]
                p1 = current_strip[x + 1]
                p2 = current_strip[x + 2]

                # Verificar a orientação do triângulo (sentido horário/anti-horário)
                orientation = GL.orientation(p0, p1, p2)
                if orientation > 0:
                    GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors)
                else:
                    GL.triangleSet([p0[0], p0[1], p0[2], p2[0], p2[1], p2[2], p1[0], p1[1], p1[2]], colors)


    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        current_face = [] 
        for index in coordIndex:
            if index == -1:
                if len(current_face) >= 3:
                    for x in range(1, len(current_face) - 1):
                        p0 = current_face[0]
                        p1 = current_face[x]
                        p2 = current_face[x + 1]

                        orientation = GL.orientation(p0,p1,p2)
                        if orientation > 0:
                            # Se a orientação estiver correta, desenha o triângulo
                            GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors,colorPerVertex,color)
                        else:
                            # Se a orientação estiver invertida, troca a ordem dos vértices
                            GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors,colorPerVertex,color)

                # Limpa a face atual para começar a próxima
                current_face = []
            else:
                # Adiciona o vértice correspondente ao índice atual na face corrente
                current_face.append((coord[index*3], coord[index*3 + 1], coord[index*3 + 2]))

        # Caso o último grupo de vértices não seja seguido por um -1
        if len(current_face) >= 3:
            for x in range(1, len(current_face) - 1):
                p0 = current_face[0]
                p1 = current_face[x]
                p2 = current_face[x + 1]

                orientation = GL.orientation(p0,p1,p2)
                if orientation > 0:
                    # Se a orientação estiver correta, desenha o triângulo
                    GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors,colorPerVertex,color)
                else:
                    # Se a orientação estiver invertida, troca a ordem dos vértices
                    GL.triangleSet([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]], colors,colorPerVertex,color)

        

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
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
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
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
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
