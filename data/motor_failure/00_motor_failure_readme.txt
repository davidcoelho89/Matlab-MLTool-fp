--------SHORT CIRCUIT DATA BASE-----------

Este banco de dados é formado por um conjunto de vetores que representam as
harmônicas das correntes do estator de um motor de indução trifásico gaiola 
de esquilo, sujeito a diversos niveis de curto-cirtuito.

-------------data_ESPEC_1-----------------

- 6 attributes (harmonics - 0,5 1,5 2,5 3 5 7)

- 294 samples (42 - normal motors / 252 - faulty motors)
-    42 samples per class
	3 niveis de carga (0% 50% 100%)
	2 fases (fase em contato direto / fase em contato indireto com falha)
	7 velocidades conversor (30; 35; 40; 45; 50; 55; 60 Hz)

- 7 classes (0 - SF / 1 - A1 / 2 - A2 / 3 - A3
            4 - B1 / 5 - B2 / 6 - B3)

-------------data_ESPEC_2-----------------

- 6 attributes (harmonics - 0,5 1,5 2,5 3 5 7)
- 392 samples (56 - normal motors / 336 - faulty motors)
- 56 samples per class (SF, A1, A2, A3, B1, B2, B3)

-------------data_ESPEC_3-----------------

- 16 harmonicas (0,5 à 8)
- 56 amostras de cada classe (SF, A1, A2, A3, B1, B2, B3)
- 392 amostras no total

-------------data_ESPEC_4-----------------

- 16 harmonicas (0,5 à 8)
- 42 amostras de cada classe (SF, A1, A2, A3, B1, B2, B3)
- 294 amostras no total

-------------data_ESPEC_5-----------------

- 6 harmonicas (0,5 1,5 2,5 3 5 7)
- 252 dados normais 42 reais e 210 gerados
- 252 dados de falha - 42 por classe (A1, A2, A3, B1, B2, B3)
- 504 amostras no total (42 x 252)

-------------data_ESPEC_6-----------------

Cada vetor possui 11 atributos. 7 deles são utilizáveis para
o treinamento e teste, e 4 são para definir o vetor. São eles:

1 - Harmônica fundamental
2 - 2a Harmônica
3 - 3a Harmônica
4 - 5a Harmônica
5 - 7a Harmônica
6 - 0,5*Harmônica fundamental
7 - 1,5*Harmônica fundamental

8 - tipo de falha: 0 (sem falha), 1 (falha de alta impedancia), 2 (falha de baixa impedancia)
9 - Gravidade da falha (de acordo com o número de espiras em curto: (0,1,2,3)
10 - Frequência da tensão do conversor de frequência: (30, 35, 40, 45, 50, 55, 60)
11 - Porcentagem de carga aplicada: 0, 50, 100


------------------------------------------