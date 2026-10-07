# 2048_Game_Bot

Python bot for the game 2048

## Sobre

Antes de tudo, você precisa conhecer o jogo. Então, caso você nunca tenha jogado 2048 e está aqui somente procurando um novo projeto em Python para programar, entre em [http://2048game.com/](http://2048game.com/) e jogue um pouco. Entenda as regras do jogo e depois podemos começar o nosso bot.

Esse repositório foi feito para que você possa programar o seu bot em Python sem muita dificuldade, com um material em português e aprendendo o que cada passo determina dentro do corpo do programa que está sendo escrito.

Para escrever esse bot, eu utilizei como referência os vídeos do The PC Geek. Estão todos em inglês (e com alguns erros), por isso quis colocar esse manualzinho aqui. De qualquer forma, se você quiser dar uma olhada nos vídeos, visite o canal do The PC Geek: [https://www.youtube.com/channel/UCu2tUcvNeknPxRN6jhyk6gA](https://www.youtube.com/channel/UCu2tUcvNeknPxRN6jhyk6gA)

Para começarmos, vamos precisar de duas bibliotecas bastante comuns de Python. Reserve um pouco do seu tempo para ler a documentaçao dessas libs - você vai me agradecer mais tarde. Por estarmos fazendo um bot do zero, sem nenhum tipo de lib ou interface voltada especificamente para a criação de bots, a leitura da documentação é essencial. Além de te ajudar com esse projeto aqui, tenho certeza que a leitura te dará várias ideias para projetos futuros.

Bibliotecas:

- PIL ou Pillow
- Pyautogui

## PIL

Quando eu disse que ler a documentação te dará várias ideias de projetos, eu pensava principalmente no PIL. A Python Imaging Library é uma lib totalmente voltada para manipulação de imagens. Aqui nesse projeto, utilizaremos somente as funções de captura de tela, de conversão de imagem para preto e branco (grayscale) e a captação do valor de cor de um pixel. Entretanto, essa lib conta com todo tipo de função para trabalhar com imagens que você imagina, o que abre um leque amplo de possibilidades de projetos, como pequenos apps para aplicar filtros em imagens ou simplesmente redimensionadores e conversores de formato de imagem.

## Pyautogui

O PyAutoGui vai te dar o controle que você precisa sem consumir tanta memória quanto um framework de testes como o Selenium, por exemplo. Como o objetivo aqui é ter acesso a alguns comandos simples de teclado, ele é a biblioteca ideal. Assim como o PIL, também tem uma documentação muito boa que deve ser lida por todo dev.

## Como rodar

```bash
pip install -r requirements.txt
python main.py
```

Então posicione o jogo 2048 no navegador e o bot vai jogar sozinho!

---

⚠️ **Nota histórica**: Esta é a **v1.0.0-legacy** - meu primeiro projeto "sério" em Python, escrito quando estava aprendendo a programar. O código usa screenshot + OCR de pixels para detectar o estado do jogo e envia teclas via PyAutoGUI.

A versão moderna (v2.0.0) está na branch `main` com engine puro, múltiplos algoritmos de IA, benchmarking estatístico, dashboard e testes.