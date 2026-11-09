# Roteiro do vídeo de demonstração (gerado por scripts/07_gerar_video_demo.py)

A narração abaixo foi sintetizada com a voz pt-BR do Windows. A equipe pode regravá-la com a própria voz.

## 1. abertura

Olá! Este é o VigiaVia, o componente de visão computacional da nossa CitizenTech para a Global Solution da FIAP. Integrantes: Deivison Pertel, Eduardo Akira Murata e Wesley Souza de Oliveira.

## 2. problema

Buracos causam quedas de motociclistas e ciclistas, danos a veículos e atrasos no transporte. Só em São Paulo foram 113 mil pedidos de tapa-buraco em 2025, e a própria prefeitura reconhece que um mesmo buraco gera vários chamados. Hoje, tudo depende de alguém parar e registrar manualmente.

## 3. solucao

O VigiaVia detecta buracos em tempo real, com a câmera do celular ou de ônibus e caminhões de coleta. Treinamos um detector YOLO26 com o dataset internacional RDD2022, rastreamos cada buraco (BoT-SORT) para gerar um único relato, estimamos a severidade e enviamos só um recorte da via e as coordenadas para a fila da zeladoria.

## 4. dados

Antes de treinar, auditamos os rótulos: numa cópia do dataset, faixas apagadas e tampas de bueiro estavam marcadas como buraco. Usamos a versão oficial (4.192 imagens de cinco países).

> Revisão de 09/11/2026: a frase "(de cinco países) e separamos treino e teste por trajeto, sem vazamento" foi cortada do
> vídeo (de 1:07,4 a 1:13,2 da versão de 04/11; a fala termina em "…com 4.192 imagens"). A divisão por blocos de números
> de arquivo não separou trajetos: veja a seção 5.3 do relatório e `scripts/03b_cenas_repetidas.py`.

## 5. demo_sp

Vídeo real gravado na Rua Sílvia, Bela Vista, São Paulo. O modelo nunca viu este vídeo, nem imagens do Brasil, durante o treino. Cada caixa mostra a confiança e a severidade estimada; quando o buraco entra na trajetória, a interface de realidade aumentada dispara o alerta.

## 6. demo_outros

Em outra cidade e com outra câmera, o rastreamento mantém o mesmo identificador para cada buraco ao longo dos quadros, e o contador soma apenas buracos únicos. Rostos são borrados automaticamente.

## 7. resultados

No teste (622 imagens nunca vistas): mAP@0,5 de 58%, precisão de 70% e revocação de 50% no limiar escolhido. Comparamos três modelos e escolhemos o mais leve dentro da margem de precisão: 62 FPS na GPU e 39 FPS na CPU.

## 8. exemplos

À esquerda, acertos. À direita, buracos que o modelo perdeu: em geral pequenos, distantes ou com sombra. Por isso a detecção é uma sugestão: uma pessoa da zeladoria valida antes da ordem de serviço.

## 9. vieses

Medimos o desempenho por país de origem e por iluminação. Saber onde o modelo erra orienta a próxima etapa: coletar imagens brasileiras pelo próprio aplicativo e retreinar.

## 10. integracao

A visão está integrada ao MVP por uma API. O cidadão envia uma foto ou usa a câmera ao vivo, inclusive com o modelo rodando no próprio celular, sem internet. Relatos a menos de 15 m viram confirmações do mesmo buraco, e a zeladoria vê uma fila priorizada, com mapa.

## 11. encerramento

VigiaVia: ruas mais seguras com visão computacional responsável, conectada aos ODS 11, 9 e 3. Código completo no nosso GitHub. Obrigado!
