"""Biblioteca compartilhada do projeto de credit scoring: dados, features, modelo, avaliação e serving.

Todas as tarefas do job em src/jobs importam daqui, para que exatamente o mesmo pré-processamento
rode nos testes locais, no treino, na inferência em lote e por trás do endpoint em tempo real.
As decisões vêm da fase de experimentação (notebooks/exploration).
"""
