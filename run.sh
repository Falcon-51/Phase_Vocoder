#!/bin/bash

# Проверка количества переданных аргументов
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input.wav> <output.wav> <time_stretch_ratio>"
    exit 1
fi

# Передача аргументов в Python скрипт
python3 phase_vocoder.py "$1" "$2" "$3"
