import numpy as np
import soundfile as sf
import sys
import librosa 


def phase_vocoder(input_signal: np.ndarray, stretch_factor: float) ->  np.ndarray:
    """
    input_signal - Входной аудиосигнал.
    stretch_factor - Коэффициент растяжения/сжатия звука.
    
    Возвращает измененный аудиосигнал по фазе.
    """
    # Выбран оптимальный размер окна 256. С таким параметром достигается наилучшее качество выходящего аудиосигнала.
    # В соответствии с алгоритмом перекрытие составляет 75% => длина сдвига будет 256 / 4

    window_size = 256
    hop_size = 64


    # Разделение входного сигнала на перекрывающиеся фрагменты
    frames = []
    for i in range(0, len(input_signal) - window_size, hop_size):
        frame = input_signal[i:i + window_size]
        frames.append(frame)

    # Применение быстрого преобразования Фурье к каждому фрагменту
    spectra = [librosa.stft(frame) for frame in frames]

    # Создание нового массива для выходного сигнала
    output_signal = np.zeros(int(len(input_signal) * stretch_factor))

    # Интерполяция Фурье-преобразований и создание выходного сигнала
    for i in range(len(spectra) - 1):

        # Вычисляем параметр alpha для интерполяции между текущим и следующим спектрами
        alpha = (i * hop_size) * stretch_factor % 1

        # Производим интерполяцию между двумя спектрами с учетом параметра alpha
        spectra_interp = (1 - alpha) * spectra[i] + alpha * spectra[i + 1]

        # Получаем фрейм выходного сигнала путем обратного преобразования Фурье
        output_frame = np.real(np.fft.ifft(spectra_interp))

        # Определяем начало и конец области в выходном сигнале
        output_start = int(i * hop_size * stretch_factor)
        output_end = min(output_start + window_size, len(output_signal))

        # Полученный фрейм позиционируется в выходном сигнале output_signal с учетом растяжения и размера окна.
        output_signal[output_start:output_end] += output_frame[:output_end - output_start]

    return output_signal



def main():

    # Переданные аргументы
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    stretch_factor = float(sys.argv[3])

    # input_file = 'input.wav'
    # output_file = 'result.wav'
    # stretch_factor = 0.5

    # Загрузка входного аудиофайла
    input_audio, fs = sf.read(input_file)
    # Применение алгоритма фазового вокодера
    output_audio = phase_vocoder(input_audio, stretch_factor)

    # Сохранение результирующего аудиофайла
    sf.write(output_file, output_audio, fs)



if __name__ == "__main__":
    main()