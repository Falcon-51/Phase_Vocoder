import numpy as np
import soundfile as sf
import sys


def phase_vocoder(input_signal: np.ndarray, time_stretch_ratio: float) ->  np.ndarray:
    """
    input_signal : np.ndarray
            Входной одноканальный аудиосигнал;
    time_stretch_ratio : float
            Коэффициент растяжения/сжатия звука;
            При time_stretch_ratio > 1 аудиофайл растягивается, при 0 < time_stretch_ratio <= 1 сжимается
    Возвращает измененный аудиосигнал по фазе.
    """
    # Размер окна 1024.
    # В соответствии с алгоритмом перекрытие составляет 75% => длина сдвига будет 1024 / 4

    window_size = 1024
    hop_size = 256


    # Разделение входного сигнала на перекрывающиеся фрагменты
    frames = []
    for i in range(0, len(input_signal) - window_size, hop_size):
        frame = input_signal[i:i + window_size]
        frames.append(frame)

    # Применение оконной функции к каждому кадру
    window = np.hanning(window_size)
    frames *= window

    # Применение быстрого преобразования Фурье к каждому фрагменту
    spectra = [np.fft.fft(frame) for frame in frames]

    # Создание нового массива для выходного сигнала
    output_signal = np.zeros(int(len(input_signal) * time_stretch_ratio))

    # Интерполяция Фурье-преобразований и создание выходного сигнала
    for i in range(len(spectra) - 1):

        # Вычисляем параметр alpha для интерполяции между текущим и следующим спектрами
        alpha = (i * hop_size) * time_stretch_ratio % 1

        # Производим интерполяцию между двумя спектрами с учетом параметра alpha
        spectra_interp = (1 - alpha) * spectra[i] + alpha * spectra[i + 1]

        # Получаем фрейм выходного сигнала путем обратного преобразования Фурье
        output_frame = np.real(np.fft.ifft(spectra_interp))

        # Определяем начало и конец области в выходном сигнале
        output_start = int(i * hop_size * time_stretch_ratio)
        output_end = min(output_start + window_size, len(output_signal))

        # Полученный фрейм позиционируется в выходном сигнале output_signal с учетом растяжения и размера окна.
        output_signal[output_start:output_end] += output_frame[:output_end - output_start] 

    return output_signal



def main():

    # Переданные аргументы
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    time_stretch_ratio = float(sys.argv[3])

    # Загрузка входного аудиофайла
    input_audio, fs = sf.read(input_file)
    # Применение алгоритма фазового вокодера
    output_audio = phase_vocoder(input_audio, time_stretch_ratio)

    # Сохранение результирующего аудиофайла
    sf.write(output_file, output_audio, fs)



if __name__ == "__main__":
    main()