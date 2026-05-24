import numpy as np
import csv
from config import block_params


class GMSKDetector:
    def __init__(self, params, block_params):
        self.BT = params.get("BT", 0.3)
        self.T = params.get("T", 3.69e-6)
        self.sps = params.get("sps", 4)
        self.h = params.get("h", 0.5)
        self.gaus_duration = params.get("gaus_duration", 4)
        self.rect_duration = params.get("rect_duration", 1)
        self.type_demod = params.get("type_demod", "diff_phase") # diff_phase / vit_hard / vit_soft 

        self.mf_is_working = block_params["matched_filter"]["is_working"]

        self.counter = 0
    
    def calc_new_rhh(self, h):
        
        E_h = np.sum(np.abs(h**2))
        h_mf = np.conj(h[::-1] / np.sqrt(E_h))

        if self.mf_is_working == False:
            center_idx = h.size // 2
            rhh = h[center_idx::self.sps]
        else:
            rhh_full = np.convolve(h, h_mf)
            center_idx = h_mf.size - 1
            rhh = rhh_full[center_idx :: self.sps]

        return rhh

    def calc_old_rhh(self, h):
        # Раньше в СФ и в модуляторе была нормирвоаннная ИХ
        E_h = np.sum(np.abs(h**2))
        h_mf = h / np.sqrt(E_h)
        h = h_mf

        # Расчет инкрементов - ИХ на выходе СФ x(t). Используется для определения влияния соседних символов
        # Если СФ выключен, то инкременты не используем
        if self.mf_is_working == False:
            center_idx = h.size // 2
            rhh = h[center_idx::self.sps]
        else:
            rhh_full = np.convolve(h, np.conj(h[::-1]))
            center_idx = h.size - 1
            rhh = rhh_full[center_idx :: self.sps]

        return rhh

    # Определяем влияние предыдущих бит для каждого состояния
    def calc_new_increment(self, rhh):

        increment = np.zeros(16)
        increment[0] = rhh[4] + rhh[3] + rhh[2] + rhh[1]
        increment[1] = rhh[4] + rhh[3] + rhh[2] - rhh[1]
        increment[2] = rhh[4] + rhh[3] - rhh[2] + rhh[1]
        increment[3] = rhh[4] + rhh[3] - rhh[2] - rhh[1]
        increment[4] = rhh[4] - rhh[3] + rhh[2] + rhh[1]
        increment[5] = rhh[4] - rhh[3] + rhh[2] - rhh[1]
        increment[6] = rhh[4] - rhh[3] - rhh[2] + rhh[1]
        increment[7] = rhh[4] - rhh[3] - rhh[2] - rhh[1]
        increment[8] = - increment[7]
        increment[9] = - increment[6]
        increment[10] = - increment[5]
        increment[11] = - increment[4]
        increment[12] = - increment[3]
        increment[13] = - increment[2]
        increment[14] = - increment[1]
        increment[15] = - increment[0]

        return increment

    def calc_old_increment(self, rhh):
        # С учетом деротации:
        # [+Im(s), -Re(s), -Im(s), +Re(s)] = [s*j^(-1), s*j^(-2), s*j^(-3), s*j^(-4)]

        increment = np.zeros(16)
        increment[0] = rhh[4].real - rhh[3].imag - rhh[2].real + rhh[1].imag
        increment[1] = rhh[4].real - rhh[3].imag - rhh[2].real - rhh[1].imag
        increment[2] = rhh[4].real - rhh[3].imag + rhh[2].real + rhh[1].imag
        increment[3] = rhh[4].real - rhh[3].imag + rhh[2].real - rhh[1].imag
        increment[4] = rhh[4].real + rhh[3].imag - rhh[2].real + rhh[1].imag
        increment[5] = rhh[4].real + rhh[3].imag - rhh[2].real - rhh[1].imag
        increment[6] = rhh[4].real + rhh[3].imag + rhh[2].real + rhh[1].imag
        increment[7] = rhh[4].real + rhh[3].imag + rhh[2].real - rhh[1].imag
        increment[8] = - increment[7]
        increment[9] = - increment[6]
        increment[10] = - increment[5]
        increment[11] = - increment[4]
        increment[12] = - increment[3]
        increment[13] = - increment[2]
        increment[14] = - increment[1]
        increment[15] = - increment[0]

        # increment = np.zeros(16)
        # increment[0] = rhh[4] + rhh[3] + rhh[2] + rhh[1]
        # increment[1] = rhh[4] + rhh[3] + rhh[2] - rhh[1]
        # increment[2] = rhh[4] + rhh[3] - rhh[2] + rhh[1]
        # increment[3] = rhh[4] + rhh[3] - rhh[2] - rhh[1]
        # increment[4] = rhh[4] - rhh[3] + rhh[2] + rhh[1]
        # increment[5] = rhh[4] - rhh[3] + rhh[2] - rhh[1]
        # increment[6] = rhh[4] - rhh[3] - rhh[2] + rhh[1]
        # increment[7] = rhh[4] - rhh[3] - rhh[2] - rhh[1]
        # increment[8] = - increment[7]
        # increment[9] = - increment[6]
        # increment[10] = - increment[5]
        # increment[11] = - increment[4]
        # increment[12] = - increment[3]
        # increment[13] = - increment[2]
        # increment[14] = - increment[1]
        # increment[15] = - increment[0]


        return increment
    
    # Расчет метрик для всех возможных состояний
    def calc_metric(self, increment, sampled_signal, start_state):
        is_to_table  = False

        if is_to_table:
            if block_params['channel']['is_working']:
                ch = str(int(self.ebn0 - 116))
            else:
                ch = 'no'

            if ((self.counter % 16) == 0):

                    name = 'res_' + ch + '_ch_old_rhh.csv'
            else:
                    name = 'res_' + ch + '_ch_new_rhh.csv'

        # Инициализируем начальное состояние решетки
        old_path_metrics = np.ones(16) * -1e30
        old_path_metrics[start_state] = 0.0
        new_path_metrics = np.zeros(16)

        total_symbols = sampled_signal.size
        trans_table = np.zeros((total_symbols, 16))
        symbol_num = 0

        # Знак для деротации
        sign_rotate = 1
        
        data_to_save = []
        
        while symbol_num < total_symbols:
            
            # Деротация
            if (symbol_num % 2) == 0:
                input_symbol =  sign_rotate * sampled_signal[symbol_num].imag
            else:
                sign_rotate = - sign_rotate
                input_symbol =  sign_rotate * sampled_signal[symbol_num].real
            
            #  Расчет метрик для всех возможных состояний на текущем отсчете
            for i in range(8):
                pm_candidate1 = old_path_metrics[i] + input_symbol - increment[i]
                pm_candidate2 = old_path_metrics[i + 8] + input_symbol - increment[i + 8]
                paths_difference = pm_candidate2 - pm_candidate1
                if paths_difference < 0:
                    new_path_metrics[2 * i] = pm_candidate1
                else:
                    new_path_metrics[2 * i] = pm_candidate2
                trans_table[symbol_num][2 * i] = paths_difference

                pm_candidate1 = old_path_metrics[i] - input_symbol + increment[i]
                pm_candidate2 = old_path_metrics[i + 8] - input_symbol + increment[i + 8]
                paths_difference = pm_candidate2 - pm_candidate1
                if paths_difference < 0:
                    new_path_metrics[2 * i + 1] = pm_candidate1
                else:
                    new_path_metrics[2 * i + 1] = pm_candidate2
                trans_table[symbol_num][2 * i + 1] = paths_difference

            # Обновление путей
            tmp = new_path_metrics
            new_path_metrics = old_path_metrics
            old_path_metrics = tmp
            
            if is_to_table:
                # Первые 8 и последние записываем в data_to_save
                if symbol_num < 8 or symbol_num > 143:
                    # Прогоняем для каждого берста решетку с инкрементами и без них
                    # Учитываем, что счетчик один на разные потоки. Поэтому смотрим (х % 16)
                    # Интересует только первый берст (с и без инкрментами) каждого потока.
                    # Это соответствует значениям счетчика (0, 1); (16, 17) и т.д
                    if ((self.counter % 16) == 0) or ((self.counter % 16) == 1):
                        for i in range (16):
                            formatted_metric = f"{old_path_metrics[i]:.3f}"      
                            data_to_save.append([symbol_num, i,  formatted_metric])

                    if symbol_num == 147:
                        self.counter +=1

            symbol_num += 1

        if is_to_table:
            # запись листа в csv
            # так как в counter добавили 1, то и проверки сдвинулись на 1
            if ((self.counter % 16) == 1) or ((self.counter % 16) == 2):
                with open(name, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';') 
                    writer.writerow(['Number bit', 'State', 'Metric'])
                    writer.writerows(data_to_save)

        return trans_table, old_path_metrics
    
    def find_best_stop_state(self, old_path_metrics, stop_states=[0, 8]):
        best_stop_state = stop_states[0]
        max_stop_state_metric = old_path_metrics[best_stop_state]
        for s in stop_states:
            if old_path_metrics[s] > max_stop_state_metric:
                max_stop_state_metric = old_path_metrics[s]
                best_stop_state = s

        return best_stop_state
    
    @staticmethod
    def calc_llr(total_symbols, hard_bits, survivor_states, trans_table, state_transfer, ebn0):

        # Задержка принятия решения
        decision_delay = 16

        # Инициализация
        # L = log((1 - p_k) / p_k). p_k - вероятность, что символ определен неверно. L > 0
        L = np.ones(total_symbols) * 1000.0
        symbol_num = total_symbols

        merge_distances = []

        while symbol_num > 0:
            symbol_num -= 1
            current_state = survivor_states[symbol_num]

            # Ищем предыдщуее ошибочное состояние
            paths_difference = trans_table[symbol_num][current_state]
            if paths_difference > 0:
                previous_wrong_state = state_transfer[current_state][0]
            else:
                previous_wrong_state = state_transfer[current_state][1]

            # Так как Eb = 1, остается только дисперсия шума
            ebn0_liner =  10 ** (ebn0/10)
            # Нормированная на дисперсию шума разница путей
            delta = np.abs(paths_difference) * 2 * ebn0_liner
            L[symbol_num] = min(L[symbol_num], delta)

            current_wrong_state = previous_wrong_state
            for j in range (symbol_num - 1, max(-1, symbol_num - decision_delay), -1):

                # Проверка, сошлись ли пути
                if current_wrong_state == survivor_states[j]:
                        merge_distances.append(symbol_num - j)
                        break
                
                # Получаем жесткие решения
                survivor_bit = survivor_states[j] % 2
                wrong_bit = current_wrong_state % 2
                # Если биты разные - обновляем llr
                if survivor_bit != wrong_bit:
                    L[j] = min(L[j], delta)
                
                # Идем дальше по неправильному пути (пока биты не станут равными)
                paths_difference = trans_table[j][current_wrong_state]
                if paths_difference > 0:
                    current_wrong_state = state_transfer[current_wrong_state][1]
                else:
                    current_wrong_state = state_transfer[current_wrong_state][0]

        raw_llr = np.zeros(total_symbols, dtype=float)
        for i in range(total_symbols):
            
            # Мягкое решение
            if hard_bits[i] == 0:
                raw_llr[i] = L[i]
            else:
                raw_llr[i] = - L[i]

        # Проецирование на сетку
        llr_scale = 50 # Учитывая, что на 8 дБ (BER = 0.1) среднее 30.
        llr = (np.clip(raw_llr / llr_scale, -1.0, 1.0) * 127.0).astype(np.int8)

        return llr, merge_distances
    
    def traceback(self, trans_table, best_stop_state, ebn0):
        # Таблица переходов: из каких состояний (значения списка) возможно попасть в текущее состояние (индекс списка)
        state_transfer = [
            [0, 8],
            [0, 8],
            [1, 9],
            [1, 9],
            [2, 10],
            [2, 10],
            [3, 11],
            [3, 11],
            [4, 12],
            [4, 12],
            [5, 13],
            [5, 13],
            [6, 14],
            [6, 14],
            [7, 15],
            [7, 15]
        ]

        # Инициализация
        total_symbols = 148
        hard_bits = np.zeros(total_symbols)
        survivor_states = np.zeros(total_symbols, dtype=int)
        symbol_num = total_symbols
        current_state = best_stop_state

        while symbol_num > 0:
            symbol_num -= 1
            
            # Определяем бит по индексу текущего состояния
            hard_bits[symbol_num] = current_state % 2
            survivor_states[symbol_num] = current_state

            # Переходим к предыдущему состоянию с бОльшой метрикой
            paths_difference = trans_table[symbol_num][current_state]
            if paths_difference > 0:
                current_state = state_transfer[current_state][1]
            else:
                current_state = state_transfer[current_state][0]

        if self.type_demod == "vit_soft": 
            llr, merge_distances = self.calc_llr(total_symbols, hard_bits, survivor_states, trans_table, state_transfer, ebn0)        
        else:
            llr = np.zeros(total_symbols, dtype=float)
            merge_distances = np.zeros(total_symbols, dtype=float)

        return hard_bits, llr, merge_distances

    def diff_phase(self, burst_samples):
        y_k = burst_samples[self.sps - 1 :: self.sps]

        y_k_prev = np.zeros(y_k.size, dtype=complex)
        y_k_prev[1:] = y_k[:-1]
        y_k_prev[0] = 1 + 0j

        delta_phi = np.angle(y_k * np.conj(y_k_prev))

        alpha = np.ones(delta_phi.size)
        alpha[delta_phi <= 0] = -1

        d_curr = ((1 - alpha) / 2).astype(int)

        burst_bits = np.zeros(d_curr.size, dtype=int)
        d_prev = 0
        for i in range(d_curr.size):
            burst_bits[i] = d_curr[i] ^ d_prev
            d_prev = burst_bits[i]

        llr = np.zeros(d_curr.size, dtype=float)

        return burst_bits, llr

    def process_detect(self, complex_signal, h, ebn0):

        self.ebn0 = ebn0
        
        sps = self.sps
        samples_per_burst = 156 * sps
        num_bursts = len(complex_signal) // samples_per_burst
    
        burst_bits_output = []
        burst_llr_output = []
        all_merge_distances = []

        for b in range(num_bursts):
 
            start_idx = b * samples_per_burst
            burst = complex_signal[start_idx : start_idx + 148 * sps]

            if self.type_demod == "diff_phase":
                burst_bits, llr = self.diff_phase(burst)
                burst_bits_output.append(burst_bits)
                burst_llr_output.append(llr) 

            elif self.type_demod in ["vit_soft", "vit_hard"]:
                
                rhh_old = self.calc_old_rhh(h[b])
                rhh_new = self.calc_new_rhh(h[b])

                # Старый инкремент новый rhh
                increment_new = self.calc_old_increment(rhh_new)
                increment_old = self.calc_old_increment(rhh_old)

                # Берем отсчеты на конце символьного интервала
                sampled_burst = burst[self.sps - 1 :: self.sps]

                # Строим решетку с инкрменетами
                trans_table, old_path_metrics = self.calc_metric(increment_old, sampled_burst, start_state=0)

                # # Строим решетку без инкрменетов
                # trans_table_2, old_path_metrics_2 = self.calc_metric(increment_new, sampled_burst, start_state=0)

                # Находим наиболее вероятное последнее состояние 
                best_stop_state = self.find_best_stop_state(old_path_metrics)
                # Проходимся от конца к началу по выстроенной решетке
                burst_bits, llr, merge_distances = self.traceback(trans_table, best_stop_state, ebn0)
                # Объединяем результаты разных пакетов 
                burst_bits_output.append(burst_bits)
                burst_llr_output.append(llr)

                all_merge_distances.append(merge_distances)

        detector_bits= np.concatenate(burst_bits_output)
        detector_llr = np.concatenate(burst_llr_output)
        
        detector_merge_distances = np.concatenate(all_merge_distances)

        return detector_bits, detector_llr, detector_merge_distances