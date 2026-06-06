import numpy as np


class DFEEqualizer:

    DEFAULTS = {
        "pk_K1":              6,      # число отводов FF 
        "pk_K2":              10,     # число отводов FB 
        "pk_noise_var":       None,   # фиксированная sigma^2; None -> оценивать по сигналу
        "pk_noise_var_scale": 1.0,    # множитель к оценённой sigma^2
        "pk_noise_var_floor": 1e-3,   # нижний порог sigma^2 (защита от деления на ~0)
        "dfe_llr_scale":      10.0,   # масштаб мягких решений перед 
    }

    def __init__(self, modulation_params, channel_model="awgn"):

        self.sps = int(modulation_params.get("sps", 4))

        def _p(key):
            return modulation_params.get(key, self.DEFAULTS[key])

        self.K1 = int(_p("pk_K1"))
        self.K2 = int(_p("pk_K2"))

        self.noise_var_fixed = _p("pk_noise_var")
        self.noise_var_scale = float(_p("pk_noise_var_scale"))
        self.noise_var_floor = float(_p("pk_noise_var_floor"))
        self.llr_scale       = float(_p("dfe_llr_scale"))

        # Для отладки
        self.last_alpha_decisions = None
        self.last_decisions       = None
        self.last_y_eq            = None
        self.last_soft            = None   
        self.last_noise_var       = None  
        self.last_hard_bits       = None  

    def _build_f_sym(self, h_est):
        h = np.asarray(h_est, dtype=np.complex128)
        E_h = np.sum(np.abs(h**2))
        h_mf = np.conj(h[::-1] / np.sqrt(E_h))
        f = np.convolve(h, h_mf)
        peak = int(np.argmax(np.abs(f)))
        sps = self.sps

        Nside = max(self.K1, self.K2) + 2
        f_sym = np.zeros(2 * Nside + 1, dtype=np.complex128)
        for n in range(-Nside, Nside + 1):
            idx = peak + n * sps
            if 0 <= idx < len(f):
                f_sym[n + Nside] = f[idx]
        return f_sym, Nside

    def _solve_ff(self, f_sym, Nside, noise_var):
        K1 = self.K1
        Nff = K1 + 1

        def fn(n):
            idx = n + Nside
            if 0 <= idx < len(f_sym):
                return f_sym[idx]
            return 0.0 + 0.0j

        Psi = np.zeros((Nff, Nff), dtype=np.complex128)
        for i in range(Nff):
            l = -K1 + i
            for j_idx in range(Nff):
                j = -K1 + j_idx
                s = 0.0 + 0.0j
                for m in range(-K1, 1):
                    s += np.conj(fn(m - l)) * fn(m - j)
                Psi[i, j_idx] = s
            Psi[i, i] += noise_var

        xi = np.zeros(Nff, dtype=np.complex128)
        for j_idx in range(Nff):
            j = -K1 + j_idx
            xi[j_idx] = np.conj(fn(-j))

        try:
            c_ff = np.linalg.solve(Psi, xi)
        except np.linalg.LinAlgError:
            c_ff = np.linalg.lstsq(Psi, xi, rcond=None)[0]
        return c_ff

    def _build_fb(self, c_ff, f_sym, Nside):
        K1, K2 = self.K1, self.K2

        def fn(n):
            idx = n + Nside
            if 0 <= idx < len(f_sym):
                return f_sym[idx]
            return 0.0 + 0.0j

        c_fb = np.zeros(K2, dtype=np.complex128)
        for k in range(1, K2 + 1):
            s = 0.0 + 0.0j
            for i in range(K1 + 1):
                l = -K1 + i
                s += c_ff[i] * fn(k - l)
            c_fb[k - 1] = -s
        return c_fb

    def _estimate_noise_var_residual(self, soft_raw, hard_alpha):

        s = np.abs(np.asarray(soft_raw, dtype=np.float64))
        if s.size == 0:
            return self.noise_var_floor
        m = float(np.mean(s))                 # средний модуль решающей статистики
        nv = float(np.mean((s - m) ** 2))     # дисперсия вокруг него
        return max(nv, self.noise_var_floor) * self.noise_var_scale

    def _make_llr(self, soft_raw, info_bits, noise_var):

        d_bit = np.asarray(info_bits, dtype=int)

        nv = max(float(noise_var), self.noise_var_floor)
        mag = np.abs(soft_raw) * (2.0 / nv)

        # Знак по биту: бит 0 -> +, бит 1 -> -
        raw_llr = np.where(d_bit == 0, mag, -mag)

        llr = (np.clip(raw_llr / self.llr_scale, -1.0, 1.0) * 127.0).astype(np.int8)
        return llr

    @staticmethod
    def _alpha_to_bits(alpha_dec):

        d = ((1.0 - alpha_dec) / 2.0).astype(int)
        bits = np.zeros_like(d)
        dprev = 0
        for i in range(len(d)):
            bits[i] = d[i] ^ dprev
            dprev = bits[i]
        return bits

    def _equalize_burst(self, mf_burst, h_est, noise_var):
        sps = self.sps
        K1, K2 = self.K1, self.K2

        f_sym, Nside = self._build_f_sym(h_est)

        c_ff = self._solve_ff(f_sym, Nside, noise_var)
        c_fb = self._build_fb(c_ff, f_sym, Nside)

        n_sym = 148
        v_sym = mf_burst[sps - 1 :: sps][:n_sym].astype(np.complex128)
        v_pad = np.concatenate([v_sym, np.zeros(K1, dtype=np.complex128)])

        alpha_dec = np.zeros(n_sym, dtype=np.float64)
        soft_raw  = np.zeros(n_sym, dtype=np.float64)
        u_raw     = np.zeros(n_sym, dtype=np.complex128)   # сырой выход эквалайзера u_k
        fb_buf    = np.zeros(K2, dtype=np.complex128)
        u_prev_synth = 1.0 + 0.0j
        phi_synth = 0.0

        for k in range(n_sym):
            I_ff = 0.0 + 0.0j
            for i in range(K1 + 1):
                I_ff += c_ff[i] * v_pad[k + K1 - i]

            I_fb = complex(np.dot(c_fb, fb_buf))
            u_k = I_ff + I_fb
            u_raw[k] = u_k

            rot = u_k * np.conj(u_prev_synth)
            d_phi = np.angle(rot)
            a = 1.0 if d_phi > 0 else -1.0
            alpha_dec[k] = a

            soft_raw[k] = a * np.imag(rot)

            phi_synth += a * (np.pi / 2.0)
            u_prev_synth = np.exp(1j * phi_synth)
            fb_buf = np.roll(fb_buf, 1)
            fb_buf[0] = u_prev_synth

        phi_arr = np.cumsum(alpha_dec * (np.pi / 2.0))
        u_synth = np.exp(1j * phi_arr)

        return alpha_dec, u_synth, soft_raw, u_raw

    def process_eq(self, match_signal, h):
        sps = self.sps
        num_bursts = len(h)
        spb = 156 * sps

        equalized_full = np.zeros(len(match_signal), dtype=np.complex128)
        all_alpha = []
        all_u_synth = []
        all_soft = []
        all_llr = []
        all_nv = []
        all_bits = []

        for b in range(num_bursts):
            start = b * spb
            burst = np.asarray(match_signal[start:start + spb], dtype=np.complex128)
            h_est = np.asarray(h[b], dtype=np.complex128)

            if self.noise_var_fixed is not None:
                nv_weights = float(self.noise_var_fixed)
            else:
                v0 = burst[sps - 1 :: sps][:148]
                p0 = float(np.mean(np.abs(v0) ** 2)) if len(v0) else 0.0
                nv_weights = max(0.01 * p0, self.noise_var_floor) * self.noise_var_scale
            nv_weights = max(nv_weights, self.noise_var_floor)

            alpha_dec, u_synth, soft_raw, u_raw = self._equalize_burst(burst, h_est, nv_weights)

            if self.noise_var_fixed is not None:
                nv_llr = float(self.noise_var_fixed)
            else:
                nv_llr = self._estimate_noise_var_residual(soft_raw, alpha_dec)
            nv_llr = max(nv_llr, self.noise_var_floor)

            info_bits = self._alpha_to_bits(alpha_dec)
            llr = self._make_llr(soft_raw, info_bits, nv_llr)

            n_sym = len(alpha_dec)
            out_idx = (sps - 1) + np.arange(n_sym) * sps
            valid = out_idx < spb
            equalized_full[start + out_idx[valid]] = u_synth[valid]

            all_alpha.append(alpha_dec)
            all_u_synth.append(u_synth)
            all_soft.append(soft_raw)
            all_llr.append(llr)
            all_nv.append(nv_llr)
            all_bits.append(info_bits)

        self.last_alpha_decisions = np.concatenate(all_alpha) if all_alpha else np.array([])
        self.last_decisions = self.last_alpha_decisions.astype(np.complex128)
        self.last_y_eq = np.concatenate(all_u_synth) if all_u_synth else np.array([])
        self.last_soft = np.concatenate(all_soft) if all_soft else np.array([])
        self.last_noise_var = np.array(all_nv) if all_nv else np.array([])

        self.last_hard_bits = (np.concatenate(all_bits)
                               if all_bits else np.array([], dtype=int))

        llr_full = np.concatenate(all_llr) if all_llr else np.array([], dtype=np.int8)
        return equalized_full, llr_full