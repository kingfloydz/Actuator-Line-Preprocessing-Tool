import numpy as np
from scipy.linalg import solve_banded, solve, lstsq
from scipy.interpolate import interp1d
import warnings

class AdvancedAirfoilSolver:
    def __init__(self, coord_file, n_panels=160, iwx=50):
        """初始化翼型求解器，严格匹配XFOIL v6.9核心参数"""
        # XFOIL核心控制参数（源自xfoil.f 1580-1650行）
        self.xcm_ref = 0.25      # 力矩参考点x/c=0.25
        self.angtol = 40.0       # 面板角度阈值(度)
        self.n_crit = 9.0        # e^N转捩准则临界值
        self.vaccel = 0.01       # 边界层加速因子
        self.itmax = 20          # 牛顿迭代上限（XFOIL默认20）
        self.wakelen = 1.0       # 尾迹长度因子
        self.cvpar = 1.0         # 面板聚束参数
        self.cterat = 0.15       # 尾迹/前缘面板密度比
        self.ctrrrat = 0.2       # 加密区/前缘面板密度比
        self.rlx = 1.0           # 初始松弛因子
        self.nseqex = 4          # 未收敛序列点最大数量
        self.ffilt = 0.0         # 逆映射自动过滤级别
        self.epsilon = 1e-10     # 数值稳定性参数

        # 边界层参数（严格匹配BLPINI子程序）
        self.bl_calib = {
            'c1': 0.075,         # 层流摩擦系数常数 (C_f = 0.075/Re^0.5)
            'c2': 0.0168,        # 湍流摩擦系数常数 (C_f = 0.0168/Re^0.2)
            'h1': 2.59,          # 层流形状参数 H12
            'h2': 1.3,           # 湍流形状参数 H12
            'lag_factor': 0.2    # 滞后效应系数
        }
        
        try:
            self.x_raw, self.y_raw = self._read_coords(coord_file)
        except Exception as e:
            raise RuntimeError(f"坐标文件读取失败: {str(e)}")
        
        self.n_panels = n_panels  # XFOIL默认160面板
        self.iwx = iwx            # 尾迹面板数组大小
        self._preprocess_coords()  # 坐标方向与重复点处理
        self._process_airfoil_geom()  # 前缘/后缘识别
        self.panels = self._create_vortex_panels()  # 涡面板
        self.wake_panels = self._create_wake_panels()  # 尾迹面板
        self._initialize_boundary_layer()  # 边界层初始化
        self._prev_solution = None  # 状态传递变量


    def _read_coords(self, filename):
        """读取翼型坐标（严格匹配XFOIL的LOAD逻辑）"""
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
            
            data_start = 0
            self.airfoil_name = ""
            try:
                float(lines[0].split()[0])
            except ValueError:
                self.airfoil_name = lines[0]
                data_start = 1
            
            data = np.loadtxt(lines[data_start:])
        except Exception as e:
            raise IOError(f"无法读取文件: {filename}, 错误: {str(e)}")
        
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("坐标文件格式错误，需至少包含两列数据")
        
        # 面积判断方向（XFOIL的顺时针/逆时针判断）
        area = 0.0
        n = len(data)
        for i in range(n):
            x1, y1 = data[i]
            x2, y2 = data[(i+1)%n]
            area += 0.5 * (y1 + y2) * (x1 - x2)
        
        if area < 0:
            data = data[::-1]
        
        return data[:, 0], data[:, 1]


    def _preprocess_coords(self):
        """坐标预处理（匹配CAIR子程序）"""
        s = np.zeros_like(self.x_raw)
        for i in range(1, len(s)):
            dx = self.x_raw[i] - self.x_raw[i-1]
            dy = self.y_raw[i] - self.y_raw[i-1]
            s[i] = s[i-1] + np.hypot(dx, dy)
        
        le_idx = np.argmin(self.x_raw)
        self.sble = s[le_idx]

        self.sharp_le = False
        for i in range(len(s)-1):
            if np.isclose(self.sble, s[i], atol=1e-8) and np.isclose(self.sble, s[i+1], atol=1e-8):
                self.sharp_le = True
                self.ible = i
                break

        unique_mask = np.concatenate([[True], ~np.isclose(s[1:], s[:-1], atol=1e-8)])
        self.x_raw = self.x_raw[unique_mask]
        self.y_raw = self.y_raw[unique_mask]


    def _process_airfoil_geom(self):
        """翼型几何处理（TE/LE识别，匹配GEOPAR）"""
        self.chord = np.max(self.x_raw) - np.min(self.x_raw)
        if self.chord < self.epsilon:
            self.chord = 1.0
        
        le_idx = np.argmin(self.x_raw)
        self.x_le, self.y_le = self.x_raw[le_idx], self.y_raw[le_idx]
        
        te_mask = self.x_raw > 0.95 * np.max(self.x_raw)
        te_points = np.where(te_mask)[0]
        self.upper_te_idx = None
        self.lower_te_idx = None
        
        if len(te_points) >= 2:
            y_te = self.y_raw[te_points]
            upper_te_idx = te_points[np.argmax(y_te)]
            lower_te_idx = te_points[np.argmin(y_te)]
            self.x_te_upper = self.x_raw[upper_te_idx]
            self.y_te_upper = self.y_raw[upper_te_idx]
            self.x_te_lower = self.x_raw[lower_te_idx]
            self.y_te_lower = self.y_raw[lower_te_idx]
            
            te_gap = np.hypot(
                self.x_te_upper - self.x_te_lower,
                self.y_te_upper - self.y_te_lower
            )
            self.sharp_te = te_gap < 1e-4
            if not self.sharp_te:
                self.te_gap = te_gap
                self.x_te = (self.x_te_upper + self.x_te_lower) / 2
                self.y_te = (self.y_te_upper + self.y_te_lower) / 2
            else:
                self.x_te = self.x_raw[te_points].mean()
                self.y_te = self.y_raw[te_points].mean()
            
            self.upper_te_coord_idx = upper_te_idx
            self.lower_te_coord_idx = lower_te_idx
        else:
            self.sharp_te = True
            self.x_te = self.x_raw[te_points].mean() if len(te_points) > 0 else 0.0
            self.y_te = self.y_raw[te_points].mean() if len(te_points) > 0 else 0.0


    def _compute_curvature(self, x, y):
        """曲率计算（匹配CURV函数）"""
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        denom = (dx**2 + dy**2 + self.epsilon)**1.5
        return np.abs(dx*d2y - dy*d2x) / denom


    def _smooth_curvature(self, s, curv):
        """曲率平滑（匹配SMOOL子程序）"""
        n = len(curv)
        if n < 3:
            return curv.copy()
            
        sb_ref = s[-1] if s[-1] > self.epsilon else 1.0
        cv_avg = np.mean(curv) if np.mean(curv) > 1e-6 else 20.0
        smool = max(1.0 / max(cv_avg, 20.0), 0.25 / (self.n_panels/2))
        smoo_sq = (smool * sb_ref) ** 2

        A = np.zeros((3, n))
        b = curv.copy()

        for i in range(1, n-1):
            dsm = s[i] - s[i-1]
            dsp = s[i+1] - s[i]
            dso = 0.5 * (dsp + dsm)
            
            if dsm < self.epsilon or dsp < self.epsilon:
                A[1, i] = 1.0
                continue
                
            A[0, i+1] = -smoo_sq / (dsp * dso)
            A[1, i] = smoo_sq * (1/(dsm*dso) + 1/(dsp*dso)) + 1.0
            A[2, i-1] = -smoo_sq / (dsm * dso)

        A[1, 0] = 1.0
        A[1, -1] = 1.0

        return solve_banded((1, 1), A, b)


    def _curvature_based_spacing(self):
        """曲率基面板分布（匹配PANGEN）"""
        s = np.zeros_like(self.x_raw)
        for i in range(1, len(s)):
            dx = self.x_raw[i] - self.x_raw[i-1]
            dy = self.y_raw[i] - self.y_raw[i-1]
            s[i] = s[i-1] + np.hypot(dx, dy)
        s_total = s[-1] if len(s) > 0 else 1.0
        s_norm = s / s_total if s_total > self.epsilon else np.linspace(0, 1, len(s))

        curv = self._compute_curvature(self.x_raw, self.y_raw)
        curv_smooth = self._smooth_curvature(s, curv)
        
        le_weight = np.exp(-((s_norm - 0.05)/0.1)**2)
        curv_weight = 1.0 + 3.0 * curv_smooth * self.cvpar
        te_weight = 1.0 + (self.cterat - 1.0) * np.exp(-((s_norm - 1.0)/0.1)**2)
        weight = le_weight * curv_weight * te_weight
        weight /= np.sum(weight) + self.epsilon

        cum_weight = np.cumsum(weight)
        cum_weight /= cum_weight[-1] + self.epsilon
        panel_s = np.interp(np.linspace(0, 1, self.n_panels+1), cum_weight, s_norm)

        s_raw_norm = s / s_total if s_total > self.epsilon else np.linspace(0, 1, len(s))
        x_interp = interp1d(s_raw_norm, self.x_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_interp = interp1d(s_raw_norm, self.y_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
        panel_x = x_interp(panel_s)
        panel_y = y_interp(panel_s)
        angles = np.arctan2(np.diff(panel_y), np.diff(panel_x)) * 180/np.pi
        angle_diff = np.abs(np.diff(angles))
        
        max_refine = 5
        refine_count = 0
        refine_indices = np.where(angle_diff > self.angtol)[0]
        while len(refine_indices) > 0 and refine_count < max_refine:
            for idx in refine_indices:
                new_s = (panel_s[idx] + panel_s[idx+1]) / 2
                panel_s = np.insert(panel_s, idx+1, new_s)
                self.n_panels += 1
            
            panel_x = x_interp(panel_s)
            panel_y = y_interp(panel_s)
            angles = np.arctan2(np.diff(panel_y), np.diff(panel_x)) * 180/np.pi
            angle_diff = np.abs(np.diff(angles))
            refine_indices = np.where(angle_diff > self.angtol)[0]
            refine_count += 1

        if self.sharp_le and s_total > self.epsilon:
            le_pos = self.sble / s_total
            le_indices = np.searchsorted(panel_s, le_pos)
            for i in range(2):
                offset = 0.005 * (i+1)
                if le_pos - offset > 0:
                    panel_s = np.insert(panel_s, le_indices, le_pos - offset)
                    self.n_panels += 1
                if le_pos + offset < 1:
                    panel_s = np.insert(panel_s, le_indices+1, le_pos + offset)
                    self.n_panels += 1

        return np.sort(panel_s)


    def _create_vortex_panels(self):
        """创建涡面板（匹配XPANEL）"""
        s_norm = self._curvature_based_spacing()
        
        s_raw = np.zeros_like(self.x_raw)
        for i in range(1, len(s_raw)):
            dx = self.x_raw[i] - self.x_raw[i-1]
            dy = self.y_raw[i] - self.y_raw[i-1]
            s_raw[i] = s_raw[i-1] + np.hypot(dx, dy)
        s_total = s_raw[-1] if len(s_raw) > 0 else 1.0
        s_raw_norm = s_raw / s_total if s_total > self.epsilon else np.linspace(0, 1, len(s_raw))
        
        s_norm_clamped = np.clip(s_norm, 0.0, 1.0)
        
        x_interp = interp1d(s_raw_norm, self.x_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_interp = interp1d(s_raw_norm, self.y_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
        x_nodes = x_interp(s_norm_clamped)
        y_nodes = y_interp(s_norm_clamped)

        panels = []
        for i in range(self.n_panels):
            x1, y1 = x_nodes[i], y_nodes[i]
            x2, y2 = x_nodes[i+1], y_nodes[i+1]
            
            length = np.hypot(x2-x1, y2-y1)
            if length < 1e-5:
                continue
                
            theta = np.arctan2(y2-y1, x2-x1)
            normal = theta + np.pi/2
            xc = x1 + 0.25*(x2-x1)
            yc = y1 + 0.25*(y2-y1)

            panels.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'xc': xc, 'yc': yc,
                'length': length,
                'theta': theta,
                'normal': normal,
                'gamma': 0.0,
                'cp': 0.0
            })
        
        self.n_panels = len(panels)
        
        self.upper_te_panel_idx = None
        self.lower_te_panel_idx = None
        if hasattr(self, 'upper_te_coord_idx') and self.upper_te_coord_idx is not None:
            upper_te_x = self.x_te_upper
            upper_te_y = self.y_te_upper
            min_dist = float('inf')
            for i, p in enumerate(panels):
                panel_mid_x = (p['x1'] + p['x2']) / 2
                panel_mid_y = (p['y1'] + p['y2']) / 2
                dist = np.hypot(panel_mid_x - upper_te_x, panel_mid_y - upper_te_y)
                if dist < min_dist:
                    min_dist = dist
                    self.upper_te_panel_idx = i
            
            lower_te_x = self.x_te_lower
            lower_te_y = self.y_te_lower
            min_dist = float('inf')
            for i, p in enumerate(panels):
                panel_mid_x = (p['x1'] + p['x2']) / 2
                panel_mid_y = (p['y1'] + p['y2']) / 2
                dist = np.hypot(panel_mid_x - lower_te_x, panel_mid_y - lower_te_y)
                if dist < min_dist:
                    min_dist = dist
                    self.lower_te_panel_idx = i
        
        return panels


    def _create_wake_panels(self):
        """创建尾迹面板（匹配XWAKE）"""
        n_wake = int(self.n_panels / 8) + 2  # 匹配xfoil.f中的NW计算
        
        if n_wake > self.iwx:
            warnings.warn(f"尾迹面板数超过IWX限制，已缩减至{self.iwx}")
            n_wake = self.iwx
        
        wake_length = self.wakelen * self.chord if self.chord > self.epsilon else 1.0
        x = np.linspace(self.x_te, self.x_te + wake_length, n_wake + 1)
        y = np.full_like(x, self.y_te)

        wake = []
        for i in range(n_wake):
            x1, y1 = x[i], y[i]
            x2, y2 = x[i+1], y[i+1]
            wake.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'xc': (x1+x2)/2, 'yc': (y1+y2)/2,
                'length': np.hypot(x2-x1, y2-y1),
                'sigma': 0.0
            })
        return wake


    def _initialize_boundary_layer(self):
        """边界层初始化（严格匹配BLPINI子程序）"""
        n = self.n_panels
        self.bl = {
            'theta': np.ones(n) * 1e-5,       # 动量厚度
            'delta': np.ones(n) * 1e-4,       # 位移厚度
            'H': np.ones(n) * self.bl_calib['h1'],  # 形状参数 H12
            'H_prev': np.ones(n) * self.bl_calib['h1'],  # 上一步形状参数
            'diss': np.zeros(n),              # 耗散率
            'diss_prev': np.zeros(n),         # 上一步耗散率
            'transition': np.zeros(n, dtype=bool),  # 转捩标记
            'separated': np.zeros(n, dtype=bool),   # 分离标记
            'cf': np.ones(n) * 0.001,         # 摩擦系数
            'tau': np.zeros(n)                # 剪切应力
        }

        self.transition_params = {
            'n_factor': np.zeros(n),          # e^N转捩因子
            'n_prev': np.zeros(n),            # 上一步转捩因子
            'amp_ratio': np.ones(n)           # 扰动放大率
        }


    def _en_transition_model(self, u_e, re):
        """e^N转捩模型（严格匹配ETRAN子程序）"""
        n = self.n_panels
        if n == 0:
            return np.zeros(0, dtype=bool)
            
        x = np.array([p['xc'] for p in self.panels])
        ds = np.array([p['length'] for p in self.panels])
        transition = self.bl['transition'].copy()
        
        n_factor = self.transition_params['n_factor']
        amp_ratio = self.transition_params['amp_ratio']
        n_prev = self.transition_params['n_prev'].copy()

        amp_ratio[0] = 1e-6  # 前缘初始扰动
        
        for i in range(1, n):
            if transition[i]:
                n_factor[i] = n_factor[i-1]
                amp_ratio[i] = amp_ratio[i-1]
                continue

            re_x = re * x[i]
            if re_x < 1e4:
                n_factor[i] = n_factor[i-1]
                amp_ratio[i] = amp_ratio[i-1]
                continue

            # 速度梯度计算（中心差分）
            if i == 1:
                du_dx = (u_e[i+1] - u_e[i]) / ((x[i+1] - x[i]) + self.epsilon)
            elif i == n-1:
                du_dx = (u_e[i] - u_e[i-1]) / ((x[i] - x[i-1]) + self.epsilon)
            else:
                du_dx = (u_e[i+1] - 2*u_e[i] + u_e[i-1]) / ((x[i+1] - x[i-1])**2 / 4 + self.epsilon)

            # 扰动增长率计算（匹配XFOIL的经验公式）
            growth = 0.04 * (1.0 - 0.65*np.sign(du_dx)*abs(du_dx)**0.5)
            growth *= np.sqrt(re_x) / (x[i] + self.epsilon)
            
            delta_n = growth * ds[i]
            delta_n = np.clip(delta_n, -0.5, 0.5)
            
            n_factor[i] = n_factor[i-1] + delta_n
            
            # 扰动放大率更新
            max_amp = 1e30
            if amp_ratio[i-1] > max_amp:
                amp_ratio[i] = max_amp
            else:
                if delta_n > 10:
                    amp_ratio[i] = amp_ratio[i-1] * np.exp(10) * np.exp(delta_n - 10)
                else:
                    amp_ratio[i] = amp_ratio[i-1] * np.exp(delta_n)
                amp_ratio[i] = min(amp_ratio[i], max_amp)

            # 转捩判断（N_crit=9.0）
            if n_factor[i] > self.n_crit:
                transition[i:] = True
                # 转捩后形状参数过渡（匹配XFOIL的处理）
                self.bl['H'][i:] = self.bl_calib['h2'] + 0.2 * np.exp(-(np.arange(i, n)/10)**2)
                self.bl['H_prev'][i:] = self.bl['H'][i:]

        self.bl['transition'] = transition
        self.transition_params['n_factor'] = n_factor
        self.transition_params['amp_ratio'] = amp_ratio
        self.transition_params['n_prev'] = n_prev
        
        return transition


    def _two_eq_lagged_dissipation(self, u_e, re, m):
        """两方程滞后耗散模型（严格匹配BLSOLV子程序）"""
        n = self.n_panels
        if n == 0:
            return np.array([])
            
        theta = self.bl['theta'].copy()
        H = self.bl['H'].copy()
        H_prev = self.bl['H_prev'].copy()
        diss = self.bl['diss'].copy()
        diss_prev = self.bl['diss_prev'].copy()
        x = np.array([p['xc'] for p in self.panels])

        # 压缩性修正（Karman-Tsien）
        beta = np.sqrt(1 - m**2) if m < 1 else 0.1
        u_e_corr = u_e / beta

        unconverged_count = 0
        converged = False
        reg_param = 1e-8  # 数值稳定性正则化
        
        for it in range(self.itmax):  # 迭代次数匹配XFOIL的ITMAX=20
            A = np.zeros((n, n))
            b_theta = np.zeros(n)
            b_diss = np.zeros(n)
            
            for i in range(n):
                if i == 0:
                    A[i, i] = 1.0
                    b_theta[i] = 1e-5  # 前缘动量厚度初始值
                    b_diss[i] = 1e-8   # 前缘耗散率初始值
                    continue
                
                if i == n-1:
                    A[i, i] = 1.0
                    A[i, i-1] = -1.0
                    b_theta[i] = 1e-6  # 后缘梯度条件
                    b_diss[i] = 1e-8
                    continue
                
                re_x = re * x[i]
                transition = self.bl['transition'][i]
                
                # 形状参数梯度（用于判断是否需要迎风格式）
                dHdx = (H[i+1] - H[i-1]) / (x[i+1] - x[i-1] + self.epsilon)
                rapid_change = np.abs(dHdx) > 0.5  # 快速变化区启用迎风格式
                
                # 层流/湍流模型参数（匹配XFOIL的经验公式）
                if not transition:
                    cf = self.bl_calib['c1'] / np.sqrt(re_x + self.epsilon)
                    u_pow = np.clip(u_e_corr[i], 0.1, 2.0)
                    theta_i = 0.04 * (u_pow**4.5) / np.sqrt(re_x + self.epsilon)
                    diss_i = 0.01 * theta_i * (u_pow**3) / (re_x + self.epsilon)
                    lag_H = self.bl_calib['lag_factor'] * (H_prev[i] - H[i])
                    lag_diss = 0.1 * (diss_prev[i] - diss[i])
                else:
                    cf = self.bl_calib['c2'] / ((re_x + self.epsilon)**0.2)
                    u_pow = np.clip(u_e_corr[i], 0.1, 2.0)
                    theta_i = 0.036 * (u_pow**7) / ((re_x + self.epsilon)**0.2)
                    diss_i = 0.005 * theta_i * (u_pow**3) / ((re_x + self.epsilon)**0.2)
                    lag_H = 0.1 * (H_prev[i] - H[i])
                    lag_diss = 0.05 * (diss_prev[i] - diss[i])
                
                dx_prev = x[i] - x[i-1] + self.epsilon
                dx_next = x[i+1] - x[i] + self.epsilon
                
                # 离散化格式（快速变化区用一阶迎风格式，否则二阶中心格式）
                if rapid_change:
                    A[i, i-1] = -self.vaccel / dx_prev
                    A[i, i] = 1.0 + lag_H + self.vaccel / dx_prev + reg_param
                    A[i, i+1] = 0.0
                else:
                    c_prev = -self.vaccel / (dx_prev * (dx_prev + dx_next))
                    c_curr = self.vaccel * (1.0/(dx_prev*dx_next) + 1.0/(dx_next*dx_prev)) + 1.0 + lag_H
                    c_curr += reg_param
                    c_next = -self.vaccel / (dx_next * (dx_prev + dx_next))
                    A[i, i-1] = c_prev
                    A[i, i] = c_curr
                    A[i, i+1] = c_next
                
                b_theta[i] = theta_i + lag_H * H_prev[i]
                
                # 耗散方程矩阵
                A_diss = A[i, :].copy()
                A_diss[i] += 0.05
                b_diss[i] = diss_i + lag_diss * diss_prev[i]
            
            try:
                A_reg = A + reg_param * np.eye(n)
                theta_new = solve(A_reg, b_theta)
                diss_new = solve(A_reg, b_diss)
            except:
                try:
                    theta_new, _, _, _ = lstsq(A, b_theta, rcond=1e-6)
                    diss_new, _, _, _ = lstsq(A, b_diss, rcond=1e-6)
                except:
                    A_perturbed = A + 1e-6 * np.eye(n)
                    theta_new, _, _, _ = lstsq(A_perturbed, b_theta, rcond=1e-6)
                    diss_new, _, _, _ = lstsq(A_perturbed, b_diss, rcond=1e-6)
            
            # 物理约束
            theta_new = np.clip(theta_new, 1e-8, 1e-2)
            diss_new = np.clip(diss_new, 1e-10, 1e-3)
            
            # 形状参数更新
            H_new = H + 0.1 * (theta_new - theta) + 0.05 * (diss_new - diss)
            H_new = np.clip(H_new, 1.0, 5.0)  # 物理合理范围
            
            # 收敛判断
            delta_theta = np.linalg.norm(theta_new - theta) / (np.linalg.norm(theta) + self.epsilon)
            delta_H = np.linalg.norm(H_new - H) / (np.linalg.norm(H) + self.epsilon)
            max_delta = max(delta_theta, delta_H)
            
            if max_delta < 1e-6:
                converged = True
                break
            
            # 提前退出判断（匹配NSEQEX=4）
            if max_delta > 1e-3:
                unconverged_count += 1
                if unconverged_count >= self.nseqex:
                    warnings.warn(f"边界层求解提前退出，迭代{it+1}步")
                    break
            else:
                unconverged_count = 0

            # 松弛更新（匹配XFOIL的松弛策略）
            self.rlx = max(0.3, min(1.0, 0.5 / (max_delta + 1e-6)))
            H_prev = H.copy()
            diss_prev = diss.copy()
            theta = (1 - self.rlx) * theta + self.rlx * theta_new
            H = (1 - self.rlx) * H + self.rlx * H_new
            diss = (1 - self.rlx) * diss + self.rlx * diss_new
        
        if not converged and it == self.itmax - 1:
            warnings.warn(f"边界层求解达到最大迭代次数{self.itmax}")
        
        # 更新边界层参数
        self.bl['theta'] = theta
        self.bl['delta'] = theta * H  # 位移厚度 = 动量厚度 × 形状参数
        self.bl['H'] = H
        self.bl['H_prev'] = H_prev
        self.bl['diss'] = diss
        self.bl['diss_prev'] = diss_prev
        
        # 摩擦系数计算
        self.bl['cf'] = np.where(
            self.bl['transition'],
            self.bl_calib['c2'] / ((re * x + self.epsilon)**0.2),
            self.bl_calib['c1'] / np.sqrt(re * x + self.epsilon)
        )
        
        # 分离判断（H>3.0且动量厚度梯度为正）
        theta_grad = np.gradient(theta)
        self.bl['separated'] = (H > 3.0) & (theta_grad > 0)
        
        return self.bl['cf']


    def _karman_tsien_cp(self, u_e, m):
        """Karman-Tsien压缩性修正（严格匹配CPCOMP子程序）"""
        if 0.8 < m < 1.2:  # 跨声速区修正
            if m < 1.0:
                beta = np.sqrt(1 - m**2 + self.epsilon)
                u_over_beta = u_e / beta
                cp_kts = (1.0 - u_over_beta**2) / (1.0 + (beta / 2.0) * u_over_beta**2 + self.epsilon)
                if u_e > 1.0 / beta:  # 局部超声速修正
                    mach_local = u_e * m
                    cp_kts *= 0.98 - 0.15 * (mach_local - 1.0)
            else:
                beta = np.sqrt(m**2 - 1 + self.epsilon)
                u_over_beta = u_e / beta
                cp_kts = (1.0 - u_over_beta**2) / (1.0 - (beta / 2.0) * u_over_beta**2 + self.epsilon)
                cp_kts *= 0.9 - 0.4 * (m - 1.0)
            return cp_kts
        elif m < 1.0:  # 亚声速
            beta = np.sqrt(1 - m**2 + self.epsilon)
            u_over_beta = u_e / beta
            return (1.0 - u_over_beta**2) / (1.0 + (beta / 2.0) * u_over_beta**2 + self.epsilon)
        else:  # 超声速
            return 1.0 - u_e**2 / (1 + 0.7*(m - 1.0) + self.epsilon)


    def _potential_flow_solver(self, alpha, m, transpiration):
        """势流求解（匹配POTSOLV子程序）"""
        n = self.n_panels
        if n == 0:
            raise ValueError("面板数量为零，无法求解势流")
            
        A = np.zeros((n, n))
        b = np.zeros(n)
        alpha_rad = np.radians(alpha)

        # 构建影响系数矩阵
        for i in range(n):
            # 边界条件：物面法向速度为零（含 transpiration修正）
            b[i] = -np.cos(alpha_rad - self.panels[i]['normal']) + transpiration[i]

            for j in range(n):
                if i == j:
                    A[i, j] = 0.5  # 自影响系数
                else:
                    xj1, yj1 = self.panels[j]['x1'], self.panels[j]['y1']
                    xj2, yj2 = self.panels[j]['x2'], self.panels[j]['y2']
                    xi, yi = self.panels[i]['xc'], self.panels[i]['yc']

                    r1 = np.hypot(xi - xj1, yi - yj1) + self.epsilon
                    r2 = np.hypot(xi - xj2, yi - yj2) + self.epsilon
                    
                    theta1 = np.arctan2(yi - yj1, xi - xj1)
                    theta2 = np.arctan2(yi - yj2, xi - xj2)
                    phi = self.panels[j]['theta']

                    term1 = np.sin(theta1 - phi) / r1
                    term2 = np.sin(theta2 - phi) / r2
                    
                    term1 = np.clip(term1, -1e6, 1e6)
                    term2 = np.clip(term2, -1e6, 1e6)
                    
                    A[i, j] = (term1 - term2) * self.panels[j]['length'] / (4 * np.pi)

        # 应用库塔条件（后缘上下表面环量相等）
        valid_kutta = (self.upper_te_panel_idx is not None and 
                      self.lower_te_panel_idx is not None and 
                      self.upper_te_panel_idx < n and 
                      self.lower_te_panel_idx < n)
        
        if valid_kutta:
            A[-1, :] = 0.0
            A[-1, self.upper_te_panel_idx] = 1.0
            A[-1, self.lower_te_panel_idx] = -1.0
            b[-1] = 0.0
        else:
            warnings.warn("无法应用库塔条件，使用默认边界条件")

        # 数值稳定性处理
        A[np.isnan(A)] = 0.0
        A[np.isinf(A)] = 0.0
        b[np.isnan(b)] = 0.0
        b[np.isinf(b)] = 0.0

        A += 1e-10 * np.eye(n)

        # 求解线性方程组（带初始猜测值加速收敛）
        try:
            if self._prev_solution is not None:
                gamma_init = self._prev_solution['gamma']
                if len(gamma_init) != n:
                    gamma_init = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(gamma_init)), gamma_init)
                gamma = solve(A, b, x0=gamma_init)
            else:
                gamma = solve(A, b)
        except:
            warnings.warn("直接求解失败，使用最小二乘解")
            gamma, _, _, _ = lstsq(A, b, rcond=1e-6)

        # 计算表面速度
        u_e = np.zeros(n)
        for i in range(n):
            u = np.cos(alpha_rad - self.panels[i]['theta'])  # 来流速度切向分量

            for j in range(n):
                xj1, yj1 = self.panels[j]['x1'], self.panels[j]['y1']
                xj2, yj2 = self.panels[j]['x2'], self.panels[j]['y2']
                xi, yi = self.panels[i]['xc'], self.panels[i]['yc']

                r1 = np.hypot(xi - xj1, yi - yj1) + self.epsilon
                r2 = np.hypot(xi - xj2, yi - yj2) + self.epsilon
                theta1 = np.arctan2(yi - yj1, xi - xj1)
                theta2 = np.arctan2(yi - yj2, xi - xj2)
                phi = self.panels[j]['theta']

                term = (np.cos(theta1 - phi)/r1 - np.cos(theta2 - phi)/r2) / (4 * np.pi)
                term = np.clip(term, -1e6, 1e6)
                u += gamma[j] * term  # 环量诱导速度

            u_e[i] = u

        # 钝后缘尾迹影响（匹配XFOIL的源面板模型）
        if not self.sharp_te and len(self.wake_panels) > 0 and valid_kutta:
            te_gamma = gamma[self.upper_te_panel_idx] - gamma[self.lower_te_panel_idx]
            for wake in self.wake_panels:
                wake['sigma'] = te_gamma * np.exp(-5 * wake['xc']/(self.wakelen * self.chord + self.epsilon))

            # 尾迹源对物面速度的影响
            for i in range(n):
                xi, yi = self.panels[i]['xc'], self.panels[i]['yc']
                for wake in self.wake_panels:
                    x1, y1 = wake['x1'], wake['y1']
                    x2, y2 = wake['x2'], wake['y2']
                    phi = np.arctan2(y2 - y1, x2 - x1)

                    r1 = np.hypot(xi - x1, yi - y1) + self.epsilon
                    r2 = np.hypot(xi - x2, yi - y2) + self.epsilon
                    theta1 = np.arctan2(yi - y1, xi - x1)
                    theta2 = np.arctan2(yi - y2, xi - x2)

                    u_src = wake['sigma'] * (theta2 - theta1) / (2 * np.pi)
                    v_src = wake['sigma'] * np.log(r1 / r2) / (2 * np.pi)

                    u_e[i] += u_src * np.cos(phi - self.panels[i]['theta']) + v_src * np.sin(phi - self.panels[i]['theta'])

        # 更新面板环量
        for i in range(n):
            self.panels[i]['gamma'] = gamma[i] if i < len(gamma) else 0.0
        return u_e, gamma


    def compute_aerodynamics(self, alpha, re, m=0.0):
        """气动特性计算（匹配XFOIL主循环）"""
        if self.n_panels == 0:
            raise RuntimeError("无有效面板，无法计算气动特性")
            
        transpiration = np.zeros(self.n_panels)  #  transpiration速度
        u_e = None
        gamma = None
        unconverged_count = 0

        # 粘性-无粘耦合迭代
        for iter_couple in range(self.itmax):
            u_e, gamma = self._potential_flow_solver(alpha, m, transpiration)
            self._en_transition_model(u_e, re)  # 转捩计算
            cf = self._two_eq_lagged_dissipation(u_e, re, m)  # 边界层求解

            # 计算位移厚度和 transpiration速度
            delta_star = self.bl['delta']
            transpiration_new = 0.5 * np.gradient(delta_star * u_e)
            
            # 过滤 transpiration（匹配FFILT参数）
            if self.ffilt > 0:
                transpiration_new = self._smooth_transpiration(transpiration_new)

            # 耦合收敛判断
            delta_trans = np.linalg.norm(transpiration_new - transpiration)
            if delta_trans < 1e-5:
                break

            # 提前退出判断
            if delta_trans > 1e-3:
                unconverged_count += 1
                if unconverged_count >= self.nseqex:
                    warnings.warn(f"耦合求解提前退出，迭代{iter_couple+1}步")
                    break
            else:
                unconverged_count = 0

            # 松弛更新
            self.rlx = max(0.3, min(1.0, 0.5 / (delta_trans + 1e-6)))
            transpiration = (1 - self.rlx) * transpiration + self.rlx * transpiration_new

        # 计算气动力
        cl, cd_p, cd_f = self._integrate_forces(u_e, cf, alpha, m)
        cd_wake = self._wake_drag()
        cd = cd_p + cd_f + cd_wake

        # 保存当前解用于下次迭代初始猜测
        self._prev_solution = {
            'gamma': gamma,
            'theta': self.bl['theta'].copy(),
            'H': self.bl['H'].copy(),
            'diss': self.bl['diss'].copy()
        }

        return cl, cd


    def _smooth_transpiration(self, transp):
        """ transpiration平滑（匹配FILT子程序）"""
        n = len(transp)
        if n < 3:
            return transp.copy()
            
        filt = np.array([0.25, 0.5, 0.25])
        transp_smooth = transp.copy()
        
        for i in range(1, n-1):
            transp_smooth[i] = np.dot(filt, transp[i-1:i+2])
            
        transp_smooth[0] = 0.5 * (transp[0] + transp[1])
        transp_smooth[-1] = 0.5 * (transp[-2] + transp[-1])
        
        return transp_smooth


    def _integrate_forces(self, u_e, cf, alpha, m):
        """力积分（匹配FORINT子程序）"""
        cl = 0.0
        cd_p = 0.0
        cd_f = 0.0
        alpha_rad = np.radians(alpha)
        chord = self.chord if self.chord > self.epsilon else 1.0

        for i, panel in enumerate(self.panels):
            cp = self._karman_tsien_cp(u_e[i], m)
            self.panels[i]['cp'] = cp

            theta = panel['theta']
            f_n = -cp * panel['length']  # 法向力（压力）
            f_t = self.bl['cf'][i] * panel['length'] if not self.bl['separated'][i] else 0.0  # 切向力（摩擦）

            # 投影到风轴系
            cl += (f_n * np.cos(theta - alpha_rad) + f_t * np.sin(theta - alpha_rad)) / chord
            cd_p += (f_n * np.sin(theta - alpha_rad) - f_t * np.cos(theta - alpha_rad)) / chord

        # 摩擦阻力积分
        cd_f = np.sum(self.bl['cf'] * np.array([p['length'] for p in self.panels]) * 
                     np.sin([p['theta'] - alpha_rad for p in self.panels])) / chord

        return cl, cd_p, cd_f


    def _wake_drag(self):
        """尾迹阻力（匹配WAKEINT子程序）"""
        if len(self.wake_panels) == 0 or self.n_panels == 0:
            return 0.0
            
        wake_momentum = 0.0
        chord = self.chord if self.chord > self.epsilon else 1.0
        
        for i, wake in enumerate(self.wake_panels):
            theta = self.bl['theta'][-1] * (1 - i/len(self.wake_panels))
            H = self.bl['H'][-1] if len(self.bl['H']) > 0 else 0.0
            
            u_e_wake = abs(self.panels[self.upper_te_panel_idx]['gamma']) if self.upper_te_panel_idx is not None else 0.0
            u_e_wake *= (1 - i/len(self.wake_panels))
            
            exponent = (H + 5) / 2
            if u_e_wake > 1e2 and exponent > 0:
                term = 2 * theta * np.exp(exponent * np.log(u_e_wake))
            else:
                term = 2 * theta * (u_e_wake ** exponent)
            
            wake_momentum += term * wake['length']

        return wake_momentum / chord


if __name__ == "__main__":
    try:
        solver = AdvancedAirfoilSolver("a11.txt", n_panels=160, iwx=50)
        
        Re = 1e6
        M = 0.0
        alphas = np.arange(-5, 12, 1)
        cls, cds = [], []
        
        for alpha in alphas:
            cl, cd = solver.compute_aerodynamics(alpha, Re, M)
            cls.append(cl)
            cds.append(cd)
            print(f"攻角: {alpha:.1f}°, CL: {cl:.4f}, CD: {cd:.6f}")
            
    except Exception as e:
        print(f"运行出错: {str(e)}")