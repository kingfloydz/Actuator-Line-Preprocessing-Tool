import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, log10
from scipy.interpolate import interp1d


class AirfoilAerodynamicEstimator:
    """基于论文方法的翼型气动数据估算工具类（无3D修正+移动平均+OpenFOAM格式输出）"""
    
    # -------------------------- 核心常量（移除3D修正相关，新增移动窗口参数）--------------------------
    # 论文基础参数（移除3D修正常量）
    CD_FLAT_PLATE = 1.98
    CD_LE_RADIUS_COEF = -0.64
    CD_THICKNESS_COEF = -0.44
    CN_X1 = 0.0023
    CN_X2 = 0.38
    CN_X3 = 0.62
    CN_X4 = 3.7
    CT_THICKNESS_COEF = 0.3
    CT_SIN2A_COEF = 0.1
    CDf_PRANDTL_A = 1700
    CL_CLIP_MIN = -2.0
    CL_CLIP_MAX = 2.0
    CD_MIN_DEFAULT = 0.001
    EPS = 1e-10

    # Cd_max计算常量
    CD_MAX_BASE = 1.976
    LE_THICK_COEF = -5.366
    TE_SENSITIVITY_BASE = -0.00246
    TE_SENSITIVITY_LE_COEF = -0.05815
    CD_MAX_MIN = 1.0

    # 数据有效性阈值
    ALPHA_EXP_MAX = 90.0
    ALPHA_EXP_MIN = -90.0
    CL_VALID_MIN = -3.0
    CL_VALID_MAX = 3.0
    CD_VALID_MIN = 0.0
    CD_VALID_MAX = 3.0

    # -------------------------- Cl/Cd十个区域的八个边界常量 --------------------------
    ALPHA_CL_N1 = -20.0  # Cl负理论区 ↔ 负平滑区
    ALPHA_CL_N2 = -12.0  # Cl负平滑区 ↔ 原始数据区
    ALPHA_CL_P2 = 28.0   # Cl原始数据区 ↔ 正平滑区
    ALPHA_CL_P1 = 38.0   # Cl正平滑区 ↔ 正理论区

    ALPHA_CD_N1 = -16.0  # Cd负理论区 ↔ 负平滑区
    ALPHA_CD_N2 = -8.0   # Cd负平滑区 ↔ 原始数据区
    ALPHA_CD_P2 = 10.0   # Cd原始数据区 ↔ 正平滑区
    ALPHA_CD_P1 = 20.0   # Cd正平滑区 ↔ 正理论区

    # -------------------------- 新增配置常量 --------------------------
    # 移动窗口平均参数（可调整）
    ROLLING_WINDOW_SIZE = 5  # 移动窗口大小
    ROLLING_MIN_PERIODS = 1  # 最小有效数据点

    # 统一插值攻角范围（与示例一致：-180~180°，步长0.5°）
    TARGET_ALPHAS = np.linspace(-180.0, 180.0, 721)  # 721个点


    # 原始数据攻角列表（用户提供）
    ANGLES_NON_NEG = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
        2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75,
        5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25,
        7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75,
        10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75,
        12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0,
        14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5,
        19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5,
        25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0,
        29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5,
        34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0
    ]
    ANGLES_NEG = [
        -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0, -2.25,
        -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.25,
        -4.5, -4.75, -5.0, -5.25, -5.5, -5.75, -6.0, -6.25,
        -6.5, -6.75, -7.0, -7.25, -7.5, -7.75, -8.0,
        -8.5, -9.0, -9.5, -10.0, -10.5, -11.0, -11.5, -12.0,
        -12.5, -13.0, -13.5, -14.0, -14.5, -15.0, -15.5, -16.0,
        -16.5, -17.0, -17.5, -18.0, -18.5, -19.0, -19.5, -20.0
    ]
    ALL_RAW_ANGLES = ANGLES_NEG + ANGLES_NON_NEG



    def __init__(self, logger, param_manager, geometry,
                 results_excel_filename="airfoil_results.xlsx",
                 section_geo_filename="section_geometry.txt"):
        """初始化参数"""
        self.logger = logger
        self.param_manager = param_manager
        self.geometry = geometry
        self.workdir = param_manager.xfoil_workdir
        self.blade_sections = geometry.blade_sections if hasattr(geometry, 'blade_sections') else []
        
        # 输出文件名
        self.results_excel_filename = results_excel_filename
        self.section_geo_filename = section_geo_filename
        
        self.max_section_num = len(self.blade_sections[0]) if self.blade_sections else 0


    def export_results(self, xfoil_results):
        """主流程：处理数据+保存几何信息+输出气动数据"""
        # 1. 预处理原始数据：筛选有效截面
        raw_df = pd.DataFrame(xfoil_results, columns=['section_num', 'angle', 'cl', 'cd', 'success'])
        valid_sections = self._get_valid_sections(raw_df)
        if not valid_sections:
            self.logger.warning(f"无有效截面（共{self.max_section_num}个截面，筛选后为空）")
            return

        # 2. 保存截面几何信息（位置、弦长、弦向）到TXT
        self._save_section_geometry(valid_sections)

        # 3. 处理各截面气动数据
        excel_data = []  # 汇总所有截面的Excel数据
        for section_num in valid_sections:
            try:
                # 3.1 提取当前截面原始数据
                section_raw_df = self._get_section_raw_data(raw_df, section_num)
                if section_raw_df.empty:
                    self.logger.warning(f"截面{section_num}无有效原始数据，跳过")
                    continue

                # 3.2 获取截面字典
                blade_idx = 0
                section = self.blade_sections[blade_idx][section_num - 1]
                self._check_section_params(section, section_num)

                # 3.3 计算2D理论值
                cd_vals = section_raw_df['cd_raw'].values
                theory_df = self._calculate_section_2d_theory(section, cd_vals)

                # 3.4 Cl/Cd分区域处理
                cl_mixed_df = self._process_cl_by_range(theory_df, section_raw_df)
                cd_mixed_df = self._process_cd_by_range(theory_df, section_raw_df)

                # 3.5 合并结果+移动窗口平均（去掉3D修正）
                final_df = self._merge_cl_cd_results(cl_mixed_df, cd_mixed_df, section)

                # 3.6 插值到统一攻角
                interpolated_df = self._interpolate_to_target_alphas(final_df, section_num)

                # 3.7 保存为OpenFOAM格式
                self._save_openfoam_format(interpolated_df, section_num)

                # 3.8 收集Excel数据（添加截面编号）
                interpolated_df['section_num'] = section_num
                excel_data.append(interpolated_df)

                self.logger.info(f"截面{section_num}处理完成（插值后{len(interpolated_df)}条数据）")

            except Exception as e:
                self.logger.error(f"截面{section_num}处理失败: {str(e)}", exc_info=True)
                continue

        # 4. 保存汇总Excel文件
        if excel_data:
            self._save_excel_results(pd.concat(excel_data, ignore_index=True))
        else:
            self.logger.warning("无有效数据，不生成Excel文件")


    def _check_section_params(self, section, section_num):
        """检查截面必需参数"""
        required_params = [
            'r_LE_over_c', 't_over_c', 'h_over_c', 'reynolds', 
            'position', 'y_over_c_at_x00125', 'trailing_edge_angle',
            'chord_length', 'chord_vector'
        ]
        missing = [p for p in required_params if p not in section]
        if missing:
            raise ValueError(f"截面{section_num}缺失必需参数: {missing}")
        
        # 参数有效性检查
        if section['reynolds'] <= 0:
            raise ValueError(f"截面{section_num}雷诺数无效: {section['reynolds']}")
        if section['chord_length'] <= 0:
            raise ValueError(f"截面{section_num}弦长无效: {section['chord_length']}")


    def _save_section_geometry(self, valid_sections):
        """保存所有有效截面的位置、弦长、弦向到TXT（适配GeometryAnalyzer数据结构）"""
        geo_file_path = os.path.join(self.workdir, self.section_geo_filename)
        
        # 防护：检查blade_sections是否有效
        if not self.blade_sections or len(self.blade_sections) == 0:
            self.logger.warning("无叶片截面几何数据可保存")
            return
        
        with open(geo_file_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("="*85 + "\n")
            f.write("叶片截面几何信息\n")
            f.write("="*85 + "\n")
            # 调整表头格式，适配弦角（角度）字段
            f.write(f"{'截面编号':<10}{'X坐标(m)':<15}{'Y坐标(m)':<15}{'Z坐标(m)':<15}{'弦长(m)':<15}{'扭角(°)':<15}\n")
            f.write("-"*85 + "\n")
            
            # 遍历有效截面（适配GeometryAnalyzer的blade_sections结构）
            blade_idx = 0  # 取第一个叶片（GeometryAnalyzer中对称叶片仅保留1个）
            for section_num in sorted(valid_sections):
                try:
                    # 从blade_sections中获取当前截面（section_num从1开始，索引需-1）
                    section = self.blade_sections[blade_idx][section_num - 1]
                    
                    # 1. 处理位置（numpy数组转标量）
                    pos = section['position']
                    x = float(pos[0]) if isinstance(pos, (np.ndarray, list)) else float(pos)
                    y = float(pos[1]) if isinstance(pos, (np.ndarray, list)) else float(pos)
                    z = float(pos[2]) if isinstance(pos, (np.ndarray, list)) else float(pos)
                    
                    # 2. 处理弦长（确保为标量）
                    chord_len = section.get('chord_length', 0.0)
                    chord_len = float(chord_len) if isinstance(chord_len, (np.ndarray, list)) else float(chord_len)
                    
                    # 3. 处理弦角（弧度转角度，GeometryAnalyzer中chord_angle是弧度）
                    chord_angle_rad = section.get('chord_angle', 0.0)
                    chord_angle_rad = float(chord_angle_rad) if isinstance(chord_angle_rad, (np.ndarray, list)) else float(chord_angle_rad)
                    chord_angle_deg = np.degrees(chord_angle_rad)  # 转换为角度
                    
                    # 格式化输出（保留6位小数，弦角保留2位）
                    f.write(
                        f"{section_num:<10}{x:<15.6f}{y:<15.6f}{z:<15.6f}"
                        f"{chord_len:<15.6f}{chord_angle_deg:<15.2f}\n"
                    )
                except IndexError:
                    self.logger.warning(f"截面编号{section_num}超出索引范围，跳过")
                    continue
                except Exception as e:
                    self.logger.error(f"处理截面{section_num}几何信息失败: {str(e)}", exc_info=True)
                    continue
            
            # 写入结尾标识
            f.write("="*85 + "\n")
            f.write(f"有效截面数量: {len(valid_sections)}\n")
        
        self.logger.info(f"截面几何信息已保存至: {geo_file_path}")

    def _merge_cl_cd_results(self, cl_df, cd_df, section):
        """合并Cl/Cd结果 + 移动窗口平均（去掉3D修正）"""
        # 按攻角合并Cl/Cd数据
        merged_df = pd.merge(
            cl_df[['angle', 'cl_2d_mixed']],
            cd_df[['angle', 'cd_2d_mixed']],
            on='angle',
            how='inner'
        )

        # 添加截面位置信息
        x, y, z = section['position']
        merged_df['x'] = round(x, 6)
        merged_df['y'] = round(y, 6)
        merged_df['z'] = round(z, 6)

        # 按攻角排序（确保移动窗口正确）
        merged_df = merged_df.sort_values('angle').reset_index(drop=True)

        # 移动窗口平均
        merged_df['cl_smoothed'] = merged_df['cl_2d_mixed'].rolling(
            window=self.ROLLING_WINDOW_SIZE,
            min_periods=self.ROLLING_MIN_PERIODS,
            center=True  # 中心窗口（更合理）
        ).mean().fillna(merged_df['cl_2d_mixed'])

        merged_df['cd_smoothed'] = merged_df['cd_2d_mixed'].rolling(
            window=self.ROLLING_WINDOW_SIZE,
            min_periods=self.ROLLING_MIN_PERIODS,
            center=True
        ).mean().fillna(merged_df['cd_2d_mixed'])

        # 限制数值范围
        merged_df['cl_smoothed'] = np.clip(merged_df['cl_smoothed'], self.CL_CLIP_MIN, self.CL_CLIP_MAX)
        merged_df['cd_smoothed'] = np.maximum(merged_df['cd_smoothed'], self.CD_MIN_DEFAULT)

        # 输出列（简化：位置+攻角+平滑后的2D气动参数）
        output_columns = ['x', 'y', 'z', 'angle', 'cl_smoothed', 'cd_smoothed']
        return merged_df[output_columns]


    def _interpolate_to_target_alphas(self, df, section_num):
        """将气动数据插值到统一攻角范围"""
        # 提取原始数据
        raw_angles = df['angle'].values
        raw_cl = df['cl_smoothed'].values
        raw_cd = df['cd_smoothed'].values

        # 创建插值函数（线性插值，边界外使用最近值）
        cl_interp = interp1d(
            raw_angles, raw_cl,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        cd_interp = interp1d(
            raw_angles, raw_cd,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        # 插值到目标攻角
        target_cl = cl_interp(self.TARGET_ALPHAS)
        target_cd = cd_interp(self.TARGET_ALPHAS)

        # 构建结果DF
        interpolated_df = pd.DataFrame({
            'angle': self.TARGET_ALPHAS,
            'cl': np.clip(target_cl, self.CL_CLIP_MIN, self.CL_CLIP_MAX),
            'cd': np.maximum(target_cd, self.CD_MIN_DEFAULT)
        })

        # 添加截面位置信息
        interpolated_df['x'] = df['x'].iloc[0]
        interpolated_df['y'] = df['y'].iloc[0]
        interpolated_df['z'] = df['z'].iloc[0]

        return interpolated_df


    def _save_openfoam_format(self, df, section_num):
        """保存为指定的OpenFOAM格式TXT文件"""
        # 文件名：airfoilProperties_sectionX.txt
        of_filename = f"airfoilProperties_section{section_num}"
        of_file_path = os.path.join(self.workdir, of_filename)

        # 构建文件内容
        header = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  1.6                                   |
|   \\\\  /    A nd           | Web:      http://www.OpenFOAM.org               |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      airfoilProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

airfoilData
(
//   alpha   C_l    C_d
"""
        footer = """);\n"""

        # 构建数据行
        data_lines = []
        for _, row in df.iterrows():
            alpha = round(row['angle'], 2)
            cl = round(row['cl'], 3)
            cd = round(row['cd'], 4)
            # 格式化：对齐空格，保持示例格式
            if alpha >= 0:
                data_lines.append(f"  ( {alpha:6.2f}    {cl:6.3f}   {cd:6.4f})")
            else:
                data_lines.append(f"  ({alpha:6.2f}    {cl:6.3f}   {cd:6.4f})")

        # 写入文件
        with open(of_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write("\n".join(data_lines))
            f.write(footer)

        self.logger.info(f"OpenFOAM格式文件已保存: {of_file_path}")


    def _save_excel_results(self, all_data_df):
        """保存所有截面的插值后数据到Excel（分Sheet）"""
        excel_path = os.path.join(self.workdir, self.results_excel_filename)
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 汇总Sheet
            all_data_df.to_excel(writer, sheet_name="所有截面汇总", index=False)
            
            # 每个截面单独Sheet
            for section_num in sorted(all_data_df['section_num'].unique()):
                section_df = all_data_df[all_data_df['section_num'] == section_num].copy()
                sheet_name = f"截面{section_num}"
                section_df = section_df[['angle', 'cl', 'cd', 'x', 'y', 'z']]
                section_df.to_excel(writer, sheet_name=sheet_name, index=False)

        self.logger.info(f"Excel结果文件已保存: {excel_path}")


    # -------------------------- 原有辅助方法（仅微调）--------------------------
    def _calculate_section_2d_theory(self, section, cd_vals):
        """计算2D理论值（移除3D相关，保留原有逻辑）"""
        r_LE = section['r_LE_over_c']
        t = section['t_over_c']
        h = section['h_over_c']
        Re = section['reynolds']
        is_symmetric = abs(h) < 1e-3
        le_thickness = section['y_over_c_at_x00125']
        te_angle = section['trailing_edge_angle']

        # Cd_max计算
        le_contribution = self.LE_THICK_COEF * le_thickness
        te_sensitivity = self.TE_SENSITIVITY_BASE + self.TE_SENSITIVITY_LE_COEF * le_thickness
        te_contribution = te_sensitivity * te_angle
        cd_max = self.CD_MAX_BASE + le_contribution + te_contribution
        cd_max = max(cd_max, self.CD_MAX_MIN, max(cd_vals))
        CD_90 = cd_max
        CD_270 = cd_max
        alpha0 = 0.0 if is_symmetric else -5.0

        # 生成全攻角理论值
        theory_data = []
        for alpha_deg in np.linspace(-180.0, 180.0, 721):
            alpha_rad = radians(alpha_deg)
            sin_a, cos_a = sin(alpha_rad), cos(alpha_rad)

            # 计算CN_2D和CT_2D
            if is_symmetric:
                cn_num = sin_a + self.CN_X1 * sin(2 * alpha_rad)
                cn_den = self.CN_X2 + self.CN_X3 * abs(sin_a) + self.CN_X4 * t * (cos_a ** 8)
                CN_2D = CD_90 * (cn_num / (cn_den + self.EPS))

                cd_f = 0.455 / (log10(Re) ** 2.58) - self.CDf_PRANDTL_A / Re
                ct_term = self.CT_THICKNESS_COEF * t * abs(sin_a + self.CT_SIN2A_COEF * sin(2 * alpha_rad)) * (1 - 2 * cos_a)
                CT_2D = CD_90 * ct_term - cd_f * cos_a
            else:
                beta = alpha_rad - alpha0 * cos_a
                CD_alpha = (CD_90 + CD_270) / 2 + (CD_90 - CD_270) / 2 * sin(beta)
                
                cn_num = sin_a + self.CN_X1 * sin(2 * alpha_rad)
                cn_den = self.CN_X2 + self.CN_X3 * abs(sin_a) + self.CN_X4 * t * (cos_a ** 8)
                CN_2D = CD_alpha * (cn_num / (cn_den + self.EPS))

                cd_f = 0.455 / (log10(Re) ** 2.58) - self.CDf_PRANDTL_A / Re
                ct_term = self.CT_THICKNESS_COEF * t * abs(sin_a + self.CT_SIN2A_COEF * sin(2 * alpha_rad)) * (1 - 2 * cos_a)
                CT_2D = CD_alpha * ct_term - cd_f * cos_a

            # 计算Cl/Cd理论值
            CL_2D = np.clip(CN_2D * cos_a + CT_2D * sin_a, self.CL_CLIP_MIN, self.CL_CLIP_MAX)
            CD_2D = max(CN_2D * sin_a - CT_2D * cos_a, self.CD_MIN_DEFAULT)

            theory_data.append({
                'angle': round(alpha_deg, 1),
                'cl_2d_theory': round(CL_2D, 6),
                'cd_2d_theory': round(CD_2D, 6),
                'Re': Re
            })

        return pd.DataFrame(theory_data)


    def _process_cl_by_range(self, theory_df, raw_df):
        """Cl分区域处理（原有逻辑不变）"""
        valid_raw_angles = set(raw_df['angle'].tolist())
        merged_df = pd.merge(theory_df, raw_df[['angle', 'cl_raw']], on='angle', how='left')
        result_df = merged_df[['angle', 'cl_2d_theory', 'cl_raw']].copy()
        result_df['cl_2d_mixed'] = np.nan

        # 1. Cl负理论区
        mask_neg_theory = result_df['angle'] < self.ALPHA_CL_N1
        result_df.loc[mask_neg_theory, 'cl_2d_mixed'] = result_df.loc[mask_neg_theory, 'cl_2d_theory']

        # 2. Cl负平滑区
        mask_neg_smooth = (result_df['angle'] >= self.ALPHA_CL_N1) & (result_df['angle'] < self.ALPHA_CL_N2)
        mask_valid = mask_neg_smooth & result_df['angle'].isin(valid_raw_angles)
        if mask_valid.any():
            weights = (result_df.loc[mask_valid, 'angle'] - self.ALPHA_CL_N1) / (self.ALPHA_CL_N2 - self.ALPHA_CL_N1)
            result_df.loc[mask_valid, 'cl_2d_mixed'] = (
                result_df.loc[mask_valid, 'cl_2d_theory'] * (1 - weights) +
                result_df.loc[mask_valid, 'cl_raw'] * weights
            )

        # 3. Cl原始数据区
        mask_raw = (result_df['angle'] >= self.ALPHA_CL_N2) & (result_df['angle'] <= self.ALPHA_CL_P2)
        mask_valid = mask_raw & result_df['angle'].isin(valid_raw_angles)
        result_df.loc[mask_valid, 'cl_2d_mixed'] = result_df.loc[mask_valid, 'cl_raw']

        # 4. Cl正平滑区
        mask_pos_smooth = (result_df['angle'] > self.ALPHA_CL_P2) & (result_df['angle'] <= self.ALPHA_CL_P1)
        mask_valid = mask_pos_smooth & result_df['angle'].isin(valid_raw_angles)
        if mask_valid.any():
            weights = (self.ALPHA_CL_P1 - result_df.loc[mask_valid, 'angle']) / (self.ALPHA_CL_P1 - self.ALPHA_CL_P2)
            result_df.loc[mask_valid, 'cl_2d_mixed'] = (
                result_df.loc[mask_valid, 'cl_2d_theory'] * (1 - weights) +
                result_df.loc[mask_valid, 'cl_raw'] * weights
            )

        # 5. Cl正理论区
        mask_pos_theory = result_df['angle'] > self.ALPHA_CL_P1
        result_df.loc[mask_pos_theory, 'cl_2d_mixed'] = result_df.loc[mask_pos_theory, 'cl_2d_theory']

        return result_df.dropna(subset=['cl_2d_mixed'])


    def _process_cd_by_range(self, theory_df, raw_df):
        """Cd分区域处理（原有逻辑不变）"""
        valid_raw_angles = set(raw_df['angle'].tolist())
        merged_df = pd.merge(theory_df, raw_df[['angle', 'cd_raw']], on='angle', how='left')
        result_df = merged_df[['angle', 'cd_2d_theory', 'cd_raw']].copy()
        result_df['cd_2d_mixed'] = np.nan

        # 1. Cd负理论区
        mask_neg_theory = result_df['angle'] < self.ALPHA_CD_N1
        result_df.loc[mask_neg_theory, 'cd_2d_mixed'] = result_df.loc[mask_neg_theory, 'cd_2d_theory']

        # 2. Cd负平滑区
        mask_neg_smooth = (result_df['angle'] >= self.ALPHA_CD_N1) & (result_df['angle'] < self.ALPHA_CD_N2)
        mask_valid = mask_neg_smooth & result_df['angle'].isin(valid_raw_angles)
        if mask_valid.any():
            weights = (result_df.loc[mask_valid, 'angle'] - self.ALPHA_CD_N1) / (self.ALPHA_CD_N2 - self.ALPHA_CD_N1)
            result_df.loc[mask_valid, 'cd_2d_mixed'] = (
                result_df.loc[mask_valid, 'cd_2d_theory'] * (1 - weights) +
                result_df.loc[mask_valid, 'cd_raw'] * weights
            )

        # 3. Cd原始数据区
        mask_raw = (result_df['angle'] >= self.ALPHA_CD_N2) & (result_df['angle'] <= self.ALPHA_CD_P2)
        mask_valid = mask_raw & result_df['angle'].isin(valid_raw_angles)
        result_df.loc[mask_valid, 'cd_2d_mixed'] = result_df.loc[mask_valid, 'cd_raw']

        # 4. Cd正平滑区
        mask_pos_smooth = (result_df['angle'] > self.ALPHA_CD_P2) & (result_df['angle'] <= self.ALPHA_CD_P1)
        mask_valid = mask_pos_smooth & result_df['angle'].isin(valid_raw_angles)
        if mask_valid.any():
            weights = (self.ALPHA_CD_P1 - result_df.loc[mask_valid, 'angle']) / (self.ALPHA_CD_P1 - self.ALPHA_CD_P2)
            result_df.loc[mask_valid, 'cd_2d_mixed'] = (
                result_df.loc[mask_valid, 'cd_2d_theory'] * (1 - weights) +
                result_df.loc[mask_valid, 'cd_raw'] * weights
            )

        # 5. Cd正理论区
        mask_pos_theory = result_df['angle'] > self.ALPHA_CD_P1
        result_df.loc[mask_pos_theory, 'cd_2d_mixed'] = result_df.loc[mask_pos_theory, 'cd_2d_theory']

        return result_df.dropna(subset=['cd_2d_mixed'])


    def _get_valid_sections(self, raw_df):
        """筛选成功率≥50%的截面"""
        section_stats = {}
        for num, group in raw_df.groupby('section_num'):
            total = len(group)
            success = sum(group['success'])
            section_stats[num] = success / total if total > 0 else 0
        return [num for num, rate in section_stats.items() if rate >= 0.5]


    def _get_section_raw_data(self, raw_df, section_num):
        """提取有效原始数据"""
        section_df = raw_df[(raw_df['section_num'] == section_num) & (raw_df['success'])].copy()
        if section_df.empty:
            return section_df

        # 过滤数值异常数据
        angle_ok = (section_df['angle'] >= self.ALPHA_EXP_MIN) & (section_df['angle'] <= self.ALPHA_EXP_MAX)
        cl_ok = (section_df['cl'] >= self.CL_VALID_MIN) & (section_df['cl'] <= self.CL_VALID_MAX)
        cd_ok = (section_df['cd'] >= self.CD_VALID_MIN) & (section_df['cd'] <= self.CD_VALID_MAX)
        
        valid_df = section_df[angle_ok & cl_ok & cd_ok].copy()
        valid_df = valid_df.rename(columns={'cl': 'cl_raw', 'cd': 'cd_raw'})
        return valid_df[['angle', 'cl_raw', 'cd_raw']].drop_duplicates(subset=['angle'])