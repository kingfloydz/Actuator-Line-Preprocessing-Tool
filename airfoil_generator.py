import os
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from scipy.optimize import curve_fit


class AirfoilGenerator:
    def __init__(self, param_manager, geometry):
        self.param_manager = param_manager
        self.geometry = geometry
        self.blade_sections = geometry.blade_sections if hasattr(geometry, 'blade_sections') else []
        self.rotor_speed_is_negative = self.geometry.rotor_speed_is_negative        
        self.workdir = param_manager.xfoil_workdir
        self.num_foil_points = param_manager.num_foil_points  # 总采样点数
        self.intersection_tolerance = 1e-6  # 相交检测容差
        self.max_angle_threshold = 40.0     # 最大允许夹角
        self.target_angle_threshold = 20.0  # 目标夹角
        self.logger = param_manager.logger if hasattr(param_manager, 'logger') else None

    # ------------------------------
    # 核心转换方法（保持接口不变）
    # ------------------------------
    def convert_to_xfoil_format(self, blade_idx=0, section_idx=0, method='bspline'):
        """将叶片截面转换为XFOIL格式的点集，遵循逆时针排列并采用均匀采样"""
        # 提取共用的归一化路径数据（逻辑不变）
        section, normalized_points, upper_path, lower_path, adj = self._get_normalized_paths(blade_idx, section_idx)
        leading_edge_idx = section['leading_idx']
        trailing_edge_idx = section['trailing_idx']

        normalized_points = self.smooth_partial_edge_points(
            section_idx,
            normalized_points, 
            adj,  
            window_size=3,
            num_iterations=15,
            num_segments=10,
            lower_boundary=-1,
            upper_boundary=0.1,
        )
        
        normalized_points = self.smooth_partial_edge_points(
            section_idx,
            normalized_points, 
            adj,  
            window_size=3,
            num_iterations=2,
            num_segments=5,
            lower_boundary=0.095,
            upper_boundary=0.95
        )
        
        normalized_points = self.smooth_partial_edge_points(
            section_idx,
            normalized_points, 
            adj,  
            window_size=3,
            num_iterations=20,
            num_segments=3,
            lower_boundary=0.97,
            upper_boundary=2
        )
        
        upper_splits = self._process_angle_correction(upper_path, normalized_points)
        lower_splits = self._process_angle_correction(lower_path, normalized_points)
              
        self._calculate_airfoil_parameters(section, upper_path, lower_path, normalized_points)
        leading_edge, trailing_edge = (normalized_points[leading_edge_idx], 
                                    normalized_points[trailing_edge_idx])
        
        upper_segments, lower_segments = self._generate_smooth_curves(
            normalized_points, upper_path, lower_path, method, upper_splits, lower_splits
        )
        sampled_upper, sampled_lower = self._uniform_sampling(upper_segments, lower_segments)
        sampled_upper, sampled_lower = self._process_sampled_intersections(
            sampled_upper, sampled_lower, leading_edge, trailing_edge, section_idx
        )
        
        xfoil_points = self._construct_xfoil_points(sampled_upper, sampled_lower, trailing_edge)
        
        return xfoil_points
    
    def _calculate_airfoil_parameters(self, section, upper_path, lower_path, normalized_points):
        # 提取并排序上下表面点
        upper_points = np.array([normalized_points[idx] for idx in reversed(upper_path)])
        lower_points = np.array([normalized_points[idx] for idx in lower_path])
        upper_points = upper_points[np.argsort(upper_points[:, 0])]
        lower_points = lower_points[np.argsort(lower_points[:, 0])]

        # 提取x/c和y/c坐标
        upper_x, upper_y = upper_points[:, 0], upper_points[:, 1]
        lower_x, lower_y = lower_points[:, 0], lower_points[:, 1]

        # 创建插值函数（参数统一，无需冗余注释）
        upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, 
                            fill_value=(upper_y[0], upper_y[-1]))
        lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, 
                            fill_value=(lower_y[0], lower_y[-1]))

        # 计算x/c=0.0125处y/c值
        target_x = 0.0125   
        section['y_over_c_at_x00125'] = (upper_interp(target_x) - lower_interp(target_x)) / 2.0

        # 计算后缘气流角（ζ）
        trailing_edge_x_threshold = 0.95
        upper_te_mask = upper_x >= trailing_edge_x_threshold
        lower_te_mask = lower_x >= trailing_edge_x_threshold
    
        # 检查后缘区域数据点，优化条件分支
        logger = self.logger if hasattr(self, 'logger') else None
        if np.sum(upper_te_mask) < 2 or np.sum(lower_te_mask) < 2:
            if logger:
                logger.warning("后缘区域数据点不足，无法准确计算后缘角度")
            section['trailing_edge_angle'] = 0.0  # 异常情况默认值
        else:
            # 提取后缘区域点并拟合
            upper_te_points = upper_points[upper_te_mask]
            lower_te_points = lower_points[lower_te_mask]
            m_upper, b_upper = np.polyfit(upper_te_points[:, 0], upper_te_points[:, 1], 1)
            m_lower, b_lower = np.polyfit(lower_te_points[:, 0], lower_te_points[:, 1], 1)
            
            # 计算角度并转换为度
            theta_upper, theta_lower = np.arctan(m_upper), np.arctan(m_lower)
            section['trailing_edge_angle'] = np.degrees(np.abs(theta_upper - theta_lower))

        # 计算论文必需参数
        # 最大厚度/弦长（t_over_c）
        all_x = np.linspace(min(np.min(upper_x), np.min(lower_x)), 
                        max(np.max(upper_x), np.max(lower_x)), 1000)
        upper_y_all, lower_y_all = upper_interp(all_x), lower_interp(all_x)
        full_thickness = upper_y_all - lower_y_all
        section['t_over_c'] = max(np.max(full_thickness), 0)  # 确保非负

        # 弯度/弦长（h_over_c）
        midline_y = (upper_y_all + lower_y_all) / 2.0
        max_camber = np.max(midline_y)
        section['h_over_c'] = 0.0 if abs(max_camber) < 1e-4 else max_camber
        if logger and abs(max_camber) < 1e-4:
            logger.debug(f"对称翼型（弯度：{max_camber:.6f}），h_over_c设为0")

        # 前缘半径/弦长（r_LE_over_c）
        le_x_threshold = 0.05
        upper_le_points = upper_points[upper_x <= le_x_threshold]
        lower_le_points = lower_points[lower_x <= le_x_threshold]
        # 提取重复计算的默认半径表达式
        default_radius = max(0.012 * (section['t_over_c'] **2), 1e-6)

        if len(upper_le_points) < 3 or len(lower_le_points) < 3:
            if logger:
                logger.warning(f"前缘区域点不足（上表面：{len(upper_le_points)}个，下表面：{len(lower_le_points)}个），用经验公式估算r_LE_over_c")
            section['r_LE_over_c'] = default_radius
        else:
            upper_le_x, upper_le_y = upper_le_points[:, 0], upper_le_points[:, 1]
            valid_le_mask = upper_le_y >= 0
            
            if np.sum(valid_le_mask) < 3:
                if logger:
                    logger.warning("前缘有效点（y≥0）不足，用经验公式估算r_LE_over_c")
                section['r_LE_over_c'] = default_radius
            else:
                # 半圆方程拟合
                def circle_eq(x, r):
                    return np.sqrt(2 * r * x - x** 2)
                
                try:
                    popt, _ = curve_fit(circle_eq, upper_le_x[valid_le_mask], 
                                    upper_le_y[valid_le_mask], bounds=(1e-6, 0.1))
                    le_radius = popt[0]
                    if le_radius <= 0 or le_radius > section['t_over_c'] / 2:
                        raise ValueError(f"拟合半径不合理（{le_radius:.6f}）")
                    section['r_LE_over_c'] = le_radius
                    if logger:
                        logger.debug(f"前缘半径/弦长：{section['r_LE_over_c']:.6f}（基于x/c≤{le_x_threshold}区域拟合）")
                except Exception as e:
                    if logger:
                        logger.warning(f"前缘半径拟合失败（{str(e)}），用经验公式估算")
                    section['r_LE_over_c'] = default_radius

        # 参数完整性检查
        required_paper_params = ['r_LE_over_c', 't_over_c', 'h_over_c', 'AR']
        missing_params = [p for p in required_paper_params if p not in section]
        if missing_params and logger:
            logger.error(f"论文必需参数缺失：{missing_params}，可能导致后续外推失败")
        elif logger:
            logger.info(f"翼型参数计算完成：r_LE_over_c={section['r_LE_over_c']:.6f}, "
                    f"t_over_c={section['t_over_c']:.6f}, h_over_c={section['h_over_c']:.6f}, AR={section['AR']:.2f}")

    def smooth_partial_edge_points(self, section_idx, normalized_points, adj, window_size, num_iterations, num_segments, lower_boundary=-1, upper_boundary=2):
        """对选定边缘点进行光滑处理，不动点自身位置不变但参与其他点的平均计算"""
        # 复制原始点集，保存不动点的原始位置
        original_points = normalized_points.copy()
        processed_points = normalized_points.copy()
        
        # 确定目标点（选定范围内的边缘点）
        target_mask = (normalized_points[:, 0] >= lower_boundary) & (normalized_points[:, 0] <= upper_boundary)
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) < num_segments:  # 确保有足够的点进行处理
            return processed_points
        
        # 辅助函数：获取有效邻居（在目标索引集中的邻居）
        index_set = set(target_indices)
        get_valid_neighbors = lambda idx: [n for n in adj.get(idx, []) if n in index_set]
        
        # 确定所有端点（只有1个有效邻居的点）
        endpoints = [idx for idx in target_indices if len(get_valid_neighbors(idx)) == 1]
        num_paths = len(endpoints) // 2  # 路径数为端点数的一半
        if num_paths == 0:
            return processed_points  # 没有足够的端点形成路径
        # 端点按x值降序排序，便于有序选择起点并避免重复
        sorted_endpoints = sorted(endpoints, key=lambda x: normalized_points[x][0], reverse=True) 
        # 辅助函数：基于邻接关系对路径点进行排序（从start_idx开始）
        def sort_by_adjacency(start_idx):
            sorted_list = [start_idx]
            prev_idx = None
            current_idx = start_idx
            
            # 即使起点只有一个邻点，也尝试向前延伸
            while True:
                neighbors = get_valid_neighbors(current_idx)
                # 排除上一个点，获取可能的下一个点
                possible_next = [n for n in neighbors if n != prev_idx]
                
                if not possible_next:
                    break  # 没有可继续的点，结束路径
                
                next_point = possible_next[0]
                if next_point in sorted_list:
                    break  # 避免循环
                
                sorted_list.append(next_point)
                prev_idx, current_idx = current_idx, next_point
                
                # 当遇到新的端点（只有一个有效邻居）时，结束路径
                if len(get_valid_neighbors(current_idx)) == 1:
                    break
            
            return sorted_list

        fixed_set = set()  # 存储所有路径的不动点
        paths = []  # 存储所有路径
        used_endpoints = set()  # 记录已使用的端点

        # 生成路径，确保每个端点只使用一次
        for start_idx in sorted_endpoints:
            if start_idx in used_endpoints:
                continue
                    
            sorted_path = sort_by_adjacency(start_idx)

            if len(sorted_path) < 3:  # 只处理足够长的路径
                continue
                    
            end_idx = sorted_path[-1]
            # 检查是否形成有效路径且终点未被使用
            if end_idx in index_set and end_idx not in used_endpoints:
                paths.append(sorted_path)
                used_endpoints.update([start_idx, end_idx])
    
            if len(paths) >= num_paths:
                break

        if not paths:  # 无有效路径时直接返回
            return processed_points

        # 对每个路径分别处理：计算评分并选择不动点
        for sorted_path in paths:
            num_points = len(sorted_path)
            index_map = {idx: pos for pos, idx in enumerate(sorted_path)}

            # 计算该路径所有点的评分（角度差/长度和）
            path_scores = {}
            for curr_idx in sorted_path:
                neighbors = get_valid_neighbors(curr_idx)

                # 按排序位置取前2个邻居
                sorted_neighbors = sorted(neighbors, key=lambda x: index_map[x])[:2]
                if len(sorted_neighbors) < 2:
                    path_scores[curr_idx] = float('inf')
                    continue
                
                prev_idx, next_idx = sorted_neighbors
                # 向量计算
                vec1, vec2 = normalized_points[curr_idx] - normalized_points[prev_idx], normalized_points[next_idx] - normalized_points[curr_idx]
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 < 1e-10 or norm2 < 1e-10:
                    path_scores[curr_idx] = float('inf')
                    continue   
                # 角度差计算
                cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                angle_diff = abs(np.pi - np.arccos(cos_theta)) 
                path_scores[curr_idx] = angle_diff / (norm1 + norm2) 

            # 将路径点均分为num_segments份
            base, remainder = divmod(num_points, num_segments)
            segment_sizes = [base + 1] * remainder + [base] * (num_segments - remainder)
            split_indices = [0]
            current = 0

            for size in segment_sizes:
                current += size
                split_indices.append(current)
            segments = [sorted_path[split_indices[i]:split_indices[i+1]] 
                    for i in range(num_segments) if split_indices[i] < split_indices[i+1]]

            # 选择该路径的不动点（每段中评分最低的点）
            for segment in segments:
                valid_scores = sorted(((idx, path_scores[idx]) for idx in segment 
                                    if path_scores[idx] != float('inf')), 
                                    key=lambda x: x[1])
                fixed_idx = valid_scores[0][0] if valid_scores else segment[len(segment) // 2]
                fixed_set.add(fixed_idx)

        # 应用移动平均光滑处理
        half_window = window_size // 2
        for _ in range(num_iterations):
            current_input = processed_points.copy()
            
            for sorted_path in paths:
                num_points = len(sorted_path)
                if num_points < 3:
                    continue
                    
                start_idx, end_idx = sorted_path[0], sorted_path[-1]
                for pos, idx in enumerate(sorted_path):
                    # 不动点和路径端点保持不变
                    if idx in fixed_set:
                        processed_points[idx] = original_points[idx]
                        continue
                    
                    # 计算窗口范围
                    start_pos = max(0, pos - half_window)
                    end_pos = min(num_points, pos + half_window + 1)
                    window_indices = sorted_path[start_pos:end_pos]
                    
                    # 计算权重
                    window_positions = np.arange(start_pos, end_pos)
                    distances = np.abs(window_positions - pos)
                    weights = (half_window + 1 - distances).astype(float)
                    weights /= weights.sum()
                    
                    # 加权平均计算（区分不动点和非不动点）
                    is_fixed = np.array([win_idx in fixed_set for win_idx in window_indices])
                    points = np.where(is_fixed[:, None], 
                                    original_points[window_indices], 
                                    current_input[window_indices])
                    processed_points[idx] = np.sum(points * weights[:, None], axis=0)
      
        return processed_points
    
    # ------------------------------
    # 通用相交处理工具方法（提取的核心逻辑）
    # ------------------------------
    def _ccw(self, A, B, C):
        """计算逆时针判断值（通用工具）"""
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def _segments_intersect_general(self, a1, a2, b1, b2, points=None):
        """
        通用线段相交检测
        - 若points为None：a1,a2,b1,b2为点坐标
        - 若points不为None：a1,a2,b1,b2为点索引，从points中获取坐标
        """
        # 获取实际点坐标
        A = a1 if points is None else points[a1]
        B = a2 if points is None else points[a2]
        C = b1 if points is None else points[b1]
        D = b2 if points is None else points[b2]

        # 边界框快速排除
        if (max(A[0], B[0]) < min(C[0], D[0]) - self.intersection_tolerance or
            max(C[0], D[0]) < min(A[0], B[0]) - self.intersection_tolerance or
            max(A[1], B[1]) < min(C[1], D[1]) - self.intersection_tolerance or
            max(C[1], D[1]) < min(A[1], B[1]) - self.intersection_tolerance):
            return False

        # 精确判断
        ccw1, ccw2 = self._ccw(A, B, C), self._ccw(A, B, D)
        ccw3, ccw4 = self._ccw(C, D, A), self._ccw(C, D, B)

        # 标准交叉情况
        if (ccw1 * ccw2 < -self.intersection_tolerance) and (ccw3 * ccw4 < -self.intersection_tolerance):
            return True

        # 检查点是否在线段上
        def on_segment(p, seg_a, seg_b):
            if (min(seg_a[0], seg_b[0]) - self.intersection_tolerance <= p[0] <= max(seg_a[0], seg_b[0]) + self.intersection_tolerance and
                min(seg_a[1], seg_b[1]) - self.intersection_tolerance <= p[1] <= max(seg_a[1], seg_b[1]) + self.intersection_tolerance):
                return abs(self._ccw(seg_a, seg_b, p)) < self.intersection_tolerance
            return False

        return (on_segment(C, A, B) or on_segment(D, A, B) or 
                on_segment(A, C, D) or on_segment(B, C, D))

    def _line_intersection_general(self, a1, a2, b1, b2, points=None):
        """
        通用线段交点计算
        - 若points为None：a1,a2,b1,b2为点坐标
        - 若points不为None：a1,a2,b1,b2为点索引，从points中获取坐标
        """
        # 获取实际点坐标
        x1, y1 = a1 if points is None else points[a1]
        x2, y2 = a2 if points is None else points[a2]
        x3, y3 = b1 if points is None else points[b1]
        x4, y4 = b2 if points is None else points[b2]

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < self.intersection_tolerance:
            # 平行或重合时返回中点
            return ((x1 + x3) / 2, (y1 + y3) / 2)

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def _calculate_local_slopes(self, points, idx):
        """计算点集在指定索引附近的局部斜率（通用方法）"""
        slopes = []
        # 检查前后2个点范围内的线段斜率
        for i in range(max(0, idx - 2), min(len(points) - 1, idx + 2)):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if abs(x2 - x1) > self.intersection_tolerance:
                slopes.append((y2 - y1) / (x2 - x1))
        return slopes if slopes else [0]

    def _calculate_separation_direction(self, upper_slopes, lower_slopes):
        """计算分离方向（垂直于平均斜率）"""
        avg_slope = (np.mean(upper_slopes) + np.mean(lower_slopes)) / 2
        # 垂直方向向量
        dir_x, dir_y = (avg_slope, -1) if abs(avg_slope) > self.intersection_tolerance else (1, 0)
        # 归一化
        length = np.hypot(dir_x, dir_y)
        if length > self.intersection_tolerance:
            dir_x /= length
            dir_y /= length
        # 确保上方向y为正
        if dir_y <= 0:
            dir_x, dir_y = -dir_x, -dir_y
        return dir_x, dir_y

    # ------------------------------
    # 采样后点的相交处理（复用通用方法）
    # ------------------------------
    def _process_sampled_intersections(self, upper_points, lower_points, leading_edge, trailing_edge,section_idx):
        """处理采样后上下表面点之间的相交问题"""
        while True:
            intersection_data = self._detect_sampled_intersections(upper_points, lower_points, leading_edge, trailing_edge)
            if not intersection_data:
                break  # 无相交时退出循环
            upper_points, lower_points = self._resolve_sampled_intersection(
                upper_points, lower_points, leading_edge, trailing_edge,** intersection_data
            )
        return upper_points, lower_points

    def _detect_sampled_intersections(self, upper_points, lower_points, leading_edge, trailing_edge):
            """检测采样后的上下表面点是否相交，排除前缘点和后缘点相关的相交"""
            intersections = []
            
            # 辅助函数：判断点是否为前缘点或后缘点
            def is_leading_or_trailing(point):
                return (np.linalg.norm(point - leading_edge) < self.intersection_tolerance or
                        np.linalg.norm(point - trailing_edge) < self.intersection_tolerance)
            
            # 检测线段相交（排除包含前缘/后缘点的线段）
            for i in range(len(upper_points) - 1):
                a1, a2 = upper_points[i], upper_points[i+1]
                # 跳过包含前缘/后缘点的上表面线段
                if is_leading_or_trailing(a1) or is_leading_or_trailing(a2):
                    continue
                    
                for j in range(len(lower_points) - 1):
                    b1, b2 = lower_points[j], lower_points[j+1]
                    # 跳过包含前缘/后缘点的下表面线段
                    if is_leading_or_trailing(b1) or is_leading_or_trailing(b2):
                        continue
                        
                    if self._segments_intersect_general(a1, a2, b1, b2):  # 复用通用检测
                        intersection = self._line_intersection_general(a1, a2, b1, b2)  # 复用通用计算
                        # 同时确保交点本身不是前缘/后缘点
                        if not is_leading_or_trailing(intersection):
                            intersections.append({
                                'x': intersection[0],
                                'upper_i': i,
                                'lower_j': j,
                                'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2
                            })
            
            # 检测重合点（排除前缘/后缘点）
            for i, upper_pt in enumerate(upper_points):
                # 跳过前缘/后缘点
                if is_leading_or_trailing(upper_pt):
                    continue
                    
                for j, lower_pt in enumerate(lower_points):
                    # 跳过前缘/后缘点
                    if is_leading_or_trailing(lower_pt):
                        continue
                        
                    if np.linalg.norm(upper_pt - lower_pt) < self.intersection_tolerance:
                        intersections.append({
                            'x': upper_pt[0],
                            'upper_i': i,
                            'lower_j': j,
                            'a1': upper_pt, 'a2': upper_pt, 'b1': lower_pt, 'b2': lower_pt
                        })
            
            # 返回x最小的相交点
            if intersections:
                intersections.sort(key=lambda x: x['x'])
                result = intersections[0]
                del result['x']
                return result
            return None


    def _resolve_sampled_intersection(self, upper_points, lower_points, leading_edge, trailing_edge, 
                                    upper_i, lower_j, a1, a2, b1, b2):
        """处理采样后点的相交问题：去除相交线段的两个点，在前后点间进行等距插值生成两个新点"""
        # ------------------------------
        # 处理上表面：等距插值
        # ------------------------------
        # 确定上表面前后参考点（确保索引有效）
        upper_prev_idx = max(0, upper_i - 1)
        upper_next_idx = min(len(upper_points) - 1, upper_i + 2)
        upper_prev_pt = upper_points[upper_prev_idx]
        upper_next_pt = upper_points[upper_next_idx]
        
        # 计算前后点之间的向量和总距离
        dx_upper = upper_next_pt[0] - upper_prev_pt[0]
        dy_upper = upper_next_pt[1] - upper_prev_pt[1]
        total_dist_upper = np.hypot(dx_upper, dy_upper)
        
        # 等距插值参数（分成3段相等距离）
        t_values = [1/3, 2/3]  # 相对于总距离的比例位置
        upper_interp_points = []
        
        for t in t_values:
            # 计算等距点坐标
            x = upper_prev_pt[0] + dx_upper * t
            y = upper_prev_pt[1] + dy_upper * t
            upper_interp_points.append([x, y])
        
        # 构建新的上表面点集
        upper_new = np.concatenate([
            upper_points[:upper_prev_idx + 1],
            np.array(upper_interp_points),
            upper_points[upper_next_idx:]
        ])
        
        # ------------------------------
        # 处理下表面：等距插值
        # ------------------------------
        # 确定下表面前后参考点（确保索引有效）
        lower_prev_idx = max(0, lower_j - 1)
        lower_next_idx = min(len(lower_points) - 1, lower_j + 2)
        lower_prev_pt = lower_points[lower_prev_idx]
        lower_next_pt = lower_points[lower_next_idx]
        
        # 计算前后点之间的向量和总距离
        dx_lower = lower_next_pt[0] - lower_prev_pt[0]
        dy_lower = lower_next_pt[1] - lower_prev_pt[1]
        total_dist_lower = np.hypot(dx_lower, dy_lower)
        
        # 等距插值参数（分成3段相等距离）
        lower_interp_points = []
        for t in t_values:
            # 计算等距点坐标
            x = lower_prev_pt[0] + dx_lower * t
            y = lower_prev_pt[1] + dy_lower * t
            lower_interp_points.append([x, y])
        
        # 构建新的下表面点集
        lower_new = np.concatenate([
            lower_points[:lower_prev_idx + 1],
            np.array(lower_interp_points),
            lower_points[lower_next_idx:]
        ])
        
        return upper_new, lower_new
    # ------------------------------
    # 路径的相交处理（复用通用方法）
    # ------------------------------
    def _extract_surface_paths(self, points, adj, leading_edge_idx, trailing_edge_idx,section_idx):
        """分离上下表面路径，循环处理相交情况直到无相交"""
        le_neighbors = adj[leading_edge_idx]
        if len(le_neighbors) != 2:
            raise ValueError(f"前缘点应有且仅有两个邻点，但实际有 {len(le_neighbors)} 个")
       
        n1, n2 = le_neighbors

        upper_start, lower_start = (n1, n2) if points[n1][1] > points[n2][1] else (n2, n1)

        # 构建初始上下表面路径
        upper_path = self._build_surface_path(
            upper_start, leading_edge_idx, trailing_edge_idx, adj, points
        )
        lower_path = self._build_surface_path(
            lower_start, leading_edge_idx, trailing_edge_idx, adj, points
        )
        # 循环检测并处理相交
        while True:
            intersect_data = self._detect_intersections(upper_path, lower_path, points, leading_edge_idx, trailing_edge_idx)
            if not intersect_data:
                break
            upper_path, lower_path, points, adj = self._resolve_intersection(
                upper_path, lower_path, points, adj, leading_edge_idx, 
                trailing_edge_idx,** intersect_data
            )
        
        return upper_path, lower_path, points,adj

    def _detect_intersections(self, upper_path, lower_path, points, leading_edge_idx, trailing_edge_idx):
        """检测上下表面路径是否相交（排除前缘点和后缘点）"""
        intersections = []
        excluded_indices = {leading_edge_idx, trailing_edge_idx}
        
        # 检测线段相交（排除包含前缘/后缘点的线段）
        for i in range(len(upper_path)-1):
            a1, a2 = upper_path[i], upper_path[i+1]
            if a1 in excluded_indices or a2 in excluded_indices:
                continue
                
            for j in range(len(lower_path)-1):
                b1, b2 = lower_path[j], lower_path[j+1]
                if b1 in excluded_indices or b2 in excluded_indices:
                    continue
                    
                if self._segments_intersect_general(a1, a2, b1, b2, points):  # 复用通用检测
                    intersection = self._line_intersection_general(a1, a2, b1, b2, points)  # 复用通用计算
                    intersections.append({
                        'x': intersection[0],
                        'upper_i': i, 'lower_j': j,
                        'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2
                    })
        
        # 检测上下路径是否有相同的点（排除前缘/后缘点）
        for upper_i, upper_idx in enumerate(upper_path):
            if upper_idx in excluded_indices:
                continue
            for lower_j, lower_idx in enumerate(lower_path):
                if lower_idx in excluded_indices:
                    continue
                if np.linalg.norm(points[upper_idx] - points[lower_idx]) < self.intersection_tolerance:
                    intersections.append({
                        'x': points[upper_idx][0],
                        'upper_i': upper_i, 'lower_j': lower_j,
                        'a1': upper_idx, 'a2': upper_idx, 'b1': lower_idx, 'b2': lower_idx
                    })
        
        # 返回x最小的相交点
        if intersections:
            intersections.sort(key=lambda x: x['x'])
            result = intersections[0]
            del result['x']
            return result
        return None

    def _resolve_intersection(self, upper_path, lower_path, points, adj, 
                            leading_idx, trailing_idx, upper_i, lower_j, a1, a2, b1, b2):
        """处理路径相交问题"""
        # 计算交点
        intersection = self._line_intersection_general(a1, a2, b1, b2, points)  # 复用通用计算
        
        # 获取局部斜率（复用通用方法）
        upper_slopes = self._calculate_local_slopes([points[idx] for idx in upper_path], upper_i)
        lower_slopes = self._calculate_local_slopes([points[idx] for idx in lower_path], lower_j)
        
        # 计算分离方向（复用通用方法）
        dir_x, dir_y = self._calculate_separation_direction(upper_slopes, lower_slopes)
        
        # 计算新的上下点位置
        offset = 0.00005
        new_upper = (intersection[0] + dir_x * offset, intersection[1] + dir_y * offset)
        new_lower = (intersection[0] - dir_x * offset, intersection[1] - dir_y * offset)

        # 更新点集和索引
        points = np.vstack([points, new_upper, new_lower])
        upper_new_idx, lower_new_idx = len(points) - 2, len(points) - 1
        
        # 更新路径
        upper_path = upper_path[:upper_i+1] + [upper_new_idx] + upper_path[upper_i+1:]
        lower_path = lower_path[:lower_j+1] + [lower_new_idx] + lower_path[lower_j+1:]
        
        # 更新邻接表
        self._update_adjacency(adj, upper_new_idx, upper_path, upper_i)
        self._update_adjacency(adj, lower_new_idx, lower_path, lower_j)
        
        # 移除交点后多余的点
        upper_path,adj = self._remove_points_beyond_intersection(
            upper_path, trailing_idx, points, new_upper[0],adj
        )
        lower_path,adj = self._remove_points_beyond_intersection(
            lower_path, trailing_idx, points, new_lower[0],adj
        )
        
        return upper_path, lower_path, points, adj

    def _get_normalized_paths(self, blade_idx, section_idx):
        """提取归一化点和上下表面路径的共用逻辑"""
        
        section = self._validate_and_process_section(blade_idx, section_idx)
        normalized_points = self._normalize_points(
            section['points'], section['leading_edge'], section['chord_vector'],
            section['chord_length'], section['span_direction']
        )
        adj = self._build_adjacency(section['edge'])
        
        upper_path, lower_path, normalized_points,adj = self._extract_surface_paths(
            normalized_points, adj, section['leading_idx'], section['trailing_idx'],section_idx
        )
        
        return section, normalized_points, upper_path, lower_path, adj

    def _validate_and_process_section(self, blade_idx, section_idx):
        """验证并预处理叶片截面数据"""
        if not (0 <= blade_idx < len(self.blade_sections)):
            raise IndexError(f"叶片索引 {blade_idx} 超出范围")
        
        sections = self.blade_sections[blade_idx]
        if not (0 <= section_idx < len(sections)):
            raise IndexError(f"截面索引 {section_idx} 超出范围")
        
        section = sections[section_idx]

        # 验证边数据
        for edge in section['edge']:
            if len(edge) < 2:
                raise ValueError(f"边数据格式错误：期望至少包含两个点索引，但得到 {len(edge)} 个值")
        
        # 验证弦长
        chord_length = section.get('chord_length', 0)
        if chord_length <= 0:
            raise ValueError("弦长必须为正数")
        
        return section
    
    def _normalize_points(self, points, leading_edge, chord_vector, chord_length, span_dir):
        """归一化点到二维坐标系"""
        translated_points = points - leading_edge
        projections = []
        for p in translated_points:
            component = np.dot(p, span_dir)
            proj = p - component * span_dir
            projections.append(proj)
        projected_points = np.array(projections)
        
        x_axis = -chord_vector  
        y_axis = -np.cross(span_dir, x_axis)
        
        y_axis_normalized = y_axis / np.linalg.norm(y_axis)

        if np.dot(y_axis_normalized, self.geometry.axial_direction) > 0:
            y_axis_normalized = -y_axis_normalized

        normalized_points = []
        for p in projected_points:
            x = np.dot(p, x_axis)
            y = np.dot(p, y_axis_normalized)
            normalized_points.append([x, y])
        
        return np.array(normalized_points) / chord_length

    def _build_adjacency(self, edges):
        """构建邻接表（去除自环边）"""
        adj = {}
        for edge in edges:
            p1, p2 = edge[:2]
            # 跳过自环边（当两个节点相同时不添加连接）
            if p1 == p2:
                continue
            # 为p1添加p2
            adj.setdefault(p1, []).append(p2)
            # 为p2添加p1
            adj.setdefault(p2, []).append(p1)
        return adj
    
    def _build_surface_path(self, start_idx, leading_idx, trailing_idx, adj, points):
        """构建从前缘到后缘的表面路径"""
        path = [leading_idx, start_idx]
        current = start_idx
        while True:
            neighbors = adj[current]
            candidates = []
            for n in neighbors:
                if n == trailing_idx:
                    candidates.append(n)
                elif n not in path:
                    candidates.append(n)
            if not candidates:
                break
            
            if trailing_idx in candidates:
                path.append(trailing_idx)
                break

            next_point = max(candidates, key=lambda idx: points[idx][0])
            path.append(next_point)
            current = next_point
            
            if len(path) > len(points) * 2:
                raise RuntimeError("构建表面路径时出现无限循环，可能是网格拓扑错误")
        
        return path

    def _update_adjacency(self, adj, new_idx, path, idx):
        """更新邻接表以包含新点"""
        prev_idx, next_idx = path[idx], path[idx+2]
        adj[new_idx] = [prev_idx, next_idx]
        
        adj[prev_idx] = [i for i in adj[prev_idx] if i != path[idx+1]] + [new_idx]
        adj[next_idx] = [i for i in adj[next_idx] if i != path[idx+1]] + [new_idx]
    
    def _remove_points_beyond_intersection(self, path, trailing_idx, points, intersection_x, adj):
        """移除交点后多余的点，并同步更新邻接表以保持一致性"""
        if len(path) <= 5:
            return path, adj
        
        # 确定新路径（保留有效点）
        start_idx = max(0, len(path) - len(path) // 5)
        new_path = path[:start_idx]

        for idx in path[start_idx:]:
            if idx == trailing_idx or points[idx][0] <= intersection_x:
                new_path.append(idx)

        # 确保后缘点始终在新路径中
        if trailing_idx not in new_path:
            new_path.append(trailing_idx)
        
        # 1. 识别被移除的点
        removed_points = set(path) - set(new_path)
        
        # 2. 更新邻接表：移除被删除点的所有痕迹
        # 2.1 从邻接表中删除被移除点自身的条目
        for p in removed_points:
            if p in adj:
                del adj[p]
        
        # 2.2 从其他点的邻接列表中移除被删除点的引用
        for node in adj:
            adj[node] = [neigh for neigh in adj[node] if neigh not in removed_points]
        
        # 3. 修复新路径中连续点的邻接关系（确保相邻点互相关联）
        for i in range(len(new_path) - 1):
            current = new_path[i]
            next_p = new_path[i + 1]
            # 确保当前点的邻接列表包含下一个点
            if next_p not in adj.get(current, []):
                adj.setdefault(current, []).append(next_p)
            # 确保下一个点的邻接列表包含当前点
            if current not in adj.get(next_p, []):
                adj.setdefault(next_p, []).append(current)
        
        return new_path, adj

    @staticmethod
    def _calculate_angle(A, B, C):
        """计算三点形成的夹角（度）"""
        v1 = B - A
        v2 = C - B
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_theta = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _process_angle_correction(self, path, points):
        """处理相邻点连线角度≥40度的情况，仅计算分割点，不删除点"""
        
        # 计算分割点
        split_points = []
        for i in range(1, len(path) - 1):
            idx_A, idx_B, idx_C = path[i-1], path[i], path[i+1]
            angle = self._calculate_angle(points[idx_A], points[idx_B], points[idx_C])
            if angle >= 40:
                dist_AB = np.linalg.norm(points[idx_A] - points[idx_B])
                dist_BC = np.linalg.norm(points[idx_B] - points[idx_C])
                dist_AC = np.linalg.norm(points[idx_A] - points[idx_C])
                min_dist = min(dist_AB, dist_BC, dist_AC)
                if min_dist >= 0.001:
                    split_points.append(i)

        return sorted(list(set(split_points)))


    def _generate_segmented_bspline(self, points_list, splits, total_num_ratio):
        """根据分割点对曲线进行分段处理"""
        if len(points_list) < 2:
            return [points_list]
        
        split_indices = sorted(list(set([0] + splits + [len(points_list)-1])))
        segments = [points_list[start:end+1] for start, end in zip(split_indices[:-1], split_indices[1:])]

        processed_segments = []
        for seg in segments:
            num_ratio = round(total_num_ratio*len(seg) / len(points_list), 3)
            if len(seg) < 4:
                processed_segments.append(self._generate_linear_interpolation(seg, num_ratio))
            else:
                processed_segments.append(self._generate_bspline(seg, num_ratio))
               
        return processed_segments

    def _generate_linear_interpolation(self, points, num_ratio):
        """对短段进行线性插值"""
        if len(points) < 2:
            return points
        t_dense = np.linspace(0, 1, int(round(1000 * num_ratio)))
        x = points[:, 0]
        y = points[:, 1]
        
        x_interp = np.interp(t_dense, np.linspace(0, 1, len(x)), x)
        y_interp = np.interp(t_dense, np.linspace(0, 1, len(y)), y)
        
        linear_points = np.column_stack((x_interp, y_interp))
        linear_points[0], linear_points[-1] = points[0], points[-1]  # 约束端点
        return linear_points

    def _generate_smooth_curves(self, normalized_points, upper_path, lower_path, method, 
                               upper_splits=None, lower_splits=None):
        """生成平滑曲线"""
        upper_points = np.array([normalized_points[idx] for idx in reversed(upper_path)])
        lower_points = np.array([normalized_points[idx] for idx in lower_path])
            
        if method == 'bspline':
            upper_num_ratio =  round(len(upper_path) / (len(upper_path)+len(lower_path)), 3)
            lower_num_ratio =  round(len(lower_path) / (len(upper_path)+len(lower_path)), 3)
            if upper_splits and len(upper_splits) > 0:
                upper_splits_rev = [len(upper_path) - 1 - i for i in upper_splits]
                upper_segments = self._generate_segmented_bspline(upper_points, upper_splits_rev, upper_num_ratio)
            else:
                upper_segments = [self._generate_bspline(upper_points, upper_num_ratio)]
           
            if lower_splits and len(lower_splits) > 0:
                lower_segments = self._generate_segmented_bspline(lower_points, lower_splits, lower_num_ratio)
            else:
                lower_segments = [self._generate_bspline(lower_points, lower_num_ratio)]

            
            return upper_segments, lower_segments

    def _generate_bspline(self, points, num_ratio=1, degree=2, tightness=3.0, window_size=3, num_iterations=2):
        """生成B样条曲线，仅在点数足够时使用移动平均法对原始点进行光滑处理"""
        if len(points) < degree + 1:
            return points
        smoothed_points = self._moving_average(points, window_size, num_iterations)
        # 基于处理后的点生成B样条曲线
        t = self._calculate_adaptive_t(smoothed_points, tightness)
        x_spline = make_interp_spline(t, smoothed_points[:, 0], k=degree)
        y_spline = make_interp_spline(t, smoothed_points[:, 1], k=degree)
     
        t_dense = np.linspace(0, 1, int(round(1000 * num_ratio)))

        bspline_points = np.column_stack((x_spline(t_dense), y_spline(t_dense)))
  
        # 约束端点与原始点一致
        bspline_points[0], bspline_points[-1] = points[0], points[-1]

        return bspline_points

    def _moving_average(self, points, window_size, num_iterations=1):
        """
        对输入点集进行多次加权移动平均光滑处理，越近的点权重越大
        
        参数:
            points: 原始点集，形状为(n, 2)的numpy数组
            window_size: 移动窗口大小，必须为正整数
            num_iterations: 平滑处理的次数，必须为正整数，默认为1
            
        返回:
            光滑处理后的点集，形状与输入相同
        """
        # 确保参数有效
        if window_size < 1 or len(points) <= window_size or num_iterations < 1:
            return points.copy()
        
        # 复制原始点集，用于存储光滑后的结果
        smoothed = points.copy()
        num_points = len(points)  # 缓存点数量，避免重复计算
        half_window = window_size // 2  # 计算半窗口大小，处理边界情况
        
        # 进行指定次数的平滑处理
        for _ in range(num_iterations):
            # 每次迭代都基于上一次的平滑结果进行处理
            current_input = smoothed.copy()
            
            # 对每个点应用加权移动平均（跳过首尾点，保持边界不变）
            for i in range(num_points):
                # 跳过第一个和最后一个点（保持原始值）
                if i == 0 or i == num_points - 1:
                    continue
                
                # 确定窗口的起始和结束索引
                start = max(0, i - half_window)
                end = min(num_points, i + half_window + 1)
                
                # 获取窗口内的点索引
                indices = np.arange(start, end)
                
                # 计算每个点与当前点的距离（索引差的绝对值）
                distances = np.abs(indices - i)
                
                # 计算权重：距离越近权重越大，采用线性权重
                max_distance = np.max(distances) if len(distances) > 0 else 0
                weights = (max_distance + 2 - distances).astype(float)
                
                # 归一化权重，使权重之和为1
                weights /= np.sum(weights)
                
                # 计算加权平均值，基于当前输入（上一次平滑的结果）
                smoothed[i] = np.average(current_input[start:end], axis=0, weights=weights)
        
        return smoothed


    def _calculate_adaptive_t(self, points, tightness=2.0):
        """改进的自适应参数化计算"""
        if len(points) <= 1:
            return np.linspace(0, 1, len(points)) if len(points) > 0 else np.array([])
            
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.sum(distances) == 0:
            return np.linspace(0, 1, len(points))
            
        scaled_distances = distances **(1.0 / tightness)
        cumulative_dist = np.cumsum(scaled_distances)
        return np.concatenate([[0], cumulative_dist / cumulative_dist[-1]])


    def _uniform_sampling(self, upper_segments, lower_segments):
        """根据每段的点数分配采样数"""
        all_segments = upper_segments + lower_segments
        total_segs = len(all_segments)
        
        if total_segs == 0:
            return np.array([]), np.array([])
        
        seg_point_counts = [len(seg) for seg in all_segments]
        total_points = sum(seg_point_counts)
        
        if total_points == 0:
            return np.array([]), np.array([])
        
        # 分配采样点
        base_samples = []
        remaining = self.num_foil_points
        
        for count in seg_point_counts:
            samples = max(2, int(round(count / total_points * self.num_foil_points)))
            base_samples.append(samples)
            remaining -= samples
        
        # 处理剩余采样点
        i = 0
        while remaining != 0:
            adjust = 1 if remaining > 0 else -1
            if seg_point_counts[i] > np.mean(seg_point_counts) and (base_samples[i] + adjust) >= 2:
                base_samples[i] += adjust
                remaining -= adjust
            i = (i + 1) % total_segs
        
        # 执行采样
        upper_sampled = [self._sample_curve(seg, base_samples[i]) for i, seg in enumerate(upper_segments)]
        lower_sampled = [self._sample_curve(seg, base_samples[len(upper_segments) + i]) for i, seg in enumerate(lower_segments)]
        
        return (np.vstack(upper_sampled) if upper_sampled else np.array([]),
                np.vstack(lower_sampled) if lower_sampled else np.array([]))

    def _sample_curve(self, points, target_points):
        """对单条曲线进行均匀采样"""
        if len(points) < 2:
            return points
        target_points = max(2, target_points)
        points = np.asarray(points)
        
        target_indices = np.linspace(0.0, 1.0, target_points) * (len(points) - 1)
        idx = np.clip(np.floor(target_indices).astype(int), 0, len(points) - 2)
        t_interp = target_indices - idx
        
        starts = points[idx]
        ends = points[idx + 1]
        return starts + t_interp[:, np.newaxis] * (ends - starts)
    
    def _construct_xfoil_points(self, sampled_upper, sampled_lower, trailing_edge):
        """构建XFOIL格式的点列表（逆时针排列）"""
        xfoil_points = [(round(trailing_edge[0], 8), round(trailing_edge[1], 8))]
        
        # 添加上表面点（从后缘到前缘）
        xfoil_points.extend((round(p[0], 8), round(p[1], 8)) for p in sampled_upper[1:])
        
        # 添加下表面点（从前缘到后缘）
        xfoil_points.extend((round(p[0], 8), round(p[1], 8)) for p in sampled_lower[1:])
        
        if len(xfoil_points) < 4:
            raise ValueError("生成的XFOIL点列表点数不足")
        
        return xfoil_points

    # ------------------------------
    # 导出相关方法（通用化处理）
    # ------------------------------
    def _export_points(self, points, blade_idx, section_idx, suffix=""):
        """通用点导出工具"""
        try:
            os.makedirs(self.workdir, exist_ok=True)
            suffix_part = f"_{suffix}" if suffix else ""
            filename = os.path.join(
                self.workdir, f"airfoil_blade{blade_idx+1}_section{section_idx+1}{suffix_part}.dat"
            )
            with open(filename, 'w') as f:
                f.write(f"Airfoil Blade {blade_idx+1} Section {section_idx+1}\n")
                for point in points:
                    f.write(f"{point[0]:.8f} {point[1]:.8f}\n")
            return True, filename
        except Exception as e:
            return False, str(e)

    def export_single_airfoil(self, blade_idx, section_idx):
        """导出单个翼型文件"""
        try:
            xfoil_points = self.convert_to_xfoil_format(blade_idx, section_idx)
            return self._export_points(xfoil_points, blade_idx, section_idx)
        except Exception as e:
            return False, str(e)
    
    def _get_all_exported(self, export_func):
        """通用批量导出工具"""
        export_tasks = [(i, j) for i, sections in enumerate(self.blade_sections) 
                       for j in range(len(sections))]
        results = [export_func(*args) for args in export_tasks]
        
        total_exported = sum(1 for success, _ in results if success)
        return total_exported, len(results) - total_exported, self.workdir

    def get_all_airfoils(self):
        """导出所有翼型文件"""
        return self._get_all_exported(self.export_single_airfoil)

    def export_single_raw_points(self, blade_idx, section_idx):
        """导出单个叶片截面的原始归一化点"""
        try:
            # 复用共用逻辑获取路径数据
            section, normalized_points, upper_path, lower_path, _ = self._get_normalized_paths(blade_idx, section_idx)
            
            # 构建原始点列表
            raw_points = [normalized_points[idx] for idx in reversed(upper_path)]
            raw_points.extend(normalized_points[idx] for idx in lower_path[1:])          
            
            return self._export_points(raw_points, blade_idx, section_idx, suffix="raw")
        except Exception as e:
            return False, str(e)

    def get_all_raw_airfoils(self):
        """导出所有叶片截面的原始点文件"""
        return self._get_all_exported(self.export_single_raw_points)