import numpy as np
import logging

logger = logging.getLogger('bladeprocess.xfoil')


class AirfoilAerodynamics:
    """气动参数计算器，负责计算叶片各截面的气动特性"""
    
    def __init__(self, param_manager, geometry):
        """初始化函数：直接从ParameterManager获取所有参数"""
        self.param_manager = param_manager
        self.geometry = geometry
        self.processed = False
        self._initialize_constants()
        
    def _initialize_constants(self):
        """初始化并缓存常用常量"""
        self.centroid = np.array(self.param_manager.centroid)
        self.rotor_speed = self.param_manager.rotor_speed
        self.air_density = self.param_manager.air_density
        self.air_viscosity = self.param_manager.air_viscosity
        self.inflow_velocity = np.array(self.param_manager.inflow_velocity)
        
    def calculate_reynolds_number(self):
        """计算并存储所有截面的雷诺数和切向速度"""
        
        # 直接遍历叶片和截面
        for i, blade in enumerate(self.geometry.blade_sections):
            for j, section in enumerate(blade):
                
                # 计算切向速度
                position = np.array(section['position'])
                span_direction = np.array(section['span_direction'])
                tangential_direction = np.array(section['tangential_direction'])
                chord_length = section['chord_length']
                
                position_vector = position - self.centroid
                radius = np.abs(np.dot(position_vector, span_direction))
                tangential_velocity = self.rotor_speed * radius * tangential_direction
                
                # 计算雷诺数
                flow_velocity = np.array(self.inflow_velocity)
                relative_speed = np.linalg.norm(flow_velocity - tangential_velocity)
                reynolds = (self.air_density * relative_speed * chord_length) / self.air_viscosity if relative_speed > 1e-10 else 0.0
                
                # 更新截面数据
                self.geometry.blade_sections[i][j].update({
                    'reynolds': reynolds,
                    'tangential_velocity': tangential_velocity             
                })
            
        logger.info(f"成功计算雷诺数")