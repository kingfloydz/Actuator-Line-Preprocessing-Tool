import os
import sys
import time
import logging
import numpy as np
import multiprocessing
from typing import Dict, List, Any

# 业务模块导入
from parameter_manager import ParameterManager
from geometry import GeometryAnalyzer, run_geometry_analysis
from airfoil_generator import AirfoilGenerator
from airfoil_aerodynamics import AirfoilAerodynamics
from xfoil_analyzer import XfoilAnalyzer
from viterna_extrapolation import AirfoilAerodynamicEstimator
from split_blades import BladeAnalyzer  # 导入BladeAnalyzer


def setup_logger(log_file_path: str) -> logging.Logger:
    """配置日志记录器（文件+控制台输出）"""
    logger = logging.getLogger('bladeprocess')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main():
    if multiprocessing.current_process().name != "MainProcess":
        return

    logger = None

    try:

        # 1. 初始化参数管理器（临时，获取原始模型文件和轴向参数）
        temp_param_manager = ParameterManager(initial_setup=True)
        temp_param_manager.clear_work_directory()     
        # 获取从parameter.dat解析的轴向方向
        axial_direction = temp_param_manager.axial_direction
        
        # 2. 使用BladeAnalyzer分析原始模型并分割出单个桨叶
        logger = setup_logger(os.path.join(temp_param_manager.program_dir, "process_log.txt"))
        logger.info("开始使用BladeAnalyzer分析模型...")
        
        # 创建BladeAnalyzer实例时传入轴向参数
        analyzer = BladeAnalyzer(
            file_path=temp_param_manager.original_model_file,
            axial_direction=axial_direction
        )
        analysis_result = analyzer.run_analysis(temp_param_manager.processed_model_prefix)
        
        # 3. 创建正式参数管理器，传入BladeAnalyzer的分析结果
        param_manager = ParameterManager(
            blade_analysis=analysis_result,
            processed_model_path=f"{temp_param_manager.processed_model_prefix}_blade_0.stl"
        )

        # 4. 日志设置
        model_file = param_manager.model_files[0] if param_manager.model_files else "未知模型"
        log_file = os.path.join(param_manager.program_dir, "process_log.txt")
        logger = setup_logger(log_file)
        # logger.info(f"开始处理model文件: {model_file}")

        # 5. 几何分析（使用封装函数）
        start_time = time.time()
        geometry = run_geometry_analysis(param_manager)
        logger.info(f"叶片几何分析完成，耗时: {time.time() - start_time:.2f}秒")
      
        # 6. 翼型文件生成
        start_time = time.time()
        generator = AirfoilGenerator(param_manager, geometry)
        total_exported, total_skipped, output_dir = generator.get_all_airfoils()
        logger.info(f"翼型生成至: {output_dir}，耗时: {time.time() - start_time:.2f}秒，生成 {total_exported} 个，跳过 {total_skipped} 个")

        # 7. 气动参数计算
        start_time = time.time()
        aerodynamics = AirfoilAerodynamics(param_manager, geometry)
        aerodynamics.calculate_reynolds_number()
        logger.info(f"气动属性计算完成，耗时: {time.time() - start_time:.2f}秒")

        # 8. XFOIL分析（优化并行处理）
        start_time = time.time()
        xfoil_analyzer = XfoilAnalyzer(param_manager, geometry)
        xfoil_results = xfoil_analyzer.run_xfoil_analysis()  
        logger.info(f"XFOIL分析完成，耗时: {time.time() - start_time:.2f}秒")

        #9.攻角外推
        start_time = time.time()
        extrapolator = AirfoilAerodynamicEstimator(logger, param_manager, geometry)
        extrapolator.export_results(xfoil_results)
        logger.info(f"攻角外推完成，耗时: {time.time() - start_time:.2f}秒")        

        # 9. 处理对称情况

        logger.info(f"分析完成！结果路径: {param_manager.program_dir}")
        logger.info(f"日志文件: {log_file}")
        #param_manager.clear_work_directory()

        return
    except Exception as e:
        error_msg = f"执行出错: {str(e)}"
        print(error_msg)
        if logger:
            logger.error(error_msg, exc_info=True)
        else:
            logging.basicConfig(filename='bladeprocess_error.log', level=logging.ERROR)
            logging.error(error_msg, exc_info=True)
        sys.exit(1)
        return


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
