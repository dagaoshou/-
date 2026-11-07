"""
为整个文件提供统一路径
"""
import os

def get_project_path()->str:
    """
    获取项目根目录
    :return: 项目根目录
    """
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    project_dir = os.path.dirname(current_dir)
    return project_dir

def get_abs_path(relative_path:str)->str:
    """
    获取绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_path = get_project_path()
    return os.path.join(project_path, relative_path)


if __name__ == '__main__':
    print(get_abs_path("data/config.yaml"))
