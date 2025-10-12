import os
import subprocess
import tarfile
from pathlib import Path
import sys
from tqdm import tqdm
import time
import threading
import logging
from datetime import datetime

# === 用户需要修改的部分 ===
DATASET_ROOT = "/storage/fdunphome/zhangjingzong/resnet-upc/resnet_upc_dataset_upload"  # 你的数据集目录
HF_REPO_SSH = "git@hf.co:datasets/qiuyu20030108/resnet_upc_dataset_upload"  # 你的HF仓库ssh地址
REPO_DIR = "/storage/fdunphome/zhangjingzong/resnet-upc/resnet_upc_dataset_upload_tar"  # 本地clone的目录
LOG_FILE = "uploaded.log"  # 已上传记录
DETAILED_LOG = "upload_detailed.log"  # 详细日志文件
MAX_FILE_SIZE_MB = 10000000000  # 超过此大小使用Git LFS
MAX_RETRIES = 3  # 最大重试次数
GIT_PUSH_TIMEOUT = 3600  # Git push超时时间（秒），默认1小时

# ======================================

# 设置详细日志
def setup_logging():
    """设置详细日志记录"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(DETAILED_LOG, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def run_cmd_with_logging(cmd, cwd=None, capture_output=False, timeout=None):
    """执行命令并记录详细日志"""
    logger.info(f"执行命令: {cmd}")
    logger.info(f"工作目录: {cwd}")
    
    print(f"[CMD] {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, cwd=cwd, 
                                  capture_output=True, text=True, timeout=timeout)
        else:
            # 对于长时间运行的命令，实时输出并记录到日志
            process = subprocess.Popen(
                cmd, shell=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    print(line)  # 实时输出到控制台
                    logger.info(f"命令输出: {line}")
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
            
            result = process
            result.stdout = '\n'.join(output_lines)
        
        if result.returncode != 0:
            error_msg = f"命令执行失败 (退出码: {result.returncode}): {cmd}"
            if hasattr(result, 'stderr') and result.stderr:
                error_msg += f"\n错误输出: {result.stderr.strip()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"命令执行成功: {cmd}")
        return result.stdout.strip() if hasattr(result, 'stdout') and result.stdout else None
        
    except subprocess.TimeoutExpired:
        error_msg = f"命令超时 ({timeout}秒): {cmd}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"命令执行异常: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def validate_config():
    """验证配置"""
    logger.info("开始验证配置")
    print("[INFO] 验证配置...")
    
    dataset_path = Path(DATASET_ROOT)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {DATASET_ROOT}")
    
    logger.info(f"数据集目录验证通过: {DATASET_ROOT}")
    print(f"[OK] 数据集目录存在: {DATASET_ROOT}")
    
    # 检查数据集结构
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"数据集目录为空: {DATASET_ROOT}")
    
    total_dirs = 0
    for task_dir in subdirs:
        person_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
        total_dirs += len(person_dirs)
    
    logger.info(f"发现 {len(subdirs)} 个任务目录，共 {total_dirs} 个人员目录")
    print(f"[INFO] 发现 {len(subdirs)} 个任务目录，共 {total_dirs} 个人员目录")


def check_git_lfs():
    """检查并安装Git LFS"""
    try:
        run_cmd_with_logging("git lfs version", capture_output=True)
        logger.info("Git LFS 已安装")
        print("[OK] Git LFS 已安装")
    except RuntimeError:
        logger.warning("Git LFS 未安装，正在尝试安装")
        print("[WARNING] Git LFS 未安装，正在尝试安装...")
        try:
            run_cmd_with_logging("git lfs install")
            logger.info("Git LFS 安装成功")
            print("[OK] Git LFS 安装成功")
        except RuntimeError:
            logger.error("Git LFS 安装失败")
            print("[ERROR] Git LFS 安装失败，请手动安装")
            sys.exit(1)


def check_ssh_connection():
    """检查SSH连接"""
    print("[INFO] 检查SSH连接...")
    try:
        # 测试SSH连接到Hugging Face
        result = subprocess.run("ssh -T git@hf.co", shell=True, 
                              capture_output=True, text=True, timeout=10)
        if "successfully authenticated" in result.stderr.lower():
            print("[OK] SSH连接正常")
        else:
            print("[WARNING] SSH连接可能有问题，但继续执行")
            print(f"SSH响应: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("[WARNING] SSH连接超时，但继续执行")
    except Exception as e:
        print(f"[WARNING] SSH连接检查失败: {e}，但继续执行")


def ensure_repo():
    """clone仓库（如果不存在）"""
    repo_path = Path(REPO_DIR)
    if not repo_path.exists():
        logger.info(f"克隆仓库到 {REPO_DIR}")
        print(f"[INFO] 克隆仓库到 {REPO_DIR}")
        run_cmd_with_logging(f"git clone {HF_REPO_SSH} {REPO_DIR}")
    else:
        # 检查是否为有效的git仓库
        if not (repo_path / ".git").exists():
            raise FileNotFoundError(f"目录存在但不是有效的git仓库: {REPO_DIR}")
        
        # 拉取最新更改
        logger.info("更新本地仓库")
        print("[INFO] 更新本地仓库")
        try:
            run_cmd_with_logging("git pull", cwd=REPO_DIR)
        except RuntimeError:
            logger.warning("git pull 失败，继续执行")
            print("[WARNING] git pull 失败，继续执行")
    
    # 验证远程仓库配置
    try:
        remote_url = run_cmd_with_logging("git remote get-url origin", cwd=REPO_DIR, capture_output=True)
        logger.info(f"远程仓库: {remote_url}")
        print(f"[INFO] 远程仓库: {remote_url}")
    except RuntimeError:
        logger.warning("无法获取远程仓库信息")
        print("[WARNING] 无法获取远程仓库信息")


def get_uploaded():
    """读取已上传的记录"""
    if not Path(LOG_FILE).exists():
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_uploaded(name):
    """写入已上传的记录"""
    with open(LOG_FILE, "a") as f:
        f.write(name + "\n")
    logger.info(f"记录已上传: {name}")
    print(f"[LOG] 记录已上传: {name}")


def count_files(directory):
    """计算目录中的文件数量"""
    return sum(1 for _ in Path(directory).rglob('*') if _.is_file())


def compress_dir_with_progress(src_dir, out_file):
    """带进度条的压缩功能"""
    out_path = Path(out_file)
    if out_path.exists():
        print(f"[SKIP] 压缩包已存在: {out_file}")
        return
    
    print(f"[TAR] 正在压缩 {src_dir} -> {out_file}")
    
    # 计算总文件数
    total_files = count_files(src_dir)
    print(f"[INFO] 发现 {total_files} 个文件")
    
    if total_files == 0:
        print(f"[WARNING] 目录为空: {src_dir}")
        return
    
    processed = 0
    with tarfile.open(out_file, "w:gz") as tar:
        with tqdm(total=total_files, desc="压缩进度", unit="files") as pbar:
            for file_path in Path(src_dir).rglob('*'):
                if file_path.is_file():
                    # 计算相对路径，保持目录结构
                    arcname = file_path.relative_to(Path(src_dir).parent)
                    tar.add(file_path, arcname=arcname)
                    processed += 1
                    pbar.update(1)
    
    # 检查文件大小
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] 压缩完成: {out_file} ({size_mb:.1f} MB)")
    
    return size_mb


def configure_git_lfs_for_ssh():
    """配置Git LFS使用SSH传输"""
    try:
        # 配置Git LFS使用SSH传输而不是HTTPS
        logger.info("配置Git LFS使用SSH传输")
        
        # 方法1: 设置LFS传输方式
        run_cmd_with_logging("git config lfs.transfer.maxretries 3", cwd=REPO_DIR)
        
        # 方法2: 尝试设置SSH传输URL（如果Hugging Face支持）
        try:
            remote_url = run_cmd_with_logging("git remote get-url origin", cwd=REPO_DIR, capture_output=True)
            if "git@hf.co" in remote_url:
                # 设置LFS使用相同的SSH URL
                lfs_url = remote_url.replace(".git", ".git/info/lfs")
                run_cmd_with_logging(f'git config lfs.url "{lfs_url}"', cwd=REPO_DIR)
                logger.info(f"设置LFS URL: {lfs_url}")
        except:
            logger.warning("无法设置LFS SSH URL，使用默认配置")
            
        print("[OK] Git LFS SSH配置完成")
        return True
    except Exception as e:
        logger.error(f"Git LFS SSH配置失败: {e}")
        print(f"[WARNING] Git LFS SSH配置失败: {e}")
        return False


def setup_git_lfs_if_needed(tar_path, size_mb):
    """如果文件较大，设置Git LFS追踪"""
    filename = os.path.basename(tar_path)
    if size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"文件 {filename} 大小 {size_mb:.1f}MB，使用Git LFS")
        print(f"[INFO] 文件 {filename} 大小 {size_mb:.1f}MB，使用Git LFS")
        print(f"[WARNING] Git LFS可能仍使用HTTPS，这是Git LFS的限制")
        try:
            run_cmd_with_logging(f"git lfs track \"{filename}\"", cwd=REPO_DIR)
            return True
        except RuntimeError:
            logger.warning("Git LFS 设置失败，继续尝试普通上传")
            print(f"[WARNING] Git LFS 设置失败，继续尝试普通上传")
            return False
    return False


def split_large_file(tar_path, chunk_size_mb=50):
    """将大文件分割成小块"""
    filename = os.path.basename(tar_path)
    file_size_mb = tar_path.stat().st_size / (1024 * 1024)
    
    if file_size_mb <= MAX_FILE_SIZE_MB:
        return [tar_path]  # 不需要分割
    
    logger.info(f"分割大文件 {filename} ({file_size_mb:.1f}MB)")
    print(f"[SPLIT] 分割大文件 {filename} ({file_size_mb:.1f}MB)")
    
    chunk_size = chunk_size_mb * 1024 * 1024  # 转换为字节
    chunks = []
    
    with open(tar_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            chunk_filename = f"{filename}.part{chunk_num:03d}"
            chunk_path = tar_path.parent / chunk_filename
            
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            chunks.append(chunk_path)
            chunk_num += 1
    
    # 创建重组脚本
    script_name = f"reassemble_{filename}.sh"
    script_path = tar_path.parent / script_name
    
    with open(script_path, 'w') as script:
        script.write("#!/bin/bash\n")
        script.write(f"# 重组文件: {filename}\n")
        script.write(f"cat")
        for i in range(len(chunks)):
            script.write(f" {filename}.part{i:03d}")
        script.write(f" > {filename}\n")
        script.write(f"# 删除分块文件\n")
        for i in range(len(chunks)):
            script.write(f"rm {filename}.part{i:03d}\n")
    
    chunks.append(script_path)  # 添加重组脚本到上传列表
    
    # 删除原始大文件
    tar_path.unlink()
    
    logger.info(f"文件分割完成，生成 {len(chunks)-1} 个分块和1个重组脚本")
    print(f"[OK] 分割完成: {len(chunks)-1} 个分块 + 重组脚本")
    
    return chunks


def monitor_git_push_progress(process, filename):
    """监控git push进度的后台线程"""
    start_time = time.time()
    last_output_time = start_time
    
    while process.poll() is None:
        current_time = time.time()
        elapsed = current_time - start_time
        since_last_output = current_time - last_output_time
        
        if since_last_output > 30:  # 30秒没输出就显示一次状态
            logger.info(f"[PROGRESS] {filename} 上传中... 已用时 {elapsed:.0f} 秒")
            print(f"[PROGRESS] {filename} 上传中... 已用时 {elapsed:.0f} 秒")
            last_output_time = current_time
        
        time.sleep(10)  # 每10秒检查一次


def push_file_with_detailed_logging(tar_path, size_mb=None):
    """带详细日志的文件推送"""
    filename = os.path.basename(tar_path)
    logger.info(f"开始上传文件: {filename}, 大小: {size_mb:.1f}MB")
    
    # 如果文件太大，使用分割策略而不是Git LFS
    files_to_upload = []
    if size_mb and size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"文件 {filename} 过大({size_mb:.1f}MB)，使用分割策略")
        print(f"[INFO] 文件过大({size_mb:.1f}MB)，分割后上传以避免Git LFS的HTTPS问题")
        try:
            files_to_upload = split_large_file(tar_path, chunk_size_mb=50)
        except Exception as e:
            logger.error(f"文件分割失败: {e}")
            print(f"[ERROR] 文件分割失败: {e}")
            return False
    else:
        files_to_upload = [tar_path]
    
    # 逐个上传文件
    for file_path in files_to_upload:
        file_name = os.path.basename(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"第 {attempt + 1}/{MAX_RETRIES} 次上传尝试: {file_name}")
                print(f"[PUSH] 尝试上传 {file_name} ({file_size_mb:.1f}MB) (第 {attempt + 1}/{MAX_RETRIES} 次)")
                
                # Git add (不使用Git LFS)
                logger.info(f"执行 git add: {file_name}")
                run_cmd_with_logging(f"git add \"{file_name}\"", cwd=REPO_DIR)
                
                # Git commit
                commit_msg = f"Add {file_name} ({file_size_mb:.1f}MB)"
                logger.info(f"执行 git commit: {commit_msg}")
                run_cmd_with_logging(f"git commit -m \"{commit_msg}\"", cwd=REPO_DIR)
                
                # Git push - 纯SSH
                logger.info(f"开始 git push: {file_name}")
                push_cmd = "git push origin main"
                logger.info(f"执行命令: {push_cmd}")
                
                start_time = time.time()
                try:
                    run_cmd_with_logging(push_cmd, cwd=REPO_DIR, timeout=GIT_PUSH_TIMEOUT)
                    elapsed_time = time.time() - start_time
                    logger.info(f"git push 完成，用时: {elapsed_time:.1f} 秒")
                    print(f"[OK] 上传成功: {file_name} (用时 {elapsed_time:.1f} 秒)")
                    break  # 成功，跳出重试循环
                except RuntimeError as e:
                    elapsed_time = time.time() - start_time
                    if "超时" in str(e):
                        logger.error(f"git push 超时，用时: {elapsed_time:.1f} 秒")
                        print(f"[ERROR] 上传超时 ({elapsed_time:.1f}s): {file_name}")
                    else:
                        logger.error(f"git push 失败，用时: {elapsed_time:.1f} 秒，错误: {e}")
                        print(f"[ERROR] 上传失败: {e}")
                    raise e
                    
            except RuntimeError as e:
                logger.error(f"第 {attempt + 1} 次上传失败: {file_name}, 错误: {e}")
                print(f"[ERROR] 第 {attempt + 1} 次上传失败: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 5
                    logger.info(f"等待 {wait_time} 秒后重试")
                    print(f"[RETRY] {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
                    # 重置git状态
                    try:
                        logger.info("重置git状态")
                        run_cmd_with_logging("git reset HEAD~1", cwd=REPO_DIR)
                    except:
                        logger.warning("git reset 失败，继续重试")
                        pass
                else:
                    logger.error(f"文件 {file_name} 最终上传失败")
                    print(f"[FAIL] 上传 {file_name} 最终失败")
                    return False
        else:
            # 如果某个文件上传失败，整体失败
            return False
    
    # 所有文件都成功上传
    return True


def estimate_total_work():
    """估算总工作量"""
    dataset_path = Path(DATASET_ROOT)
    total_dirs = 0
    
    for task_dir in dataset_path.iterdir():
        if not task_dir.is_dir():
            continue
        for person_dir in task_dir.iterdir():
            if person_dir.is_dir():
                total_dirs += 1
    
    return total_dirs


def main():
    print("=== Hugging Face 数据集上传工具 ===")
    print(f"数据集目录: {DATASET_ROOT}")
    print(f"HF仓库: {HF_REPO_SSH}")
    print(f"本地目录: {REPO_DIR}")
    print("=" * 50)
    
    try:
        # 配置验证
        validate_config()
        check_git_lfs()
        check_ssh_connection()
        ensure_repo()
        
        # 获取已上传记录
        uploaded = get_uploaded()
        print(f"[INFO] 已上传 {len(uploaded)} 个文件")
        
        # 估算工作量
        total_work = estimate_total_work()
        print(f"[INFO] 预计需要处理 {total_work} 个目录")
        
        # 开始处理
        dataset_root = Path(DATASET_ROOT)
        processed = 0
        failed = 0
        
        for task_dir in dataset_root.iterdir():
            if not task_dir.is_dir():
                continue

            for person_dir in task_dir.iterdir():
                if not person_dir.is_dir():
                    continue

                tar_name = f"{task_dir.name}_{person_dir.name}.tar.gz"
                tar_path = Path(REPO_DIR) / tar_name

                print(f"\n[PROCESS] {processed + 1}/{total_work} - {tar_name}")

                if tar_name in uploaded:
                    print(f"[SKIP] 已上传: {tar_name}")
                    processed += 1
                    continue

                # 压缩
                try:
                    size_mb = compress_dir_with_progress(person_dir, tar_path)
                    if size_mb is None:  # 空目录或压缩失败
                        print(f"[SKIP] 跳过空目录: {person_dir}")
                        processed += 1
                        continue
                except Exception as e:
                    print(f"[ERROR] 压缩失败: {e}")
                    failed += 1
                    continue

                # 上传
                if push_file_with_detailed_logging(tar_path, size_mb):
                    mark_uploaded(tar_name)
                    processed += 1
                else:
                    failed += 1
                    print(f"[FAIL] 请手动处理: {tar_name}")

        # 总结
        print(f"\n=== 处理完成 ===")
        print(f"成功处理: {processed}")
        print(f"失败: {failed}")
        print(f"总计: {processed + failed}")
        
        if failed > 0:
            print(f"\n[WARNING] 有 {failed} 个文件处理失败，请检查日志并手动处理")
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，程序退出")
        print("[INFO] 已处理的文件已记录在日志中，可以重新运行继续")
    except Exception as e:
        print(f"\n[FATAL] 程序错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()