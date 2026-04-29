"""CoLLM 项目的统一入口。

这个文件只负责“转发命令”，真正的训练和测试逻辑仍然放在：
- train/train_all.py：完整训练 small、large、fuzzy、reflection
- eval_test.py：加载权重，在官方测试集上评估并画图

这样做的好处是：新手可以从 main.py 开始用项目，但代码结构仍然保持清楚。
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_python_script(script_path, extra_args):
    """用当前 Python 解释器运行另一个脚本。

    例如你当前用的是环境里的 python.exe，那么这里也会继续使用同一个
    python.exe，避免出现“训练用一个环境、测试用另一个环境”的问题。
    """
    command = [sys.executable, str(script_path), *extra_args]
    print("Running:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def print_help():
    """打印 main.py 自己的简短帮助。

    更详细的参数说明可以看：
    python main.py train --help
    python main.py eval --help
    """
    print(
        """usage: python main.py {train,eval} [args...]

Unified entry for the CoLLM reproduction project.

commands:
  train    forward to train/train_all.py
  eval     forward to eval_test.py

examples:
  python main.py train --subset FD001 --save-dir train --stages all
  python main.py eval --subset FD001 --model-dir train --save-dir results_test
"""
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        raise SystemExit(0)

    command = sys.argv[1]
    extra_args = sys.argv[2:]

    if command == "train":
        script = ROOT / "train" / "train_all.py"
    elif command == "eval":
        script = ROOT / "eval_test.py"
    else:
        print(f"Unknown command: {command}\n")
        print_help()
        raise SystemExit(2)

    raise SystemExit(run_python_script(script, extra_args))


if __name__ == "__main__":
    main()
